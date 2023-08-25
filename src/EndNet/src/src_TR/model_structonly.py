import torch
import torch.nn as nn
import numpy as np
import time

from src.other_utils import to_dense_batch, make_batch_vec
import torch.utils.checkpoint as checkpoint

from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber
from src.src_TR2.trigon_2 import TriangleProteinToCompound_v2
from src.src_TR2.trigon_2 import TriangleSelfAttentionRowWise
from src.src_TR2.trigon_2 import Transition, get_pair_dis_one_hot

class TrigonModule(nn.Module):
    def __init__(self,
                 n_trigonometry_module_stack,
                 m=16,
                 c=32,
                 d=32,
                 dropout=0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout)
        
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.c = c
        self.d = d
        self.m = m
        
        self.tranistion = Transition(embedding_channels=m, n=4)

        self.Wrs = nn.Linear(d,d)
        self.Wls = nn.Linear(d,d)
        self.linear = nn.Linear(m, 1)
        
        self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=m, c=c) for _ in range(n_trigonometry_module_stack)])
        self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=m, c=c) for _ in range(n_trigonometry_module_stack)])

    def forward(self, Grec, Glig, hs_rec, hs_lig, keyidx, use_checkpoint, drop_out=False):
        # then process features
        hs_rec = self.Wrs(hs_rec)
        hs_lig = self.Wls(hs_lig)

        # make batched keyidx
        Kmax = max([idx.shape[0] for idx in keyidx])
        Nmax = max([idx.shape[1] for idx in keyidx])
        key_batched = torch.zeros((Glig.batch_num_nodes().shape[0],Kmax,Nmax)).to(Grec.device) #b x K x j
        for i,idx in enumerate(keyidx):
            key_batched[i,:idx.shape[0],:idx.shape[1]] = idx

        # prepare features for trigonometry attention
        recxyz = Grec.ndata['x'].squeeze().float()
        ligxyz = Glig.ndata['x'].squeeze().float()
        batchvec_rec = make_batch_vec(Grec.batch_num_nodes()).to(Grec.device)
        batchvec_lig = make_batch_vec(Glig.batch_num_nodes()).to(Grec.device)
        r_coords_batched, _ = to_dense_batch(recxyz, batchvec_rec)
        l_coords_batched, _ = to_dense_batch(ligxyz, batchvec_lig)

        hs_rec_batched, hs_rec_mask = to_dense_batch(hs_rec, batchvec_rec)
        hs_lig_batched, hs_lig_mask = to_dense_batch(hs_lig, batchvec_lig)

        hs_lig_batched = torch.einsum('bkj,bjd->bkd',key_batched,hs_lig_batched)
        hs_lig_mask_batched = torch.einsum('bkj,bj->bk',key_batched,hs_lig_mask.float()).bool()
        l_coords_batched = torch.einsum('bkj,bjc->bkc',key_batched,l_coords_batched)

        z = torch.einsum('bnd,bmd->bnmd', hs_rec_batched, hs_lig_batched )
        z_mask = torch.einsum('bn,bm->bnm',hs_rec_mask, hs_lig_mask_batched )

        rec_pair = get_pair_dis_one_hot(r_coords_batched, bin_size=2, bin_min=-1, num_classes=self.d).float()
        lig_pair = get_pair_dis_one_hot(l_coords_batched, bin_size=2, bin_min=-1, num_classes=self.d).float()

        #print(z.shape, hs_rec.shape, hs_lig.shape)
        # trigonometry part
        for i_module in range(self.n_trigonometry_module_stack):
            if use_checkpoint:
                zadd = checkpoint.checkpoint(self.protein_to_compound_list[i_module], z, rec_pair, lig_pair, z_mask.unsqueeze(-1))
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                zadd = checkpoint.checkpoint(self.triangle_self_attention_list[i_module], z, z_mask)
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
            else:
                zadd = self.protein_to_compound_list[i_module](z, rec_pair, lig_pair, z_mask.unsqueeze(-1))
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                zadd = self.triangle_self_attention_list[i_module](z, z_mask)
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                
            z = self.tranistion(z)
            
        return z, z_mask, hs_lig_batched, hs_lig_mask_batched

class HalfModel(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers_lig=2,
                 num_layers_rec=2,
                 num_channels=32,
                 num_degrees=3,
                 n_heads_se3=4,
                 div=4,
                 l0_in_features_lig=15,
                 l0_in_features_rec=32,
                 l0_out_features_lig=32,
                 l1_in_features=0,
                 l1_out_features=0, 
                 num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                 m=16,
                 c=128,
                 n_trigonometry_module_stack = 5,
                 classification_mode = 'tank',
                 dropout=0.1,
                 bias=True):
        super().__init__()

        self.l1_in_features = l1_in_features
        self.dropout = nn.Dropout(p=dropout)
        self.classification_mode = classification_mode
 
        fiber_in = Fiber({0: l0_in_features_lig}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features_lig, 1: l1_in_features})

        # processing ligands
        self.se3_lig = SE3Transformer(
            num_layers   = num_layers_lig,
            num_heads    = n_heads_se3,
            channels_div = div,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: l0_out_features_lig}),
            fiber_edge=Fiber({0: num_edge_features}),
        )
        
        fiber_in = Fiber({0: l0_in_features_rec}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features_rec, 1: l1_in_features})
        
        # trigonometry related
        self.trigon_module =  TrigonModule(n_trigonometry_module_stack,
                                           m=m,
                                           c=c,
                                           d=l0_out_features_lig,
                                           dropout=dropout
        )

        # last processing -- classification
        if self.classification_mode == 'ligand':
            self.lapool = nn.AdaptiveMaxPool2d((4,c)) #nn.Linear(l0_out_features,c)
            self.rapool = nn.AdaptiveMaxPool2d((100,c))
            self.lhpool = nn.AdaptiveMaxPool2d((20,c))
            self.linear_pre_aff = nn.Linear(c,1)
            self.linear_for_aff = nn.Linear(124,1)
                
        elif self.classification_mode == 'tank':
            # attention matrix to affinity 
            self.linear1 = nn.Linear(m, 1)
            self.linear2 = nn.Linear(m, 1)
            self.linear1a = nn.Linear(m, m)
            self.linear2a = nn.Linear(m, m)
            self.bias = nn.Parameter(torch.ones(1))
            self.leaky = nn.LeakyReLU()

        # last processing -- structure
        self.lastlinearlayer = nn.Linear(m,1)

    def forward(self, Grec, Glig, hs_rec, keyidx, use_checkpoint=False, drop_out=True):
        t0 = time.time()
        if self.l1_in_features > 0:
            h_rec = self.dropoutlayer( h_rec )

        t1 = time.time()
        node_features_lig = {'0':Glig.ndata['attr'][:,:,None].float()}
        edge_features_lig = {'0':Glig.edata['attr'][:,:,None].float()}
        
        hs_lig = self.se3_lig(Glig, node_features_lig, edge_features_lig)['0'] # M x d x 1
        t2 = time.time()

        hs_rec = torch.squeeze(hs_rec) # N x d
        hs_lig = torch.squeeze(hs_lig) # M x d
        
        #z: B x N x K x c(?)
        z, z_mask,hs_lig_batched, lig_mask_batched = self.trigon_module(Grec, Glig,
                                                                        hs_rec, hs_lig,
                                                                        keyidx, use_checkpoint, drop_out=drop_out)
        t4 = time.time()

        return z, z_mask, hs_lig_batched, lig_mask_batched
        
