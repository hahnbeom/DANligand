import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from src.other_utils import to_dense_batch
from src.myutils import make_batch_vec
import torch.utils.checkpoint as checkpoint
#from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias
#from equivariant_attention.modules import GConvSE3, GNormSE3
#from equivariant_attention.fibers import Fiber

from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber
from src.trigon_2 import TriangleProteinToCompound_v2
from src.trigon_2 import TriangleSelfAttentionRowWise
from src.trigon_2 import Transition
from src.trigon_2 import get_pair_dis_one_hot
from src.other_utils import MaskedSoftmax
from src.other_utils import masked_softmax

class TrigonModule(nn.Module):
    def __init__(self,
                 n_trigonometry_module_stack,
                 embedding_channels=16,
                 c=32,
                 d=32,
                 dropout=0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout)
        
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.c = c
        self.d = d
        self.embedding_channels = embedding_channels
        
        self.tranistion = Transition(embedding_channels=embedding_channels, n=4)
        #self.layernorm = nn.LayerNorm(embedding_channels)

        self.Wrs = nn.Linear(d,d)
        self.Wls = nn.Linear(d,d)
        self.linear = nn.Linear(embedding_channels, 1)
        
        self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])
        self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])

    def forward(self, Grec, Glig, hs_rec, hs_lig, keyidx, use_checkpoint, drop_out=False):
        size2 = Glig.batch_num_nodes()
        B = size2.shape[0]
        
        recxyz = Grec.ndata['x'].squeeze().float()
        ligxyz = Glig.ndata['x'].squeeze().float()

        batchvec_lig = make_batch_vec(size2).to(Grec.device)

        hs_rec = self.Wrs(hs_rec)
        hs_lig = self.Wls(hs_lig)

        Kmax = max([idx.shape[0] for idx in keyidx])
        Nmax = max([idx.shape[1] for idx in keyidx])
        key_batched = torch.zeros((len(size2),Kmax,Nmax)).to(Grec.device) #b x K x j
        for i,idx in enumerate(keyidx):
            key_batched[i,:idx.shape[0],:idx.shape[1]] = idx

        # just repeat instead
        #size1 = Grec.batch_num_nodes()
        #batchvec_rec = make_batch_vec(size1).to(Grec.device)
        #r_coords_batched, _ = to_dense_batch(recxyz, batchvec_rec)
        #hs_rec_batched, hs_rec_mask = to_dense_batch(hs_rec, batchvec_rec)

        r_coords_batched = recxyz[None,:,:].repeat((B,1,1))
        hs_rec_batched = hs_rec[None,:,:].repeat((B,1,1))
        hs_rec_mask = torch.ones((B,hs_rec.shape[0])).to(Grec.device)

        l_coords_batched, _ = to_dense_batch(ligxyz, batchvec_lig)
        hs_lig_batched, hs_lig_mask = to_dense_batch(hs_lig, batchvec_lig)

        hs_lig_batched = torch.einsum('bkj,bjd->bkd',key_batched,hs_lig_batched)
        hs_lig_mask = torch.einsum('bkj,bj->bk',key_batched,hs_lig_mask.float())
        l_coords_batched = torch.einsum('bkj,bjc->bkc',key_batched,l_coords_batched)

        hs_lig_mask = hs_lig_mask.bool()

        z = torch.einsum('bnd,bmd->bnmd', hs_rec_batched, hs_lig_batched )
        z_mask = torch.einsum('bn,bm->bnm',hs_rec_mask, hs_lig_mask )

        rec_pair = get_pair_dis_one_hot(r_coords_batched, bin_size=2, bin_min=-1, num_classes=self.d).float()
        lig_pair = get_pair_dis_one_hot(l_coords_batched, bin_size=2, bin_min=-1, num_classes=self.d).float()

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
            
        return z, z_mask, hs_lig_batched, hs_lig_mask

class HalfModel(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers_lig=2,
                 num_layers_rec=2,
                 num_channels=32, num_degrees=3, n_heads_se3=4, div=4,
                 l0_in_features_lig=19,
                 l0_in_features_rec=18,
                 l0_out_features=32,
                 l1_in_features=0,
                 l1_out_features=0, #???
                 num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                 K=4, # how many Y points
                 embedding_channels=16,
                 c=128,
                 n_trigonometry_module_stack = 5,
                 classification_mode = 'tank',
                 dropout=0.1,
                 bias=True):
        super().__init__()

        self.l1_in_features = l1_in_features
        self.scale = 1.0 # num_head #1.0/np.sqrt(float(d))
        self.K = K
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
            fiber_out=Fiber({0: l0_out_features}), #, 1:l1_out_features}),
            fiber_edge=Fiber({0: num_edge_features}),
        )
        
        fiber_in = Fiber({0: l0_in_features_rec}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features_rec, 1: l1_in_features})
        
        # processing receptor (==grids)
        self.se3_rec = SE3Transformer(
            num_layers   = num_layers_rec,
            num_heads    = n_heads_se3,
            channels_div = div,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: l0_out_features} ), #1:l1_out_features}),
            fiber_edge=Fiber({0: 1}), #always just distance
        )
    
        # trigonometry related
        self.trigon_module =  TrigonModule(n_trigonometry_module_stack,
                                           embedding_channels=embedding_channels,
                                           c=c,
                                           d=l0_out_features,
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
            self.linear1 = nn.Linear(embedding_channels, 1)
            self.linear2 = nn.Linear(embedding_channels,1)
            self.linear1a = nn.Linear(embedding_channels, embedding_channels)
            self.linear2a = nn.Linear(embedding_channels, embedding_channels)
            self.bias = nn.Parameter(torch.ones(1))
            self.leaky = nn.LeakyReLU()

        # last processing -- structure
        self.lastlinearlayer = nn.Linear(embedding_channels,1)

    def forward(self, Grec, Glig, keyidx, use_checkpoint=False, drop_out=True):
        t0 = time.time()
        B = Glig.batch_num_nodes().shape[0]
        
        node_features_rec = {'0':Grec.ndata['attr'][:,:,None].float()}#,'1':Grec.ndata['x'].float()}
        edge_features_rec = {'0':Grec.edata['attr'][:,:,None].float()}

        node_features_lig = {'0':Glig.ndata['attr'][:,:,None].float()}#,'1':Glig.ndata['x'].float()}
        edge_features_lig = {'0':Glig.edata['attr'][:,:,None].float()}

        if drop_out:
            node_features_rec['0'] = self.dropout(node_features_rec['0'])
            node_features_lig['0'] = self.dropout(node_features_lig['0'])
            
        if self.l1_in_features > 0:
            node_features_rec['1'] = Grec.ndata['x'].float()
            node_features_lig['1'] = Glig.ndata['x'].float()

        t1 = time.time()
        hs_rec = self.se3_rec(Grec, node_features_rec, edge_features_rec)['0'] # N x d x 1
        t2 = time.time()
        hs_lig = self.se3_lig(Glig, node_features_lig, edge_features_lig)['0'] # M x d x 1
        t3 = time.time()

        ## input prep to trigonometry attention
        hs_rec = torch.squeeze(hs_rec) # N x d
        hs_lig = torch.squeeze(hs_lig) # M x d
        
        #batchvec_rec = make_batch_vec(Grec.batch_num_nodes()).to(Grec.device)
        #hs_rec_batched, _ = to_dense_batch(hs_rec, batchvec_rec)
        
        hs_rec_batched = hs_rec[None,:,:].repeat((B,1,1))
        batchvec_lig = make_batch_vec(Glig.batch_num_nodes()).to(Glig.device)
        hs_lig_batched, _ = to_dense_batch(hs_lig, batchvec_lig)

        #z: B x N x K x c(?)
        z, z_mask,hs_lig_batched_k, hs_lig_mask_k = self.trigon_module(Grec, Glig, hs_rec, hs_lig, keyidx, use_checkpoint, drop_out=drop_out)
        t4 = time.time()

        # for classification
        if self.classification_mode == 'ligand':
            exp_z = torch.exp(z) 
            # soft alignment 
            # normalize each row of z for receptor counterpart
            zr_denom = exp_z.sum(axis=(-2)).unsqueeze(-2) # 1 x Nrec x 1 x c
            zr = torch.div(exp_z,zr_denom) # 1 x Nrec x K x c
            ra = zr*hs_lig_batched_k.unsqueeze(1) # 1 x Nrec x K x c
            ra = ra.sum(axis=-2) # 1 x Nrec x c

            # normalize each row of z for ligand counterpart
            zl_denom = exp_z.sum(axis=(-3)).unsqueeze(-3) # 1 x Nrec x 1 x c
            zl = torch.div(exp_z,zl_denom) # 1 x Nrec x K x c
            zl_t = torch.transpose(zl, 1, 2) # 1 x K x Nrec x c

            la = zl_t*hs_rec_batched.unsqueeze(1) # 1 x K x Nrec x numchannel
            la = la.sum(axis=-2) # 1 x K x numchannel
            la = self.lapool(la) # 1 x K x c

            # concat and then pool 
            lh = hs_lig_batched # 1 x Nlig x c

            ra = self.rapool(ra) # 1 x 100 x c
            lh = self.lhpool(lh) # 1 x 20 x c

            cat = torch.cat([ra,la,lh],dim=1) # 1 x 124 x c
            
            Aff = self.linear_pre_aff(cat).squeeze(-1) # 1 x 124
            Aff = self.linear_for_aff(Aff).squeeze(-1) # b x 1 

        elif self.classification_mode == 'tank':
            pair_energy = ( self.linear1(F.relu(self.linear1a(z)).sigmoid()) * \
                            self.linear2(F.relu(self.linear2a(z))) ).squeeze(-1) * z_mask
            affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1,-2)))) # "NK energy sum"
            Aff = affinity_pred # 1
            
        # for structureloss
        z = self.lastlinearlayer(z).squeeze(-1) #real17 ext15
        z = masked_softmax(self.scale*z, mask=z_mask, dim = 1)
        
        # final processing
        #batchvec_rec = make_batch_vec(Grec.batch_num_nodes()).to(Grec.device)
        # r_coords_batched: B x Nmax x 3
        #r_coords_batched, _ = to_dense_batch(Grec.ndata['x'].squeeze().float(), batchvec_rec) # time consuming!!
        x = Grec.ndata['x'].float().transpose(0,1)
        r_coords_batched = x.repeat((B,1,1))

        Yrec_s = torch.einsum("bij,bic->bjc",z,r_coords_batched) # "Weighted sum":  i x j , i x 3 -> j x 3
        Yrec_s = [l for l in Yrec_s]
        Yrec_s = torch.stack(Yrec_s,dim=0) # 1 x K x 3

        hs_lig_mask_k = hs_lig_mask_k[:,:,None].repeat(1,1,3) #1 x max(Nlig) x 3
        Yrec_s = Yrec_s * hs_lig_mask_k

        return Yrec_s, Aff, z #B x ? x ?

