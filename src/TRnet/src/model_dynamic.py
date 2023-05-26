import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers_lig=2,
                 num_layers_rec=2,
                 num_channels=32, num_degrees=3, n_heads_se3=4, div=4,
                 l0_in_features_lig=19,
                 l0_in_features_rec=18,
                 l0_out_features=32,
                 l1_in_features=0,
                 l1_out_features=0, #???
                 embedding_channels=16,
                 c=128,
                 n_trigonometry_module_stack = 5,
                 num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                 dropout=0.2,
                 bias=True):
        super().__init__()

        # "d": self.l0_out_features
        d = l0_out_features
        
        self.d = d
        self.l1_in_features = l1_in_features
        self.scale = 1.0 # num_head #1.0/np.sqrt(float(d))

        fiber_in = Fiber({0: l0_in_features_lig}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features_lig, 1: l1_in_features})

        # processing ligands
        self.se3_lig = SE3Transformer(
            num_layers   = num_layers_lig,
            num_heads    = n_heads_se3,
            channels_div = div,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: d}), #, 1:l1_out_features}),
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
            fiber_out=Fiber({0: d} ), #1:l1_out_features}),
            fiber_edge=Fiber({0: 1}), #always just distance
        )
    
        # trigonometry related
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.dropout = nn.Dropout(p=dropout)
        self.tranistion = Transition(embedding_channels=embedding_channels, n=4)
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.lastlinearlayer = nn.Linear(embedding_channels,1)

        self.Wrs = nn.Linear(d,d)
        self.Wls = nn.Linear(d,d)
        self.linear = nn.Linear(embedding_channels, 1)

        self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])
        self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])
        
    def forward(self, Grec, Glig, labelidx, use_checkpoint=False, drop_out=False ):
        size1 = Grec.batch_num_nodes()
        size2 = Glig.batch_num_nodes()
            
        recxyz = Grec.ndata['x'].squeeze().float()
        ligxyz = Glig.ndata['x'].squeeze().float()

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
        
        hs_rec = self.se3_rec(Grec, node_features_rec, edge_features_rec)['0'] # N x d x 1
        hs_lig = self.se3_lig(Glig, node_features_lig, edge_features_lig)['0'] # M x d x 1

        ## input prep to trigonometry attention
        hs_rec = torch.squeeze(hs_rec) # N x d
        hs_lig = torch.squeeze(hs_lig) # M x d

        if torch.cuda.is_available():
            batchvec_rec = make_batch_vec(size1).cuda()
            batchvec_lig = make_batch_vec(size2).cuda()

        else:
            batchvec_rec = make_batch_vec(size1)
            batchvec_lig = make_batch_vec(size2)

        hs_rec = self.Wrs(hs_rec)
        hs_lig = self.Wls(hs_lig)
        
        label_to_batch = labelidx[0].transpose(1,0)

        for i in range(1,len(size1)):
            label_to_batch = torch.cat((label_to_batch,labelidx[i].transpose(1,0)),dim=0)

        label_batched, labelmask2 = to_dense_batch(label_to_batch, batchvec_lig)
        label_batched = label_batched.transpose(2,1) # b x K x j

        hs_rec_batched, hs_rec_mask = to_dense_batch(hs_rec, batchvec_rec)
        hs_lig_batched, hs_lig_mask = to_dense_batch(hs_lig, batchvec_lig)

        r_coords_batched, r_coords_mask = to_dense_batch(recxyz, batchvec_rec)
        l_coords_batched, l_coords_mask = to_dense_batch(ligxyz, batchvec_lig)

        # label batched: which atom is key atom
        hs_lig_batched = torch.einsum('bkj,bjd->bkd',label_batched,hs_lig_batched)
        hs_lig_mask = torch.einsum('bkj,bj->bk',label_batched,hs_lig_mask.float())
        l_coords_batched = torch.einsum('bkj,bjc->bkc',label_batched,l_coords_batched)

        hs_lig_mask = hs_lig_mask.bool()

        z = torch.einsum('bnd,bmd->bnmd', hs_rec_batched, hs_lig_batched )
        z_mask = torch.einsum('bn,bm->bnm', hs_rec_mask, hs_lig_mask )

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

        # final processing
        z = self.lastlinearlayer(z).squeeze(-1) #real17 ext15

        # z: b x N x K
        z = masked_softmax(self.scale*z, mask=z_mask, dim = 1) # softmax over Nrec

        Yrec_s = torch.einsum("bik,bic->bkc",z,r_coords_batched) # "Weighted sum":  N x K , N x 3 -> K x 3

        Yrec_s = [l for l in Yrec_s]
        Yrec_s = torch.stack(Yrec_s,dim=0)
        
        #print(hs_lig_mask.shape, Yrec_s.shape)
        #Yrec_s = torch.einsum("bkc,bk->bkc", Yrec_s, hs_lig_mask)
        hs_lig_mask = hs_lig_mask[:,:,None].repeat(1,1,3)
        Yrec_s = Yrec_s * hs_lig_mask

        return Yrec_s, z #b x K x 3, b x N x K

