import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber
from src.src_TR.trigon_2 import TriangleProteinToCompound_v2, TriangleSelfAttentionRowWise, Transition, get_pair_dis_one_hot
from src.src_TR.other_utils import masked_softmax, to_dense_batch
from src.src_TR.myutils import make_batch_vec



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
                 K=4, # how many Y points
                 embedding_channels=32,
                 c=128,
                 n_trigonometry_module_stack =5,
                 num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                 dropout=0.1,
                 bias=True):
        super().__init__()

        # "d": self.l0_out_features
        d = l0_out_features

        self.l1_in_features = l1_in_features
        self.scale = 1.0 # num_head #1.0/np.sqrt(float(d))
        self.K = K
 
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
        self.dropout = nn.Dropout2d(p=dropout)
        self.tranistion = Transition(embedding_channels=embedding_channels, n=4)
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.lastlinearlayer = nn.Linear(embedding_channels,1)

        self.Wrs = nn.Linear(l0_in_features_rec,d)
        self.Wls = nn.Linear(d,d)
        self.linear = nn.Linear(embedding_channels, 1)



        self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])
        self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=embedding_channels) for _ in range(n_trigonometry_module_stack)])


        
    def forward(self, Grec, h_rec, Glig, labelidx):

        size1 = Grec.batch_num_nodes()
        size2 = Glig.batch_num_nodes()

        recxyz = Grec.ndata['x'].squeeze().float()
        ligxyz = Glig.ndata['x'].squeeze().float()
        # print(Grec)
        # print(Glig)

        # labelidx = torch.tensor(labelidx)
        node_features_lig = {'0':Glig.ndata['attr'][:,:,None].float()}#,'1':Glig.ndata['x'].float()}
        edge_features_lig = {'0':Glig.edata['attr'][:,:,None].float()}

        if self.l1_in_features > 0:
            node_features_lig['1'] = Glig.ndata['x'].float()

        hs_lig = self.se3_lig(Glig, node_features_lig, edge_features_lig)['0'] # M x d x 1

        hs_rec = torch.squeeze(h_rec) # N x d
        hs_lig = torch.squeeze(hs_lig) # M x d

        if torch.cuda.is_available():
            batchvec_rec = make_batch_vec(size1).cuda()
            batchvec_lig = make_batch_vec(size2).cuda()

        else:
            batchvec_rec = make_batch_vec(size1)
            batchvec_lig = make_batch_vec(size2)

        hs_rec = self.Wrs(hs_rec)
        hs_lig = self.Wls(hs_lig)

        # print(labelidx[0].shape)
        # print(labelidx)
        label_to_batch = labelidx[0].transpose(1,0)

        for i in range(1,len(size1)):
            label_to_batch = torch.cat((label_to_batch,labelidx[i].transpose(1,0)),dim=0)

        # print(batchvec_lig.is_cuda, batchvec_rec.is_cuda, label_to_batch.is_cuda)

        label_batched, label_mask = to_dense_batch(label_to_batch, batchvec_lig)
        label_batched = label_batched.transpose(2,1) # b x K x j

        # h_l = torch.matmul(idx1hot,h_l) # K x d
        # ligxyz = torch.einsum('bjk,bjkb->ikb',label_batched,ligxyz)

        hs_rec_batched, hs_rec_mask = to_dense_batch(hs_rec, batchvec_rec)
        hs_lig_batched, hs_lig_mask = to_dense_batch(hs_lig, batchvec_lig)

        r_coords_batched, r_coords_mask = to_dense_batch(recxyz, batchvec_rec)
        l_coords_batched, l_coords_mask = to_dense_batch(ligxyz, batchvec_lig)

        # print('hs_lig_mask',hs_lig_mask.shape)
        # print('l_coords_batched',l_coords_batched.shape)
        # print('label_batched',label_batched.shape)

        hs_lig_batched = torch.einsum('bkj,bjd->bkd',label_batched,hs_lig_batched)
        hs_lig_mask = torch.einsum('bkj,bj->bk',label_batched,hs_lig_mask.float())
        l_coords_batched = torch.einsum('bkj,bjc->bkc',label_batched,l_coords_batched)

        hs_lig_mask = hs_lig_mask.bool()

        z = torch.einsum('bnd,bmd->bnmd', hs_rec_batched, hs_lig_batched )
        z_mask = torch.einsum('bn,bm->bnm',hs_rec_mask, hs_lig_mask )

        rec_pair = get_pair_dis_one_hot(r_coords_batched, bin_size=2, bin_min=-1)
        lig_pair = get_pair_dis_one_hot(l_coords_batched, bin_size=2, bin_min=-1)
        
        rec_pair = rec_pair.float()
        lig_pair = lig_pair.float()
        

        for i_module in range(self.n_trigonometry_module_stack):
            z = z + self.dropout(self.protein_to_compound_list[i_module](z, rec_pair, lig_pair, z_mask.unsqueeze(-1)))
            z = z + self.dropout(self.triangle_self_attention_list[i_module](z, z_mask))
            z = self.tranistion(z)
            
        z = self.lastlinearlayer(z).squeeze(-1) #real17 ext15
        z = masked_softmax(self.scale*z, mask=z_mask, dim = 1)

        Yrec_s = torch.einsum("bij,bic->bjc",z,r_coords_batched) # "Weighted sum":  i x j , i x 3 -> j x 3

        Yrec_s = [l for l in Yrec_s]
        # print(Yrec_s)

        
        Yrec_s = torch.stack(Yrec_s,dim=0)

        #print(Yrec_s.shape)
        # exit()
        return Yrec_s, z #B x ? x ?

