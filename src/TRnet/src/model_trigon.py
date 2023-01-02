import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.other_utils import to_dense_batch

#from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias
#from equivariant_attention.modules import GConvSE3, GNormSE3
#from equivariant_attention.fibers import Fiber

from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber
from src.trigon import TriangleProteinToCompound_v2
from src.trigon import TriangleSelfAttentionRowWise
from src.trigon import Transition
from src.trigon import get_pair_dis_one_hot
# from src.trigon import get_pair_dis_one_hot


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
                 embedding_channels=16,
                 c=128,
                 n_trigonometry_module_stack = 5,
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

        self.Wrs = nn.Linear(d,d)
        self.Wls = nn.Linear(d,d)



        self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])
        self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=embedding_channels) for _ in range(n_trigonometry_module_stack)])


        
    def forward(self, Grec, Glig, labelidx):

        size1 = Grec.batch_num_nodes()
        size2 = Glig.batch_num_nodes()

        recxyz = Grec.ndata['x']
        ligxyz = Glig.ndata['x']
        
        # print(Grec)
        # print(Glig)
        # print(labelidx)

        node_features_rec = {'0':Grec.ndata['attr'][:,:,None].float()}#,'1':Grec.ndata['x'].float()}
        edge_features_rec = {'0':Grec.edata['attr'][:,:,None].float()}

        node_features_lig = {'0':Glig.ndata['attr'][:,:,None].float()}#,'1':Glig.ndata['x'].float()}
        edge_features_lig = {'0':Glig.edata['attr'][:,:,None].float()}

        if self.l1_in_features > 0:
            node_features_rec['1'] = Grec.ndata['x'].float()
            node_features_lig['1'] = Glig.ndata['x'].float()

        hs_rec = self.se3_rec(Grec, node_features_rec, edge_features_rec)['0'] # N x d x 1
        hs_lig = self.se3_lig(Glig, node_features_lig, edge_features_lig)['0'] # M x d x 1

        # print('hs_rec : ',hs_rec.shape)

        # print('hs_lig : ',hs_lig.shape)

        hs_rec = torch.squeeze(hs_rec) # N x d
        hs_lig = torch.squeeze(hs_lig) # M x d
        

        xyz_rec = Grec.ndata['x'].squeeze().float()


        # xyz_lig = Glig.ndata['x'].float()
        
        A_s = []
        Yrec_s = []
        asum,bsum=0,0


        # caution: dimension can be smaller than should be if batch == 1
        # if len(labelidx) == 1: labelidx = [labelidx.unsqueeze(0)

                                           
        for a,b,idx1hot in zip(size1,size2,labelidx):

            x = xyz_rec[asum:asum+a]

            
            h_r = hs_rec[asum:asum+a]
            x_r = recxyz[asum:asum+a]
            
            # pick key-part only
            h_l = hs_lig[bsum:bsum+b]
            x_l = ligxyz[bsum:bsum+b]
            

            if idx1hot.dim() == 1: 
                idx1hot = idx1hot.unsqueeze(0)


            h_l = torch.matmul(idx1hot,h_l) # K x d
            x_l = torch.einsum('ij,jkb->ikb',idx1hot,x_l)

            
            h_r = self.Wrs(h_r)
            h_l = self.Wls(h_l)

            h_r_batched, h_r_mask = to_dense_batch(h_r)
            h_l_batched, h_l_mask = to_dense_batch(h_l)

            r_coords_batched, r_coords_mask = to_dense_batch(x_r)
            l_coords_batched, l_coords_mask = to_dense_batch(x_l)

            z = torch.einsum('bnd,bmd->bnmd', h_r_batched, h_l_batched )
            z = torch.sigmoid(z)
            z = z.float()
            # print(z)

            z_mask = torch.einsum('bn,bm->bnm',h_r_mask, h_l_mask )
            # print(z_mask.shape)
            z_mask = z_mask.float()
            # print(z_mask)

            # print('z, zmask okay')

            rec_pair = get_pair_dis_one_hot(r_coords_batched, bin_size=2, bin_min=-1)
            lig_pair = get_pair_dis_one_hot(l_coords_batched, bin_size=2, bin_min=-1)
            
            rec_pair = rec_pair.float()
            lig_pair = lig_pair.float()
            
            # print('rec, lig pair okay')

            for i_module in range(self.n_trigonometry_module_stack):
                # print('z1', z)
                z = z + self.dropout(self.protein_to_compound_list[i_module](z, rec_pair, lig_pair, z_mask.unsqueeze(-1)))
                # print('z2',z)
                z = z + self.dropout(self.triangle_self_attention_list[i_module](z, z_mask))
                # print('z3',z)
                z = self.tranistion(z)
                # print('z4',z)
                # print('updating z okay')


            z = self.lastlinearlayer(z).squeeze()



            ## maybe add some more here to update h_r,h_l??


            Yrec = torch.einsum("ik,il->kl",z,x) # "Weighted sum":  N x K , N x 3 -> k x 3

            # print('z', z)            
            # print('x' , x)
            # print('Yrec',Yrec)

            A_s.append(z)
            Yrec_s.append(Yrec)
            asum += a
            bsum += b

        Yrec_s = torch.stack(Yrec_s,dim=0)
        #print(Yrec_s.shape)
        return Yrec_s, A_s #B x ? x ?

