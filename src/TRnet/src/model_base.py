import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias
#from equivariant_attention.modules import GConvSE3, GNormSE3
#from equivariant_attention.fibers import Fiber

from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber


class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers_lig=2,
                 num_layers_rec=2,
                 num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features_lig=19,
                 l0_in_features_rec=18,
                 l0_out_features=32,
                 l1_in_features=0,
                 l1_out_features=0, #???
                 K=4, # how many Y points
                 nattn=1,
                 num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                 dropout=0.1,
                 bias=True):
        super().__init__()

        # "d": self.l0_out_features
        d = l0_out_features

        self.l1_in_features = l1_in_features
        self.scale = 1.0 # num_head #1.0/np.sqrt(float(d))
        self.K = K
        self.nattn = nattn
        
        fiber_in = Fiber({0: l0_in_features_lig}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features_lig, 1: l1_in_features})

        # processing ligands
        self.se3_lig = SE3Transformer(
            num_layers   = num_layers_lig,
            num_heads    = n_heads,
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
            num_heads    = n_heads,
            channels_div = div,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: d} ), #1:l1_out_features}),
            fiber_edge=Fiber({0: 1}), #always just distance
        )
    
        # cross-attention related
        self.phi = nn.Linear( d, d )
        #self.phi = nn.ModuleList([])
        #self.W = nn.ModuleList([])
        #for k in range(K):
        #    self.phi.append(nn.Linear( d, d )) # should this be per-K dependent?

        self.Ascaler1 = nn.Parameter(torch.tensor(0.1))
        self.Ascaler2 = nn.Parameter(torch.tensor(0.1))


        
    def Xattention( self, h_r, h_l):
                
        for i_layer in range(self.nattn):
            # print(self.nattn)

            # h_r : N x d // h_l : K x d 

            dots = torch.einsum("id,kd->ki",h_r,h_l) # K x N

            A = nn.functional.softmax(self.scale*dots,dim=1)

            h_r = h_r + self.Ascaler1*torch.einsum("ij,id->jd", A, h_l)
            h_l = h_l + self.Ascaler2*torch.einsum("ij,jd->id", A, h_r)

    

        return A
        
    def forward(self, Grec, Glig, labelidx):

        node_features_rec = {'0':Grec.ndata['attr'][:,:,None].float()}#,'1':Grec.ndata['x'].float()}
        edge_features_rec = {'0':Grec.edata['attr'][:,:,None].float()}

        node_features_lig = {'0':Glig.ndata['attr'][:,:,None].float()}#,'1':Glig.ndata['x'].float()}
        edge_features_lig = {'0':Glig.edata['attr'][:,:,None].float()}

        if self.l1_in_features > 0:
            node_features_rec['1'] = Grec.ndata['x'].float()
            node_features_lig['1'] = Glig.ndata['x'].float()

        hs_rec = self.se3_rec(Grec, node_features_rec, edge_features_rec)['0'] # N x d x 1
        hs_lig = self.se3_lig(Glig, node_features_lig, edge_features_lig)['0'] # M x d x 1

        hs_rec = torch.squeeze(hs_rec) # N x d
        hs_lig = torch.squeeze(hs_lig) # M x d

        xyz_rec = Grec.ndata['x'].squeeze().float()

        size1 = Grec.batch_num_nodes()
        size2 = Glig.batch_num_nodes()
        
        A_s = []
        Yrec_s = []
        asum,bsum=0,0


        # caution: dimension can be smaller than should be if batch == 1
        # if len(labelidx) == 1: labelidx = [labelidx.unsqueeze(0)
                                           
        for a,b,idx1hot in zip(size1,size2,labelidx):

            h_r = hs_rec[asum:asum+a]
            x = xyz_rec[asum:asum+a]
            
            # pick key-part only
            h_l = hs_lig[bsum:bsum+b]

            if idx1hot.dim() == 1: 
                idx1hot = idx1hot.unsqueeze(0)

            h_l = torch.matmul(idx1hot,h_l) # K x d

            ## attention part
            #print(a, b, bsum, h_l.shape, idx1hot.shape)
            h_l = nn.functional.relu( self.phi(h_l) ) # M x d

            test1 = (h_r,h_l)

            A = self.Xattention( h_r, h_l )

            # print((h_r,h_l) == test1)

            ## maybe add some more here to update h_r,h_l??

            Yrec = torch.einsum("ki,il->kl",A,x) # "Weighted sum":  K x N, N x 3 -> k x 3
            A_s.append(A)
            Yrec_s.append(Yrec)
            asum += a
            bsum += b


        Yrec_s = torch.stack(Yrec_s,dim=0)
        #print(Yrec_s.shape)
        return Yrec_s, A_s #B x ? x ?

