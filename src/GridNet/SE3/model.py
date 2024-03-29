import torch
import torch.nn as nn

#from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias
#from equivariant_attention.modules import GConvSE3, GNormSE3
#from equivariant_attention.fibers import Fiber

from .se3_transformer.model import SE3Transformer
from .se3_transformer.model.fiber import Fiber

class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers=2, num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32, l0_out_features=32,
                 l1_in_features=0, l1_out_features=8,
                 num_edge_features=32, ntypes=15,
                 bias=True):
        super().__init__()

        fiber_in = Fiber({0: l0_in_features}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features, 1: l1_in_features})

        self.se3 = SE3Transformer(
            num_layers   = num_layers,
            num_heads    = 4,
            channels_div = 4,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: l0_out_features, 1:l1_out_features}),
            fiber_edge=Fiber({0: num_edge_features}),
        )

        WCblock = [] # weighting block for category; per-type
        
        WCblock.append(nn.Linear(l0_out_features,l0_out_features,bias=True))
        WCblock.append(nn.Linear(l0_out_features,ntypes,bias=False))
        WCblock.append(nn.ReLU(inplace=True)) #guarantee >0
        
        Cblock = [] 
        for i in range(ntypes):
            Cblock.append(nn.Linear(l0_out_features,ntypes,bias=False)) 
            
        self.WCblock = nn.ModuleList(WCblock)
        self.Cblock = nn.ModuleList(Cblock)

    def forward(self, G, node_features, edge_features=None):
        hs = self.se3(G, node_features, edge_features)

        wc = hs['0'].squeeze(2)
        c = hs['0'].squeeze(2)
        for layer in self.WCblock: wc = layer(wc)
        
        cs = []
        for i,layer in enumerate(self.Cblock):
            c = hs['0'].squeeze(2) #Nx32
            c = torch.sigmoid(layer(c))
            cs.append(c) # Nx2
        cs = torch.stack(cs,dim=0) #ntype x N x 2
            
        return cs 
