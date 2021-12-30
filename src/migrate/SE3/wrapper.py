import torch
import torch.nn as nn

#from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias
#from equivariant_attention.modules import GConvSE3, GNormSE3
#from equivariant_attention.fibers import Fiber

from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber

class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers=2, num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32, l0_out_features=32,
                 l1_in_features=3, l1_out_features=2,
                 num_edge_features=32, ntypes=15):
        super().__init__()
        
        self.se3 = SE3Transformer(
            num_layers   = 2,
            num_heads    = 4,
            channels_div = 4,
            fiber_in=Fiber({0: l0_in_features}),
            fiber_hidden=Fiber({0: 32, 1:32, 2:32}),
            fiber_out=Fiber({0: 32, 1:l1_out_features}),
            fiber_edge=Fiber({0: num_edge_features}),
        )

        WOblock = [] # weighting block for orientation; per-type
        WCblock = [] # weighting block for category; per-type
        WBblock = [] # weighting block for bb position; per-type
        Cblock = [] 
        Rblock = [] # constant rotation block
        WOblock.append(nn.Linear(32,ntypes,bias=False))
        WOblock.append(nn.ReLU(inplace=True)) #guarantee >0
        
        WCblock.append(nn.Linear(32,ntypes,bias=True)
        WCblock.append(nn.ReLU(inplace=True)) #guarantee >0
        
        WBblock.append(nn.Linear(32,ntypes,bias=False))
        WBblock.append(nn.ReLU(inplace=True)) #guarantee >0
        
        Cblock.append(nn.Linear(32,10,bias=False))
        Cblock.append(nn.ReLU(inplace=True)) #guarantee >0
        for i in range(ntypes):
            Rblock.append(nn.Linear(3,3,bias=False)) #rotation matrix
        
        self.WOblock = nn.ModuleList(WOblock)
        self.WCblock = nn.ModuleList(WCblock)
        self.WBblock = nn.ModuleList(WBblock)
        self.Cblock = nn.ModuleList(Cblock)

        # initialize rot blocks
        self.Rblock = nn.ModuleList(Rblock)

    def forward(self, G, node_features, edge_features=None):
        hs = self.se3(G, node_features, edge_features)

        # per-node weights for Orientation/Backbone/Category
        wo = hs['0'].squeeze(2)
        wb = hs['0'].squeeze(2)
        wc = hs['0'].squeeze(2)
        c = hs['0'].squeeze(2)
        v = hs['1']
        for layer in self.WOblock: wo = layer(wo)
        for layer in self.WBblock: wb = layer(wb)
        #for layer in self.WCblock: wc = layer(wc)
        for layer in self.Cblock: c = layer(c)
        
        return wo, wb, c, v, self.Rblock
