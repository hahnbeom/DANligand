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
                 l1_in_features=0, l1_out_features=8,
                 num_edge_features=32, ntypes=15,
                 nGMM=1, 
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

        WOblock = [] # weighting block for orientation; per-type
        WCblock = [] # weighting block for category; per-type
        WBblock = [] # weighting block for bb position; per-type
        SBblock = [] # sigma from l0
        ABblock = [] # amplitude from l0
        Cblock = [] 
        WOblock.append(nn.Linear(l0_out_features,l1_out_features,bias=bias)) #sync multiplier
        WOblock.append(nn.Tanh()) #range -1~1
        
        WCblock.append(nn.Linear(l0_out_features,ntypes,bias=False))
        WCblock.append(nn.ReLU(inplace=True)) #guarantee >0
        
        WBblock.append(nn.Linear(l0_out_features,l1_out_features,bias=False)) #sync multiplier
        WBblock.append(nn.Tanh()) #range -1~1

        # sigma
        SBblock.append(nn.Linear(l0_out_features,nGMM,bias=True))
        SBblock.append(nn.ReLU(inplace=True)) #range
        
        ABblock.append(nn.Linear(l0_out_features,nGMM,bias=True))
        ABblock.append(nn.ReLU(inplace=True)) #range
        
        Cblock.append(nn.Linear(l0_out_features,10,bias=False)) #UNUSED currently
        Cblock.append(nn.ReLU(inplace=True)) #guarantee >0
        
        RblockY = [] # constant rotation block
        RblockB = [] # constant rotation block
        for i in range(ntypes):
            RblockY.append(nn.Linear(l1_out_features,1,bias=False)) #weight vector on each channel!
            RblockB.append(nn.Linear(l1_out_features,nGMM,bias=False)) #weight vector on each channel!
        
        self.WOblock = nn.ModuleList(WOblock) #unused
        self.WCblock = nn.ModuleList(WCblock)
        self.WBblock = nn.ModuleList(WBblock)
        self.SBblock = nn.ModuleList(SBblock)
        self.ABblock = nn.ModuleList(ABblock)
        self.Cblock = nn.ModuleList(Cblock)

        # initialize rot blocks
        self.Rblock = {'y':nn.ModuleList(RblockY), 'b':nn.ModuleList(RblockB)}
        #self.Rblock = {'y':RblockY, 'b':RblockB}

    def forward(self, G, node_features, edge_features=None):
        hs = self.se3(G, node_features, edge_features)

        # per-node weights for Orientation/Backbone/Category
        wo = hs['0'].squeeze(2)
        wb = hs['0'].squeeze(2)
        sig = hs['0'].squeeze(2)
        ampl = hs['0'].squeeze(2)
        wc = hs['0'].squeeze(2)
        c = hs['0'].squeeze(2)
        v = hs['1']
        for layer in self.WOblock: wo = layer(wo)
        for layer in self.WBblock: wb = layer(wb)
        for layer in self.SBblock: sig = layer(sig) #confidence
        for layer in self.ABblock: ampl = layer(ampl) #confidence
        #for layer in self.WCblock: wc = layer(wc)
        for layer in self.Cblock: c = layer(c)
        
        return wo,wb, c, v, sig, self.Rblock
