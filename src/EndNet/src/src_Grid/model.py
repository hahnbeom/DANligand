import torch
import torch.nn as nn

from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber

class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers_grid=2,
                 num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32,
                 l0_out_features=32,
                 l1_in_features=0, 
                 l1_out_features=0,
                 num_edge_features=32, ntypes=15,
                 dropout_rate=0.1,
                 bias=True):
        super().__init__()

        fiber_in = Fiber({0: l0_in_features}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features, 1: l1_in_features})

        self.se3 = SE3Transformer(
            num_layers   = num_layers_grid,
            num_heads    = 4,
            channels_div = 4,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: l0_out_features}), #1:l1_out_features}),
            fiber_edge=Fiber({0: num_edge_features}),
        )

        Cblock = []
        for i in range(ntypes):
            Cblock.append(nn.Linear(l0_out_features,1,bias=False)) 
            
        self.Cblock = nn.ModuleList(Cblock)
        self.dropoutlayer = nn.Dropout(dropout_rate)

    def forward(self, G, node_features, edge_features=None, drop_out=False):

        node_in, edge_in = {},{}
        for key in node_features:
            node_in[key] = node_features[key]
            if drop_out: node_in[key] = self.dropoutlayer(node_in[key])

        hs = self.se3(G, node_in, edge_features)

        hs0 = hs['0'].squeeze(2)
        if drop_out:
            hs0 = self.dropoutlayer(hs0)

        # hs0 as pre-FC embedding; N x num_channels
        cs = []
        for i,layer in enumerate(self.Cblock):
            c = layer(hs0)
            cs.append(c) # Nx2
        cs = torch.stack(cs,dim=0) #ntype x N x 2
        cs = cs.T.squeeze(0)

        return hs0, cs 
