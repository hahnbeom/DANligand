import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT
from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber

class SimpleDecoder(nn.Module):
    def __init__(self, d, n_out ):
        super().__init__()
        
        self.d = d
        layers = []
        layers.append(nn.Linear(d,d,bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(d))
        layers.append(nn.Linear(d,n_out,bias=True))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MyModel(nn.Module):
    def __init__(self, modeltype,
                 num_layers=2,
                 num_channels=32, num_degrees=3, n_heads=4, div=4,
                 num_node_feats=32, n_out_emb=32,
                 l1_in_feats=0, l1_out_emb=8,
                 num_edge_feats=4, ntypes=15,
                 drop_out=0.0,
                 bias=False):
        super().__init__()

        self.modeltype = modeltype
        self.LayerIn = nn.Linear(num_node_feats,num_channels,bias=bias)

        if modeltype == 'SE3':
            fiber_in = Fiber({0: num_channels}) if l1_in_feats == 0 \
                else Fiber({0: num_channels, 1: l1_in_feats})
            fiber_out = Fiber({0: n_out_emb}) if l1_out_feats == 0 \
                else Fiber({0: n_out_emb, 1: l1_out_feats})

            self.Encoder = SE3Transformer(
                num_layers   = num_layers,
                num_heads    = n_heads,
                channels_div = div,
                fiber_in=fiber_in,
                fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
                fiber_out=fiber_out,
                fiber_edge=Fiber({0: num_edge_feats}),
            )

            
        elif modeltype == 'GAT':
            self.Encoder = GAT( num_layers = num_layers, 
                                in_channels = num_channels, hidden_channels = num_channels,
                                out_channels = n_out_emb,
                                dropout = 0.0
            )
        
        self.dropout = nn.Dropout( drop_out )
        self.Decoder = SimpleDecoder( n_out_emb, ntypes )
        
    def forward(self, G):
        # Encoding part
        if self.modeltype == 'SE3':
            node_features = {"0": G.ndata["attr"][:,:,None].float(), 'x': G.ndata['x'].float()}
            edge_features = {"0": G.edata["attr"][:,:,None].float()}
            node_in = {}
            for key in node_features:
                node_in[key] = self.dropout(node_features[key])

            node_in['0'] = self.LayerIn(torch.transpose(node_in['0'],1,2))
            node_in['0'] = torch.transpose(node_in['0'],2,1)

            hs = self.Encoder(G, node_in, edge_features)['0']
            
        if self.modeltype == 'GAT':
            node_features = G.ndata["attr"].float()
            edge_features = G.edata["attr"].float()
            u,v = G.edges()

            edge_index = torch.zeros((2,len(u))).long().to(G.device)
            edge_index[0,:] = u
            edge_index[1,:] = v
            
            node_in = self.dropout(self.LayerIn(node_features))
            
            hs = self.Encoder(node_in, edge_index=edge_index, edge_attr = edge_features)

        # Decoding part
        # to pool idx comes the last of each subgraph
        poolidx = torch.zeros(len(G.batch_num_nodes())).long()
        b = 0
        for i,n in enumerate(G.batch_num_nodes()):
            poolidx[i] = b+n-1
            b += n

        poolidx = torch.eye(G.number_of_nodes())[poolidx].to(G.device)

        xs = torch.matmul(poolidx, hs)
        xs = self.dropout(xs)
        xs = self.Decoder(xs)
        
        return xs # B x ntype
