import torch
import torch.nn as nn
import numpy as np

from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber

class SE3TransformerAutoEncoder(nn.Module):
    def __init__(self,
                 num_node_feats,
                 num_edge_feats,
                 num_channels=32,
                 n_layers_encoder = 3,
                 n_layers_decoder = 3,
                 latent_dim = 4,
                 scale = 1.0
    ):
        super().__init__()

        self.scale = scale
        self.input_layer = nn.Linear( num_node_feats, num_channels )
        
        #if latent_dim*2 < num_channels:
        #    mid_dim = latent_dim*2
        #else:
        mid_dim = num_channels

        # SE3 encoder
        self.encoder = SE3Transformer(
            num_layers   = n_layers_encoder,
            num_heads    = 4,
            channels_div = 4,
            #num_degrees  = 3,
            fiber_in = Fiber({0: num_node_feats}),
            fiber_hidden=Fiber.create( 3, num_channels ),
            fiber_out = Fiber({0: latent_dim, 1: latent_dim }),
            fiber_edge = Fiber({0: num_edge_feats}),
        )

        # SE3 encoder
        self.mid_layers0 = nn.ModuleList( [nn.Linear(latent_dim, mid_dim) ] )
        self.mid_layers1 = nn.ModuleList( [nn.Linear(latent_dim, mid_dim) ] )
        
        self.decoder = SE3Transformer(
            num_layers   = n_layers_decoder,
            num_heads    = 4,
            channels_div = 4,
            #num_degrees  = 3,
            fiber_in = Fiber({0: mid_dim, 1:mid_dim}),
            fiber_hidden=Fiber.create( 3, num_channels ),
            fiber_out = Fiber({1:1}),
            fiber_edge=Fiber({0:num_edge_feats}),
        )

    def forward(self, G):
        node_features = {"0": G.ndata["attr"][:,:,None].float()}
        edge_features = {"0": G.edata["attr"][:,:,None].float()}

         # encode time step
        z = self.encoder( G, node_features, edge_features )

        #print(z['1'].shape, z['1'][-5:,:])
        h0 = z['0'].squeeze()
        for layer in self.mid_layers0:
            h0 = layer(h0)

        h1 = z['1'].squeeze()
        h1 = h1.transpose(2,1)
        for layer in self.mid_layers1:
            h1 = layer(h1)
        h1 = h1.transpose(1,2)

        node_features = {"0": h0[:,:,None].float(), "1":h1[:,:].float()}
        h = self.decoder( G, node_features, edge_features )
        l1pred = h['1'].transpose(0,1)*self.scale # B x N x 3
        #print(l1pred.shape, l1pred[0,-5:,:])
        
        return l1pred

