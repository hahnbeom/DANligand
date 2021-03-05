import torch

import dgl
from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber

# # Defining a model
class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, 
             num_layers, 
             l0_in_features,
             l1_in_features,
             num_degrees,
             num_channels,
             edge_features,
             div, 
             n_heads,
             chkpoint,
             pooling,
             **kwargs):
        super().__init__()

        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.edge_features = edge_features
        self.div = div
        self.n_heads = n_heads
        self.num_degrees = num_degrees
        self.pooling = pooling
        self.chkpoint = chkpoint

        # Linear projection layers for 
        self.linear1 = nn.Linear(l0_in_features, l0_in_features)
        self.drop1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(l0_in_features, num_channels)

        # Define fibers
        inputf = [(self.num_channels,0),(l1_in_features,1)]
        self.fibers = {
            'in': Fiber(2, structure=inputf),
            'mid': Fiber(self.num_degrees, self.num_channels),
            'out': Fiber(1, self.num_degrees*self.num_channels)}

        # Build graph convolution net
        blocks = self._build_gcn(self.fibers, 1)
        
        self.Gblock, self.PAblock, self.Wblock = blocks
        
    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(
                fin, 
                fibers['mid'], 
                edge_dim=self.edge_features, 
                div=self.div, 
                n_heads=self.n_heads))
            
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_features))
                
        # per-atm layers, applied 
        PAblock = []
        #print("PAblock size?", self.fibers['out'].n_features)
        PAblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        PAblock.append(nn.Dropout(0.1)) #ADDED
        PAblock.append(nn.ReLU(inplace=True))
        PAblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        # weights layer
        Wblock = []
        Wblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        Wblock.append(nn.Dropout(0.1)) #ADDED
        Wblock.append(nn.ReLU(inplace=True))
        Wblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(PAblock), nn.ModuleList(Wblock)

    def forward(self, G, idx):
        from torch.utils.checkpoint import checkpoint
        def runlayer(layer, G, r, basis):
            def custom_forward(*h):
                hd = {str(i):h_i for i,h_i in enumerate(h)}
                hd = layer( hd, G=G, r=r, basis=basis )
                h = tuple(hd[str(i)] for i in range(len(hd)))
                return (h)

            return custom_forward

        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # Pass l0 features through linear layers to condense to #channels
        l0 = F.elu(self.linear1(G.ndata['0'].squeeze()))
        l0 = self.drop1(l0)
        l0 = self.linear2(l0).unsqueeze(2)

        h = [l0, G.ndata['x'].requires_grad_(True)]
        # after linear projection (linear1,2): (NxNchannelx1)
        #print("shape1",h[0].shape)

        if (self.chkpoint):
            for layer in self.Gblock:
                h = checkpoint(
                    runlayer(layer, G, r, basis), *h
                )
            h = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h)}
        else:
            h = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h)}
            for layer in self.Gblock:
                h = layer(h, G=G, r=r, basis=basis)

        # after Gblock: (NxMx1); M = ndeg x nchannel
        #print("shape2",h['0'].shape)
        
        idx = torch.transpose(idx,0,1)

        # local path: per-node prediction
        g = h['0']
        g = torch.transpose(g,1,2)
        for layer in self.PAblock:
            g = layer(g)
        #
        #print("ligidx?", idx.shape, g.shape, g[:,0,:])
        g = torch.matmul(idx,g[:,:,0])
        g = torch.transpose(g,0,1)
        
        ## old global path
        #for layer in self.POOLblock:
        #    h = layer(h, G=G, r=r, basis=basis)
        # after pool block
        #print("shape3",h.shape) #(Mx1)
        #for layer in self.FCblock:
        #    h = layer(h)

        # new, global "weight" path
        w = h['0']
        w = torch.transpose(w,1,2)
        for layer in self.Wblock:
            w = layer(w)
            
        w = torch.matmul(idx,w[:,:,0])
        w = torch.transpose(w,0,1)

        h = torch.mean(w*g)
        
        return h, g

