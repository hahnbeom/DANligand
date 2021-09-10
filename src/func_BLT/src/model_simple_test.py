import numpy as np
np.random.seed(42)
import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import dgl.function as dglF
import random
random.seed(42)

import torch
torch.manual_seed(42)
from torch import nn
from torch.nn import functional as F
from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber
from . import myutils, motif
#import Transformer
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# # Defining a model
class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    # 28: aa-type, 1: q or SA, 65: atype, 32: atm-embedding from bnds
    def __init__(self,
                 num_layers     = [2],
                 l0_in_features = [65+2], #bnd,res,atm graph
                 l1_in_features = [0],
                 num_degrees    = 2,
                 num_channels   = [1],
                 edge_features  = [2], #distance (1-hot) (&bnd, optional)
                 div            = [2],
                 n_heads        = [2],
                 pooling        = "avg",
                 chkpoint       = True,
                 modeltype      = 'comm',
                 #nntypes        = ("SE3T","SE3T","SE3T"),
                 nntypes        = ["TFN"],
                 drop_out       = 0.0,
                 outtype        = 'category', #or [list,'binary']
                 learn_orientation = False,
                 **kwargs):
        super().__init__()

        # Build the network
        self.num_layers = num_layers
        self.l1_in_features = l1_in_features
        self.num_channels = num_channels
        self.edge_features = edge_features
        self.div = div
        self.n_heads = n_heads
        self.num_degrees = num_degrees
        self.pooling = pooling
        self.chkpoint = chkpoint
        self.modeltype = modeltype
        self.nntypes = nntypes
        self.learn_OR = learn_orientation

        if outtype == 'category':
            self.noutbin = len(motif.MOTIFS)
            self.outtype = [-1]
        elif outtype == 'binary':
            self.noutbin = 2
            self.outtype = [-1]
        elif isinstance(outtype,list):
            self.noutbin = 2 #len(outtype)
            self.outtype = outtype

        # Linear projection layers for each
        self.linear1_bnd = nn.Linear(l0_in_features[0], l0_in_features[0])
        self.linear2_bnd = nn.Linear(l0_in_features[0], num_channels[0])

        # shared
        self.drop = nn.Dropout(drop_out)

        ## Build graph convolution net -- Define fibers, etc.
        # Bond
        self.fibers_bnd = {
            'in': Fiber(1, self.num_channels[0]),
            'mid': Fiber(self.num_degrees, self.num_channels[0]),
            'out': Fiber(1, self.num_channels[0])}
        self.Gblock_bnd = self._build_SE3gcn(self.fibers_bnd, 1, 0, self.nntypes[0])
        print(f"Gblock_bnd: {self.Gblock_bnd}")

    # out_dim unused
    def _build_SE3gcn(self, fibers, out_dim, g_index, nntype='SE3T'):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers[g_index]):
            Gblock.append( GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_features[g_index]) )
        #Gblock.append(GConvSE3(fibers['mid'], fibers['out'],self_interaction=True,
        #                       edge_dim=self.edge_features[g_index]))

        return nn.ModuleList(Gblock)

    def forward(self, G_bnd):
        # Pass l0 features through linear layers to condense to #channels
        if len(G_bnd.ndata['0'].squeeze().shape) != 2:
            return torch.tensor([0.0]), None, None

        l0_bnd = F.elu(self.linear1_bnd(G_bnd.ndata['0'].squeeze()))
        l0_bnd = self.drop(l0_bnd)
        l0_bnd = self.linear2_bnd(l0_bnd).unsqueeze(2)
        print("l0_bnd.shape:", l0_bnd.shape)
        print("l0_bnd[:2,:5]:", l0_bnd[:2, :5,0])
        h_bnd = {'0':l0_bnd}

        # Compute equivariant weight basis from relative positions
        basis_bnd, r_bnd = get_basis_and_r(G_bnd, self.num_degrees-1)
        #import ipdb;ipdb.set_trace()

        print("started forward pass on G_bnd")
        for layer in self.Gblock_bnd:
            print(f"\tlayer: {layer}")
            h_bnd = layer(h_bnd, G=G_bnd, r=r_bnd, basis=basis_bnd)
        print("finished forward pass on G_bnd")

        return h_bnd

class SE3TransformerJ(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_degrees: int=3,
                 edge_dim: int=1, div: float=1, n_heads: int=4, **kwargs):
        super().__init__()

        # num_layers - number of layers in the model
        # atom_feature_size - total number of node features in l0, l1, l2, l3 types
        # num_channels - number of features per degree (l0, l1, l2) for the neural network
        # num_degrees - maximum number of degrees - 1, so if num_degrees=3 then SE3 will use type 0, 1, 2 features
        # edge_dim - number of features on edges, these can only be scalar values, l0 features at this moment
        # div - a way to reduce num_channels, div has to divide num_channels
        # n_heads - number of attention heads, n_heads has to divide num_chanels


        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.n_heads = n_heads

        # define the fiber structure for the input and output - very important!
        # 'in', Fiber(2, structure=[(32,0), (16,1), (12,2)]) means that the input has 32 type 0 features, 16 type 1 features, 12 type 2 features
        # 'out', Fiber(2, structure=[(64,0), (16,1)]) means that the output has 64 type 0 features, 16 type 1 features
        #self.fibers = {'in': Fiber(2, structure=[(21,0)]),
        #               'mid': Fiber(num_degrees, self.num_channels),
        #               'out': Fiber(2, structure=[(1,1)])}
        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(2, 1),
                       'out': Fiber(1, 1)}

        blocks = self._build_gcn(self.fibers)
        self.Gblock = blocks

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            #add one residual layer and one normalization layer in the loop
            #Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
            #                      div=self.div, n_heads=self.n_heads))
            #Gblock.append(GNormSE3(fibers['mid']))
            Gblock.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            fin = fibers['mid']
        # final layer to output requires fibers['out'] shape
        #Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))
        return nn.ModuleList(Gblock)

    def forward(self, G):

        # Takes in G, but uses only G.edata['d'] to construct spherical harmonic basis functions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        #Define features
        #type_0_features = G.ndata['0'][:,:,None].float() #[num_nodes, num_features, 1]
        type_0_features = G.ndata['0'][:, :, None] #[num_nodes, num_features, 1]
        print(f"type_0_features.shape : {type_0_features.shape}")

        # Construct input to the SE3 layers
        # h = {'0': x, '1': y, '2': z} takes in type 0, 1, 2 features
        # no batch size, graphs can be batched to make more nodes
        # x.shape = [num_nodes, num_features, 1] #the last 1 is important!
        # y.shape = [num_nodes, num_featues, 3]
        # z.shape = [num_nodes, num_featues, 5]
        h = {'0': type_0_features}
        for i, layer in enumerate(self.Gblock):
            h = layer(h, G=G, r=r, basis=basis)
            if "1" in h.keys():
                print(f"h.keys(): {h.keys()}")
                print(f"h['1'].shape : {h['1'].shape}")
                print(f"h['1'][:2,0]  : \n{h['1'][:2,0]}")
        return h
