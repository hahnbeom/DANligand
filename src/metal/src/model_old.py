import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import dgl.function as dglF

import torch
from torch import nn
from torch.nn import functional as F
from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber
from . import myutils

# # Defining a model
class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    # 28: aa-type, 1: q or SA, 65: atype, 32: atm-embedding from bnds
    def __init__(self, 
                 num_layers     = (2,4,4), 
                 l0_in_features = (65+28+2,28+1,32+32),
                 l1_in_features = (0,0,1),  
                 num_degrees    = 2,
                 num_channels   = (32,32,32),
                 edge_features  = (2,2,2), #distance (1-hot) (&bnd, optional)
                 div            = (2,2,2),
                 n_heads        = (2,2,2),
                 pooling        = "avg",
                 chkpoint       = True,
                 modeltype      = 'simple',
                 nntypes        = ("SE3T","SE3T","SE3T"),
                 drop_out       = 0.1,
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

        # shared
        self.drop = nn.Dropout(drop_out)

        # Linear projection layers for each
        self.linear1_bnd = nn.Linear(l0_in_features[0], l0_in_features[0])
        self.linear1_res = nn.Linear(l0_in_features[1], l0_in_features[1])
        self.linear1_atm = nn.Linear(l0_in_features[2], l0_in_features[2])

        self.linear2_bnd = nn.Linear(l0_in_features[0], num_channels[0])
        self.linear2_res = nn.Linear(l0_in_features[1], num_channels[1])
        self.linear2_atm = nn.Linear(l0_in_features[2], num_channels[2])

        ## Build graph convolution net -- Define fibers, etc.
        # Bond
        self.fibers_bnd = {
            'in': Fiber(1, self.num_channels[0]),
            'mid': Fiber(self.num_degrees, self.num_channels[0]),
            'out': Fiber(1, self.num_channels[0])}
        self.Gblock_bnd = self._build_SE3gcn(self.fibers_bnd, 1, 0, self.nntypes[0])
            
        # Residue
        self.fibers_res = {
            'in': Fiber(1, self.num_channels[1]),
            'mid': Fiber(self.num_degrees, self.num_channels[1]),
            'out': Fiber(1, self.num_channels[1])}
        self.Gblock_res = self._build_SE3gcn(self.fibers_res, 1, 1)

        # Atom
        if l1_in_features[2] > 0:
            inputf_atm = [(self.num_channels[2],0),(l1_in_features[2],1)]
            self.fibers_atm = {
                'in': Fiber(2, structure=inputf_atm),
                'mid': Fiber(self.num_degrees, self.num_channels[2]),
                'out': Fiber(2, structure=[(self.num_degrees*self.num_channels[2],0),
                                           (1,1)])
            }
            #'out': Fiber(1, self.num_degrees*self.num_channels[2])}
        else:
            self.fibers_atm = {
                'in': Fiber(1, self.num_channels[2]),
                'mid': Fiber(self.num_degrees, self.num_channels[2]),
                'out': Fiber(2, structure=[(self.num_degrees*self.num_channels[2],0),
                                           (1,1)])
            }

        self.Gblock_atm = self._build_SE3gcn(self.fibers_atm, 1, 2)

        ## Finalize
        # 1. category 
        Cblock = []
        Cblock.append(nn.Linear(self.num_degrees*self.num_channels[2],
                                self.num_degrees*self.num_channels[2]))
        Cblock.append(nn.ReLU(inplace=True))
        # convert nfeatures -> category
        Cblock.append(nn.Linear(self.num_degrees*self.num_channels[2],
                                myutils.NMETALS+1)) #nmetal + none
                                
        self.Cblock = nn.ModuleList(Cblock)

        # 2. dxyz
        '''
        Xblock = []
        Xblock.append(nn.Linear(self.fibers_atm['out'].n_features,
                                self.fibers_atm['out'].n_features))
        Xblock.append(nn.ReLU(inplace=True))
        # convert nfeatures -> xyz
        Xblock.append(nn.Linear(self.fibers_atm['out'].n_features,3))
        self.Xblock = nn.ModuleList(Xblock)
        '''

    # out_dim unused
    def _build_SE3gcn(self, fibers, out_dim, g_index, nntype='SE3T'):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        
        for i in range(self.num_layers[g_index]):
            if nntype == 'SE3T':
                Gblock.append( GSE3Res(fin, fibers['mid'], edge_dim=self.edge_features[g_index],
                                       div=self.div[g_index], n_heads=self.n_heads[g_index]) )
            elif nntype == 'TFN':
                Gblock.append( GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_features[g_index]) )
                
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
            
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'],self_interaction=True,
                               edge_dim=self.edge_features[g_index]))

        return nn.ModuleList(Gblock)

    def forward(self, G_bnd, G_atm, G_res, r2a, ligidx):
        from torch.utils.checkpoint import checkpoint
        def runlayer(layer, G, r, basis):
            def custom_forward(*h):
                hd = {str(i):h_i for i,h_i in enumerate(h)}
                hd = layer( hd, G=G, r=r, basis=basis )
                h = tuple(hd[str(i)] for i in range(len(hd)))
                return (h)
            return custom_forward

        #G_bnd.batch_num_edges()
        
        # Pass l0 features through linear layers to condense to #channels
        if len(G_bnd.ndata['0'].squeeze().shape) != 2:
            return torch.tensor([0.0]), None
        
        l0_bnd = F.elu(self.linear1_bnd(G_bnd.ndata['0'].squeeze()))
        l0_bnd = self.drop(l0_bnd)
        l0_bnd = self.linear2_bnd(l0_bnd).unsqueeze(2)
        h_bnd = [l0_bnd]

        # Compute equivariant weight basis from relative positions
        basis_bnd, r_bnd = get_basis_and_r(G_bnd, self.num_degrees-1)
        for layer in self.Gblock_bnd:
            h_bnd = checkpoint(runlayer(layer, G_bnd, r_bnd, basis_bnd), *h_bnd)
        h_bnd = h_bnd[0].squeeze(2)

        ## Intermediate: from global to pocket-atm graphs
        basis_atm, r_atm = get_basis_and_r(G_atm, self.num_degrees-1)
            
        if r2a != None:
            l0_res = F.elu(self.linear1_res(G_res.ndata['0'].squeeze()))
            l0_res = self.drop(l0_res)
            l0_res = self.linear2_res(l0_res).unsqueeze(2)
            h_res = [l0_res]
        
            basis_res, r_res = get_basis_and_r(G_res, self.num_degrees-1)
            # reweight by num_atoms
            w = (1.0/(torch.sum(r2a,axis=0)+1.0)).unsqueeze(1)
            w = torch.transpose(w.repeat(1,r2a.shape[0]),0,1)
            a2r = torch.transpose(r2a*w,0,1) #elem-wise multiple
            
        ### Residue layer or feature prep
        # Simple case Gres runs first
        if self.num_layers[1] > 0:
            if self.modeltype == 'simple':
                for layer in self.Gblock_res:
                    h_res = checkpoint(runlayer(layer, G_res, r_res, basis_res), *h_res)
                h_res = h_res[0].squeeze(2)

            # bring directly from linear layer earlier if no Gres layer
            # h_resA: h_res spread to Gatm nodes
            h_resA = h_res[0].squeeze(2)
            h_resA = torch.matmul(r2a,h_resA)
            # feed in h_bnd (and h_res for simpler model) as input to G_atm
            l0_atm = torch.cat((h_bnd,h_resA),axis=1)

            # reduce to num_channels
            l0_atm = F.elu(self.linear1_atm(l0_atm))
            l0_atm = self.drop(l0_atm)
            l0_atm = self.linear2_atm(l0_atm).unsqueeze(2)
            
        else:
            l0_atm = h_bnd

        if self.l1_in_features[2] > 0:
            h_atm = [l0_atm, G_atm.ndata['x'].requires_grad_(True)]
        else:
            h_atm = [l0_atm]

        # then run atom layers
        if self.modeltype == "simple":
            for il,layer in enumerate(self.Gblock_atm):
                h_atm = checkpoint(runlayer(layer, G_atm, r_atm, basis_atm), *h_atm)

        elif self.modeltype == 'comm':
            # no consideration on l1 features below...
            count = 0
            for layer_atm, layer_res in zip(self.Gblock_atm, self.Gblock_res):
                count += 1
                h_atm = checkpoint(runlayer(layer_atm, G_atm, r_atm, basis_atm), *h_atm)
                h_res = checkpoint(runlayer(layer_res, G_res, r_res, basis_res), *h_res)

                if count%2 == 0 and count > 0: # after
                    h_atmd = h_atm[0].squeeze()
                    h_resd = h_res[0].squeeze()
                    
                    h_a2r = torch.matmul(a2r,h_atmd)
                    h_r2a = torch.matmul(r2a,h_resd)

                    h_res = (self.linear_res(torch.cat([h_resd,h_a2r],axis=1))[...,None],h_res[1])
                    h_atm = (self.linear_atm(torch.cat([h_atmd,h_r2a],axis=1))[...,None],h_atm[1])

        #print(len(h_atm))
        h_atml0 = h_atm[0].requires_grad_(True).squeeze(2)
        h_atml1 = h_atm[1].requires_grad_(True).squeeze(1)
        #h_atml1 = h_atm[1].requires_grad_(True)
        #h_res = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h_res)}

        batch_natms = G_bnd.batch_num_nodes()

        #ligidx: B x batch_natms: [[1,0,0,0...],[0,0,0,1,...],[]]
        # cat: batch_natms x category
        
        # 1. category
        cat = h_atml0
        for layer in self.Cblock:
            cat = layer(cat)
        cat = torch.matmul(ligidx,cat.squeeze(1)).squeeze(0) #last dim is bin dimension

        # 2. dxyz prediction
        # option1: straight from lig-node
        #dxyz = h_atml1[None,0] #torch.matmul(ligidx,h_atml1)
        dxyz = torch.matmul(ligidx,h_atml1)
        
        #for layer in self.Xblock: X = layer(X)
        #dxyz = torch.matmul(ligidx,X.squeeze(1)).squeeze(0)
        
        return cat, dxyz
