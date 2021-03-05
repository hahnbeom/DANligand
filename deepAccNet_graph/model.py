import dgl
from dgl.nn.pytorch import GraphConv, NNConv

import torch
from torch import nn
from torch.nn import functional as F
from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber

# # Defining a model
class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    # 28: aa-type, 1: q or SA, 65: atype, 32: atm-embedding from bnds
    def __init__(self, 
                 num_layers     = (2,4,4), 
                 l0_in_features = (65+28+2,28+1,32+28+1),
                 l1_in_features = (0,0,1),  
                 num_degrees    = 2,
                 num_channels   = (32,32,32),
                 edge_features  = (2,2,2), #dispacement & (bnd, optional)
                 div            = (2,2,2),
                 n_heads        = (2,2,2),
                 pooling        = "avg",
                 chkpoint       = True,
                 modeltype      = 'simple',
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
        self.modeltype = modeltype

        # Linear projection layers for each
        self.linear1_bnd = nn.Linear(l0_in_features[0], l0_in_features[0])
        self.linear1_res = nn.Linear(l0_in_features[1], l0_in_features[1])
        self.linear1_atm = nn.Linear(l0_in_features[2], l0_in_features[2])

        self.drop = nn.Dropout(0.1)

        self.linear2_bnd = nn.Linear(l0_in_features[0], num_channels[0])
        self.linear2_res = nn.Linear(l0_in_features[1], num_channels[1])
        self.linear2_atm = nn.Linear(l0_in_features[2], num_channels[2])

        ## Define fibers
        # Bond
        self.fibers_bnd = {
            'in': Fiber(1, self.num_channels[0]),
            'mid': Fiber(self.num_degrees, self.num_channels[0]),
            'out': Fiber(1, self.num_channels[0])}

        # Residue
        self.fibers_res = {
            'in': Fiber(1, self.num_channels[1]),
            'mid': Fiber(self.num_degrees, self.num_channels[1]),
            'out': Fiber(1, self.num_channels[1])}

        # Atom 
        inputf_atm = [(self.num_channels[2],0),(l1_in_features[2],1)]
        self.fibers_atm = {
            'in': Fiber(2, structure=inputf_atm),
            'mid': Fiber(self.num_degrees, self.num_channels[2]),
            'out': Fiber(1, self.num_degrees*self.num_channels[2])}

        # feed-forward
        self.linear_res = nn.Linear(num_channels[1]+num_channels[2], num_channels[1])
        self.linear_atm = nn.Linear(num_channels[1]+num_channels[2], num_channels[2])
        
        # Build graph convolution net
        self.Gblock_bnd = self._build_gcn(self.fibers_bnd, 1, 0)
        self.Gblock_res = self._build_gcn(self.fibers_res, 1, 1)
        self.Gblock_atm = self._build_gcn(self.fibers_atm, 1, 2)
        
        ## Finalize
        # per-atm prediction 
        PAblock = []
        PAblock.append(nn.Linear(self.fibers_atm['out'].n_features,
                                 self.fibers_atm['out'].n_features))
        PAblock.append(nn.Dropout(0.1))
        PAblock.append(nn.ReLU(inplace=True))
        PAblock.append(nn.Linear(self.fibers_atm['out'].n_features, 1))
        self.PAblock = nn.ModuleList(PAblock)
        
        # atm-weights layer
        Wblock = []
        Wblock.append(nn.Linear(self.fibers_atm['out'].n_features,
                                self.fibers_atm['out'].n_features))
        Wblock.append(nn.Dropout(0.1)) 
        Wblock.append(nn.ReLU(inplace=True))
        Wblock.append(nn.Linear(self.fibers_atm['out'].n_features, 1))
        self.Wblock = nn.ModuleList(Wblock)
        
    def _build_gcn(self, fibers, out_dim, g_index):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers[g_index]):
            Gblock.append(GSE3Res(
                fin, 
                fibers['mid'], 
                edge_dim=self.edge_features[g_index], 
                div=self.div[g_index], 
                n_heads=self.n_heads[g_index]))
            
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
            
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'],
                               self_interaction=True,
                               edge_dim=self.edge_features[g_index]))

        return nn.ModuleList(Gblock)

    def forward(self, G_bnd, G_atm, G_res, idx):
        from torch.utils.checkpoint import checkpoint
        def runlayer(layer, G, r, basis):
            def custom_forward(*h):
                hd = {str(i):h_i for i,h_i in enumerate(h)}
                hd = layer( hd, G=G, r=r, basis=basis )
                h = tuple(hd[str(i)] for i in range(len(hd)))
                return (h)
            return custom_forward

        # Compute equivariant weight basis from relative positions
        basis_bnd, r_bnd = get_basis_and_r(G_bnd, self.num_degrees-1)
        basis_res, r_res = get_basis_and_r(G_res, self.num_degrees-1)
        basis_atm, r_atm = get_basis_and_r(G_atm, self.num_degrees-1)

        # Pass l0 features through linear layers to condense to #channels
        #print("?", G_bnd.ndata['0'].squeeze().shape)
        l0_bnd = F.elu(self.linear1_bnd(G_bnd.ndata['0'].squeeze()))
        l0_bnd = self.drop(l0_bnd)
        l0_bnd = self.linear2_bnd(l0_bnd).unsqueeze(2)
        h_bnd = [l0_bnd]

        l0_res = F.elu(self.linear1_res(G_res.ndata['0'].squeeze()))
        l0_res = self.drop(l0_res)
        l0_res = self.linear2_res(l0_res).unsqueeze(2)
        h_res = [l0_res]

        # first get atm-embedding by running G_bnd
        for layer in self.Gblock_bnd:
            h_bnd = checkpoint(runlayer(layer, G_bnd, r_bnd, basis_bnd), *h_bnd)
        h_bnd = h_bnd[0].squeeze(2)

        if self.modeltype == 'simple':
            ### let' build a simpler model
            for layer in self.Gblock_res:
                h_res = checkpoint(runlayer(layer, G_res, r_res, basis_res), *h_res)

            r2a = idx['r2a']
            h_res = h_res[0].squeeze(2)
            h_res = torch.matmul(r2a,h_res)
            
            # feed in h_bnd (and h_res for simpler model) as input to G_atm
            l0_atm = torch.cat((h_bnd,h_res),axis=1)
            
            # reduce to num_channels
            l0_atm = F.elu(self.linear1_atm(l0_atm))
            l0_atm = self.drop(l0_atm)
            l0_atm = self.linear2_atm(l0_atm).unsqueeze(2)

            h_atm = [l0_atm, G_atm.ndata['x'].requires_grad_(True)]
            # then run atom layers
            for il,layer in enumerate(self.Gblock_atm):
                h_atm = checkpoint(runlayer(layer, G_atm, r_atm, basis_atm), *h_atm)

        else:
            ##TODO
            broadcast_atm2res = torch.transpose(idx['a2r'],0,1)
            broadcast_res2atm = torch.transpose(idx['r2a'],0,1)
            # after linear projection (linear1,2): (NxNchannelx1)
            count = 0 
            for layer_atm, layer_res in zip(self.Gblock_atm, self.Gblock_res):
                count += 1
                h_atm = checkpoint(runlayer(layer_atm, G_atm, r_atm, basis_atm), *h_atm)
                h_res = checkpoint(runlayer(layer_res, G_res, r_res, basis_res), *h_res)

                if count%2 == 0:
                    h_a2r = torch.matmul(broadcast_atm2res,h_atm)
                    h_r2a = torch.matmul(broadcast_res2atm,h_res)
                
                    h_res = self.linear_res(torch.cat([h_res,h_a2r]))
                    h_atm = self.linear_atm(torch.cat([h_atm,h_r2a]))
                
        h_atm = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h_atm)}
        h_res = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h_res)}

        ## local path: per-node prediction
        # any way to combine w/ h_atm predictions??
        idx = torch.transpose(idx['ligidx'],0,1)
        g = h_atm['0']
        g = torch.transpose(g,1,2)
        for layer in self.PAblock: g = layer(g)
        g = torch.matmul(idx,g[:,:,0])
        g = torch.transpose(g,0,1)

        ## new, global "weight" path
        w = h_atm['0']
        w = torch.transpose(w,1,2)
        for layer in self.Wblock:
            w = layer(w)
            
        w = torch.matmul(idx,w[:,:,0])
        w = torch.transpose(w,0,1)

        h = torch.mean(w*g)
        
        return h, g
