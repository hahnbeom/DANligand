import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import dgl.function as dglF

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
                 l1_in_features = (0,0,0),  
                 num_degrees    = 2,
                 num_channels   = (32,32,32),
                 edge_features  = (2,2,2), #normalized distance (+bnd, optional)
                 div            = (2,2,2),
                 n_heads        = (2,2,2),
                 pooling        = "avg",
                 chkpoint       = True,
                 modeltype      = 'simple',
                 nntypes        = ("SE3T","SE3T","SE3T"),
                 variable_gcn   = False,
                 drop_out       = 0.1,
                 naffinitybins  = 7,
                 hfinal_from    = (0,0),
                 se3_on_energy  = 0,
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
        self.variable_gcn = variable_gcn
        self.naffinitybins = naffinitybins
        self.hfinal_from = hfinal_from
        self.se3_on_energy = se3_on_energy

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
        if self.nntypes[0] in ['SE3T','TFN']:
            self.fibers_bnd = {
                'in': Fiber(1, self.num_channels[0]),
                'mid': Fiber(self.num_degrees, self.num_channels[0]),
                'out': Fiber(1, self.num_channels[0])}
            self.Gblock_bnd = self._build_SE3gcn(self.fibers_bnd, 1, 0, self.num_layers[2], self.nntypes[0])
            
        elif self.nntypes[0] == 'GCN':
            # share the weights throughout the layers
            # only ~1k params!
            if variable_gcn:
                GCNlinear = []
                GCNnorm   = []
                for i in range(num_layers[0]):
                    GCNlinear.append(nn.Linear( num_channels[0], num_channels[0] ))
                    GCNnorm.append(nn.InstanceNorm1d( num_channels[0], eps=1e-06, affine=True ))
                self.GCNlinear = nn.ModuleList(GCNlinear)
                self.GCNnorm   = nn.ModuleList(GCNnorm)
            else:
                self.GCNlinear = nn.Linear( num_channels[0], num_channels[0] )
                self.GCNnorm   = nn.InstanceNorm1d( num_channels[0], eps=1e-06, affine=True ) #affine: learn params
                
        # Residue
        self.fibers_res = {
            'in': Fiber(1, self.num_channels[1]),
            'mid': Fiber(self.num_degrees, self.num_channels[1]),
            'out': Fiber(1, self.num_channels[1])}
        self.Gblock_res = self._build_SE3gcn(self.fibers_res, 1, 1, self.num_layers[2])

        # Atom
        if l1_in_features[2] > 0:
            inputf_atm = [(self.num_channels[2],0),(l1_in_features[2],1)]
            self.fibers_atm = {
                'in': Fiber(2, structure=inputf_atm),
                'mid': Fiber(self.num_degrees, self.num_channels[2]),
                'out': Fiber(1, self.num_degrees*self.num_channels[2])}
        else:
            self.fibers_atm = {
                'in': Fiber(1, self.num_channels[2]),
                'mid': Fiber(self.num_degrees, self.num_channels[2]),
                'out': Fiber(1, self.num_degrees*self.num_channels[2])}
        self.Gblock_atm = self._build_SE3gcn(self.fibers_atm, 1, 2, self.num_layers[2])

        # Communication
        # feed-forward -- unused for simpler model that doesn't communicate R <-> A
        self.linear_res = nn.Linear(num_channels[1]+num_channels[2], num_channels[1])
        self.linear_atm = nn.Linear(num_channels[1]+num_channels[2], num_channels[2])

        # Final process/Skip connection
        n_in = self.fibers_atm['out'].n_features
        if self.hfinal_from[0]: n_in += num_channels[2]
        if self.hfinal_from[1]: n_in += num_channels[1]

        self.tile_hatm = nn.Linear(n_in, self.fibers_atm['out'].n_features)
        
        ## Finalize
        # per-atm prediction 
        PAblock = []
        PAblock.append(nn.Linear(self.fibers_atm['out'].n_features,
                                 self.fibers_atm['out'].n_features))
        PAblock.append(nn.Dropout(drop_out))
        PAblock.append(nn.ReLU(inplace=True))
        PAblock.append(nn.Linear(self.fibers_atm['out'].n_features, 1))
        self.PAblock = nn.ModuleList(PAblock)
        
        # atm-weights layer
        Wblock = []
        Wblock.append(nn.Linear(self.fibers_atm['out'].n_features,
                                self.fibers_atm['out'].n_features))
        Wblock.append(nn.Dropout(drop_out)) 
        Wblock.append(nn.ReLU(inplace=True))
        Wblock.append(nn.Linear(self.fibers_atm['out'].n_features, 1))
        self.Wblock = nn.ModuleList(Wblock)

        # Energy layer
        if self.se3_on_energy:
            self.Gblock_enr_pre = nn.Linear(self.fibers_atm['out'].n_features,
                                            self.num_channels[2])
            self.fibers_enr = {
                'in': Fiber(1, self.num_channels[1]), #+self.num_channels[2]),
                'mid': Fiber(self.num_degrees, self.num_channels[1]), #+self.num_channels[2]),
                'out': Fiber(1, self.num_channels[1]+self.num_channels[2])}
            self.Gblock_enr = self._build_SE3gcn(self.fibers_enr, 1, 0, self.se3_on_energy) # same architecture as bnd but atm connectivity 
        
        Eblock = []
        Eblock.append(nn.Linear(self.fibers_atm['out'].n_features,
                                self.fibers_atm['out'].n_features))
        #Eblock.append(nn.InstanceNorm2d(self.fibers_atm['out'].n_features))
        Eblock.append(nn.ReLU(inplace=True))
        # add one more layer
        Eblock.append(nn.Linear(self.fibers_atm['out'].n_features,
                                self.fibers_atm['out'].n_features))

        # convert nfeatures -> bin probability
        Eblock.append(nn.Linear(self.fibers_atm['out'].n_features, self.naffinitybins))
        self.Eblock = nn.ModuleList(Eblock)
        
    def _build_SE3gcn(self, fibers, out_dim, g_index, num_layers, nntype='SE3T'):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        
        for i in range(num_layers):
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

    def forward(self, G_bnd, G_atm, G_res, idx):
        from torch.utils.checkpoint import checkpoint
        def runlayer(layer, G, r, basis):
            def custom_forward(*h):
                hd = {str(i):h_i for i,h_i in enumerate(h)}
                hd = layer( hd, G=G, r=r, basis=basis )
                h = tuple(hd[str(i)] for i in range(len(hd)))
                return (h)
            return custom_forward

        # Pass l0 features through linear layers to condense to #channels
        l0_bnd = F.elu(self.linear1_bnd(G_bnd.ndata['0'].squeeze()))
        l0_bnd = self.drop(l0_bnd)
        l0_bnd = self.linear2_bnd(l0_bnd).unsqueeze(2)
        h_bnd = [l0_bnd]

        l0_res = F.elu(self.linear1_res(G_res.ndata['0'].squeeze()))
        l0_res = self.drop(l0_res)
        l0_res = self.linear2_res(l0_res).unsqueeze(2)
        h_res = [l0_res]

        if self.nntypes[0] in ['SE3T','TFN']:
            # Compute equivariant weight basis from relative positions
            basis_bnd, r_bnd = get_basis_and_r(G_bnd, self.num_degrees-1)
            for i,layer in enumerate(self.Gblock_bnd):
                h_bnd = checkpoint(runlayer(layer, G_bnd, r_bnd, basis_bnd), *h_bnd)
            h_bnd = h_bnd[0].squeeze(2)

        elif self.nntypes[0] == 'GCN':
            gcn_msg, gcn_reduce = dglF.copy_src(src='h', out='m'), dglF.sum(msg='m', out='h')
            
            h_bnd = h_bnd[0].squeeze()
            for i in range(self.num_layers[0]):
                G_bnd.ndata['h'] = h_bnd
                G_bnd.update_all( gcn_msg, gcn_reduce )
                h_bnd = G_bnd.ndata['h']

                if self.variable_gcn:
                    h_bnd = self.GCNlinear[i](h_bnd)[:,:,None] #make 3D for instancenorm1d
                    h_bnd = F.elu( self.GCNnorm[i](h_bnd) ).squeeze()
                else:
                    h_bnd = self.GCNlinear(h_bnd)[:,:,None] 
                    h_bnd = F.elu( self.GCNnorm(h_bnd) ).squeeze()
            
        ## Intermediate: from global to pocket-atm graphs
        basis_res, r_res = get_basis_and_r(G_res, self.num_degrees-1)
        basis_atm, r_atm = get_basis_and_r(G_atm, self.num_degrees-1)
        r2a = idx['r2a'] #1hot
        # reweight by num_atoms
        w = (1.0/(torch.sum(r2a,axis=0)+1.0)).unsqueeze(1)
        w = torch.transpose(w.repeat(1,r2a.shape[0]),0,1)
        a2r = torch.transpose(r2a*w,0,1) #elem-wise multiple
        
        ### Simple case Gres runs first
        if self.modeltype == 'simple':
            if self.num_layers[1] > 0:
                for layer in self.Gblock_res:
                    h_res = checkpoint(runlayer(layer, G_res, r_res, basis_res), *h_res)

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
                    
        #h_atm = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h_atm)}
        #h_res = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h_res)}

        h_atm = h_atm[0].requires_grad_(True)
        h_lig = h_res[0][:1].requires_grad_(True)

        h_atm  = torch.transpose(h_atm,1,2)
        l0_atm = torch.transpose(l0_atm,1,2)
        h_lig  = torch.transpose(h_lig,1,2)
        
        N = h_atm.shape[0]
        h_lig  = h_lig.repeat(N,1,1) #broadcast ligand residue info

        # take skip_connect and/or ligres info
        if self.hfinal_from[0]: h_atm = torch.cat((h_atm, l0_atm), axis=2)
        if self.hfinal_from[1]: h_atm = torch.cat((h_atm,h_lig), axis=2)
        h_atm = self.tile_hatm(h_atm)

        ## Finalize -- local path: per-node prediction
        # any way to combine w/ h_atm predictions??
        idx = torch.transpose(idx['ligidx'],0,1)
        g = h_atm
        for layer in self.PAblock: g = layer(g)
        g = torch.matmul(idx,g[:,:,0])
        g = torch.transpose(g,0,1)

        ## global "weight" path
        w = h_atm
        for layer in self.Wblock: w = layer(w)
        w = torch.matmul(idx,w[:,:,0])
        w = torch.transpose(w,0,1)

        # out 1: global accuracy as weighted-per_atm_acc
        Acc = torch.mean(w*g)
        
        ## Energy path for binding affinity
        if self.se3_on_energy:
            h_atm = self.Gblock_enr_pre(h_atm)
            h_atm = [torch.transpose(h_atm,1,2)]
            basis_atm, r_atm = get_basis_and_r(G_atm, self.num_degrees-1)
            for layer in self.Gblock_enr:
                h_atm = checkpoint(runlayer(layer, G_atm, r_atm, basis_atm), *h_atm)
            h_atm = h_atm[0].squeeze(2)
                
        E = h_atm
        for layer in self.Eblock: E = layer(E)
        E = torch.matmul(idx,E.squeeze(1)) #last dim is bin dimension

        #E: natm, nchannel, g: nchannel, natm?
        # per-bin probability (pre-softmax)
        dG  = torch.sum(E,axis=0) # pool over atom-level, dimension: [1,8]
        
        return Acc, g, dG
