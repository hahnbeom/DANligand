import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import dgl.function as dglF

import torch
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
                 num_layers     = (2,4,4), 
                 l0_in_features = (65+2,28+1,32+32), #bnd,res,atm graph
                 l1_in_features = (0,0,1),  
                 num_degrees    = 2,
                 num_channels   = (32,32,32),
                 edge_features  = (2,2,2), #distance (1-hot) (&bnd, optional)
                 div            = (2,2,2),
                 n_heads        = (2,2,2),
                 pooling        = "avg",
                 chkpoint       = True,
                 modeltype      = 'comm',
                 nntypes        = ("SE3T","SE3T","SE3T"),
                 drop_out       = 0.1,
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
                                           #(2,1)]
                                           (self.num_degrees*self.num_channels[2],1)]
                )
            }
        else:
            self.fibers_atm = {
                'in': Fiber(1, self.num_channels[2]),
                'mid': Fiber(self.num_degrees, self.num_channels[2]),
                'out': Fiber(2, structure=[(self.num_degrees*self.num_channels[2],0),
                                           #(2,1)]
                                           (self.num_degrees*self.num_channels[2],1)]
                )
            }

        self.Gblock_atm = self._build_SE3gcn(self.fibers_atm, 1, 2)

        # feed-forward -- unused for simpler model that doesn't communicate R <-> A
        self.linear_res = nn.Linear(num_channels[1]+num_channels[2], num_channels[1])
        self.linear_atm = nn.Linear(num_channels[1]+num_channels[2], num_channels[2])

        ## Finalize
        # 1. category
        Cblock = []
        Wblock = []
        Tblock = []
        Qblock = []
        for key in self.outtype:
            Cblock.append(nn.Linear(self.num_degrees*self.num_channels[2],
                                    self.num_degrees*self.num_channels[2]))
            Cblock.append(nn.ReLU(inplace=True))

            # convert nfeatures -> category
            Cblock.append(nn.Linear(self.num_degrees*self.num_channels[2],
                                    self.noutbin)) #nmetal (w/ none)

            Wblock.append(nn.Linear(self.num_degrees*self.num_channels[2],
                                    self.num_degrees*self.num_channels[2]))
            Wblock.append(nn.Dropout(drop_out)) 
            Wblock.append(nn.ReLU(inplace=True))
            Wblock.append(nn.Linear(self.num_degrees*self.num_channels[2], 1))
            #Wblock.append(nn.Sigmoid()) #not do this

            # orientation (quaternion) from single l1
            # how to guaruntee |q| = 1?
            if self.learn_OR:
                Tblock.append(nn.Linear(self.num_degrees*self.num_channels[2],
                                        self.num_degrees*self.num_channels[2]))
                Tblock.append(nn.Linear(self.num_degrees*self.num_channels[2], 1))
                
                Qblock.append(nn.Linear(3*self.num_degrees*self.num_channels[2],
                                        self.num_degrees*self.num_channels[2]))
                Qblock.append(nn.Linear(self.num_degrees*self.num_channels[2], 4))
            
        self.Cblock = nn.ModuleList(Cblock)
        self.Wblock = nn.ModuleList(Wblock)
        
        self.Tblock = nn.ModuleList(Tblock) #translation
        self.Qblock = nn.ModuleList(Qblock) #orientation

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

    def forward(self, G_bnd, G_atm, G_res, r2a):
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
            return torch.tensor([0.0]), None, None
        
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
            w = (1.0/(torch.sum(r2a,axis=0)+1.0)).unsqueeze(1) # 1/natm-per-res
            w = torch.transpose(w.repeat(1,r2a.shape[0]),0,1) #copy to res
            a2r = torch.transpose(r2a*w,0,1) #elem-wise multiple
            
        ### Residue layer or feature prep
        # Simple case Gres runs first
        if self.num_layers[1] > 0:
            if self.modeltype == 'simple':
                for layer in self.Gblock_res:
                    h_res = checkpoint(runlayer(layer, G_res, r_res, basis_res), *h_res)

            # bring directly from linear layer earlier if no Gres layer
            # h_resA: h_res spread to Gatm nodes
            #h_resA = h_res[0].squeeze(2)

            h_resA = h_res[0].squeeze(2)
            #print(h_resA.shape, r2a.shape)
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
        # N x 32*2 x 3
        h_atml0 = h_atm[0].requires_grad_(True).squeeze(2)
        h_atml1 = h_atm[1].requires_grad_(True)

        batch_natms = G_bnd.batch_num_nodes()

        #ligidx: B x batch_natms: [[1,0,0,0...],[0,0,0,1,...],[]]
        # cat: batch_natms x category
        
        # 1. category
        nc = int(len(self.Cblock)/len(self.outtype))
        nw = int(len(self.Wblock)/len(self.outtype))
        nq = int(len(self.Qblock)/len(self.outtype))

        # works only for single batch!
        n = h_atml0.shape[0]

        # dxyz/orientation prediction
        # straight from lig-node
        l1_lig = h_atml1[0,:,:]
        #dxyz = h_atml1[0,:,:] #64 x 1
        #rot0 = h_atml1[0,:,:] #64 x 2
        
        # walk through each func group defined
        logits_all = []
        dxyz_all = []
        orientations_all = []

        for i,key in enumerate(self.outtype):
            b,e = i*nc,(i+1)*nc
            cat = h_atml0
            #print("key,b,e",key,b,e)
            for layer in self.Cblock[b:e]: cat = layer(cat)

            b,e = i*nw,(i+1)*nw
            w = h_atml0
            for layer in self.Wblock[b:e]: w = layer(w)

            ## Category
            # average pooling
            w = torch.transpose(w,0,1)/n
            logit = torch.matmul(w,cat)[None,:,:] # 1,1,2
            logits_all.append(logit)

            ## Rotation
            # process rot0 to learn per-functype orientation (i.e. quaternion)
            if self.learn_OR:
                b,e = i*nq,(i+1)*nq

                t = torch.transpose(l1_lig,0,1)
                q = torch.flatten(l1_lig)

                for layer in self.Tblock[b:e]: t = layer(t)
                for layer in self.Qblock[b:e]: q = layer(q)

                t = torch.squeeze(t)
                q = q/torch.sqrt(torch.sum(q*q)) #normalize
                dxyz_all.append(t)
                orientations_all.append(q)

        logits_all = torch.cat(logits_all,dim=1)
        dxyz_all = torch.cat(dxyz_all,dim=0).reshape(len(dxyz_all),3)
        orientations_all = torch.cat(orientations_all,dim=0).reshape(len(dxyz_all),4)
        Pall = torch.nn.functional.softmax(logits_all,dim=2) + 1.0e-6

        #print(Pall.shape, dxyz_all.shape, orientations_all.shape)
        return Pall, dxyz_all, orientations_all
