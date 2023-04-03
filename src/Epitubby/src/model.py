import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self,
                 num_node_feats,
                 num_edge_feats,
                 num_channels=32,
                 n_layers_rec  = 3,
                 n_layers_frag = 3,
                 head = 8, # heads in GAT
                 hid = 4, # hidden dim in GAT
                 dropout=0.2
    ):
        from torch_geometric.nn import GATConv

        super().__init__()
        self.num_channels = num_channels
        self.n_layers_rec = n_layers_rec

        collapse_esm = []
        linear_comb = []

        mid1 = int(num_node_feats/2)
        mid2 = int(num_node_feats/4)
        mid3 = 24 + mid2

        self.dropoutlayer = nn.Dropout(dropout)
        
        collapse_esm.append(nn.Linear( num_node_feats-24, mid1 ))
        collapse_esm.append(nn.Linear( mid1, mid2 ))
        
        linear_comb.append(nn.Linear( mid3, num_channels ))
        for _ in range(n_layers_frag):
            linear_comb.append( nn.Linear( num_channels, num_channels ) )
        
        self.collapse_esm = nn.ModuleList( collapse_esm )
        self.linear_comb = nn.ModuleList( linear_comb )

        # GAT at receptor level
        GAT = []
        for _ in range(n_layers_rec): # use unshared weights
            GATconv1 = GATConv(self.num_channels, hid, head, dropout=dropout)
            GATconv2 = GATConv(head * hid, self.num_channels,
                               concat=False, heads=1, dropout=dropout)
            GAT.append(GATconv1)
            GAT.append(GATconv2)

        self.GAT = nn.ModuleList( GAT )
        self.final_linear = nn.Linear(num_channels,1,bias=True)

    def forward(self, frag_emb, G_rec ):
        node_features = G_rec.ndata["attr"].float()
        edge_features = G_rec.edata["attr"].float()
        frag_emb = frag_emb.float()

        u,v = G_rec.edges()
        N = node_features.shape[0]
        edge_index = torch.zeros((2,len(u)),dtype=int).to(G_rec.device)
        edge_index[0,:] = u
        edge_index[1,:] = v

        #print(edge_index.shape, len(u), edge_features.shape, N)

        x = self.dropoutlayer( node_features )
        x_esm = x[:,24:] #"batched_graph_nodes"
        for layer in self.collapse_esm:
            x_esm = layer(x_esm)

        x = torch.cat([x[:,:24],x_esm],dim=1)
        for layer in self.linear_comb:
            x = layer(x)

        for i,layer in enumerate(self.GAT):
            x = layer( x, edge_index=edge_index, edge_attr=edge_features )
            if i%2 == 0: x = torch.nn.functional.elu(x)

        #TODO: split into batch

        # frag_emb: B x 9 x c
        ys = self.dropoutlayer( frag_emb )
        y_esm = ys[:,:,24:]
        for layer in self.collapse_esm:
            y_esm = layer(y_esm)

        ys = torch.cat([ys[:,:,:24],y_esm],dim=2)
        for layer in self.linear_comb:
            ys = layer(ys)

        x = torch.nn.functional.layer_norm(x,x.shape)
        ys = torch.nn.functional.layer_norm(ys,ys.shape)
            
        ps = []
        i = 0
        for n,y in zip(G_rec.batch_num_nodes(),ys):
            A = torch.einsum('id,jd->ij', x[i:i+n,:], y)

            '''
            A = torch.softmax(A,dim=0) #sum-up to 1 over dim=0 (==fragment res index)

            # pool over receptor dim
            #"per-recres fragP" 
            p = torch.einsum('ij,jd->id', A,y)
            p = self.final_linear(p) # shrink to 1-dim; replace pooling
            '''

            ## m4
            '''
            A = torch.nn.functional.layer_norm(A,A.shape) #values normalized around 0
            p = torch.sum(A,dim=1) #better maxpool?
            '''

            ## m5
            p = torch.nn.functional.max_pool1d(A,A.shape[-1]).squeeze()
            p = 1.0/(1.0+torch.exp(-(p-3.0))) 
            #p = torch.sigmoid(p).squeeze()
            ps.append(p)
            i += n

        return ps

# No graph, no intermediate layers -- just by attention
class AttnModel(nn.Module):
    def __init__(self,
                 num_node_feats,
                 num_edge_feats, #unused
                 num_channels_collapse=128,
                 num_channels_attn=32, #"K"
                 n_attn_layers=3,
                 dropout=0.2
    ):
        super().__init__()
        #self.num_channels = num_channels
        self.dropoutlayer = nn.Dropout(dropout)
        self.n_attn_layers = n_attn_layers

        self.input_linear = nn.Linear(num_node_feats,num_channels_collapse)

        #self.U = nn.Linear(num_channels_collapse,num_channels_attn) # frag "W"
        #self.V = nn.Linear(num_channels_collapse,num_channels_attn) # receptor "W"
        #self.q = nn.Linear(num_channels_attn,1) # d->k reduction

        self.U = nn.Parameter(torch.rand(num_channels_collapse,num_channels_attn)) # frag "W"
        self.V = nn.Parameter(torch.rand(num_channels_collapse,num_channels_attn)) # receptor "W"
        self.q = nn.Parameter(torch.rand(num_channels_attn)) # d->k reduction
        self.gate_y = nn.Linear(num_channels_collapse*2, 1) # y+y_p -> y gating
        
        self.final_linear = nn.Linear(num_channels_attn,1,bias=True)

    def forward(self, frag_emb, G_rec ):
        node_features = G_rec.ndata["attr"].float()
        edge_features = G_rec.edata["attr"].float()
        frag_emb = frag_emb.float()
        N = node_features.shape[0]

        #print(edge_index.shape, len(u), edge_features.shape, N)

        xs = self.dropoutlayer( node_features )
        xs = self.input_linear(xs)
        ys = self.dropoutlayer( frag_emb )
        ys = self.input_linear(ys)

        xs = torch.nn.functional.layer_norm(xs,xs.shape)
        ys = torch.nn.functional.layer_norm(ys,ys.shape)
            
        ps = []
        i = 0
        # d: num_channel_collapse
        # k: num_channel_attn
        # x: rec y: frag
        for (n,y) in zip(G_rec.batch_num_nodes(),ys):
            y_p = F.relu( torch.einsum('dk,jd->jk', self.V, y ) )
            y_p = torch.einsum( 'k,jk->jk', self.q, y_p ) # MxK; normalized 0~5
            y_p = F.relu( y_p )
            
            x = xs[i:i+n,:]
            for ilayer in range(self.n_attn_layers):
                x_p = F.relu( torch.einsum( 'dk,id->ik', self.U, x ) )

                I = torch.einsum('ik,jk->ij', x_p, y_p ) # unnormalized

                if ilayer < self.n_attn_layers-1:
                    A = torch.softmax(I,dim=0)
                    x_p = F.relu(torch.einsum('ij,jk->ik',A,y_p))
                    x_p = torch.einsum( 'ik,dk->id', x_p, self.V ) # revert back the channel size

                    # update y
                    z = torch.sigmoid( self.gate_y(torch.cat([x,x_p],-1)).squeeze() ) # N x 2D -> N x 1

                    # per-channel weighted sum
                    x = torch.einsum('j,jd->jd',1-z,x) + torch.einsum('j,jd->jd',z,x_p) # N x D

            # bilinear pooling
            p = torch.einsum('ij,jk->ik', I, y_p) # un-normalized I not A; 
            p = torch.einsum('ik,ik->i', x_p, p)
            p = torch.sigmoid(p)
            ps.append(p)

            i += n

        return ps #list of tensor

class InteractionPooling(nn.Module):
    def __init__(self, pool_size = 5):

        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(1))
        self.maxpool = nn.MaxPool1d( pool_size, padding=pool_size // 2)

    def forward(self, C):
        #P = self.maxpool(C)
        P = C 
        
        std = torch.std(P)
        Q = P - torch.mean(P) - self.gamma*std*std
        Q = nn.functional.relu(Q)

        # originally sum(Q)/sum(sign(Q)+1)
        rho = torch.sum(Q,axis=1) # B x N
        rho = rho / (torch.mean(rho,axis=1)+1.0)[:,None] # significance at i-th -- make sense?
        
        rho = 1.0/(1.0 + torch.exp(rho-0.5))
        return rho # should be B or B x 1 
        
class DSCRIPTModel(nn.Module):
    def __init__(self,
                 num_node_feats, #==d0
                 d=128,
                 h=32,
                 dropout=0.2
    ):
        super().__init__()
        d0 = num_node_feats
        dropoutlayer = nn.Dropout(dropout)
        
        self.projection_module = nn.ModuleList( [ nn.Linear(d0,d), nn.ReLU(), dropoutlayer ])
        self.contact_module1 = nn.ModuleList( [ nn.Linear(2*d, h), nn.BatchNorm2d(h), nn.ReLU() ] )
        self.contact_module2 = nn.ModuleList( [ nn.Conv2d(h, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid() ] )

        self.pool_module = InteractionPooling()

    def forward(self, frag_emb, G_rec ):
        E1 = G_rec.ndata["attr"].float() # BN x d0
        E2 = frag_emb.float() # B x M x d0
        
        # make as B x N x d0
        nnodes = G_rec.batch_num_nodes()
        N = int(torch.max(nnodes))
        M = E2.shape[1]
        d0 = E2.shape[2]

        E1 = torch.zeros((nnodes.shape[0], N, d0)).to(G_rec.device)
        mask = torch.zeros((nnodes.shape[0], M, N)).to(G_rec.device)
        b = 0
        for i,n in enumerate(nnodes):
            E1[i,:n,:] = G_rec.ndata["attr"][b:b+n].float()
            b += n
            mask[i,:,:n] = 1.0

        E1 = E1.requires_grad_()

        for layer in self.projection_module:
            E1 = layer(E1)
            E2 = layer(E2)

        Z1 = E1[:,:,None,:].repeat(1,1,M,1) 
        Z2 = E2[:,None,:,:].repeat(1,N,1,1)
        '''
        # old - simple concatenation
        A = torch.cat([Z1,Z2],dim=-1) # B x N x M x 2d
        '''
        h = Z1.shape[-1]
        A = torch.zeros((nnodes.shape[0], N, M, 2*h)).to(G_rec.device)
        A[...,:h] = Z1*Z2
        A[...,h:] = torch.abs(Z1-Z2)

        for i,layer in enumerate(self.contact_module1):
            A = layer(A) #shuffle to support batchnorm & conv1d
            if i == 0:
                A = A.transpose(3,1)

        B = A # B x h x M x N
        for layer in self.contact_module2:
            B = layer(B)

        C = B
        C = mask*B.squeeze() # B x M x N; elementwise multi
        p = self.pool_module( C )

        return p

