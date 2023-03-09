import torch
import torch.nn as nn
import numpy as np

from torch_geometric.nn import GATConv

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
        #linear_comb.append()

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

        ps = []
        i = 0
        for n,y in zip(G_rec.batch_num_nodes(),ys):
            A = torch.einsum('id,jd->ij', x[i:i+n,:], y)
            A = torch.softmax(A,dim=1) #sum-up to 1 over dim=1 (==fragment res index)
            #print(A.shape, torch.sum(A,dim=1))
        
            # pool over receptor dim
            #"per-recres fragP" 
            p = torch.einsum('ij,jd->id', A,y)
            ps.append(p)
            i += n

        ps = torch.stack(ps,dim=0)
        ps = self.final_linear(ps) # shrink to 1-dim; replace pooling
        ps = torch.sigmoid(ps).squeeze()

        return ps

