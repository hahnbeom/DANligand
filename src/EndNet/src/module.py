import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber
from src.trigon_2 import * 

class Grid_SE3(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers_grid=2,
                 num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32,
                 l0_out_features=32,
                 l1_in_features=0, 
                 l1_out_features=0,
                 num_edge_features=32, ntypes=15,
                 dropout_rate=0.1,
                 bias=True):
        super().__init__()

        fiber_in = Fiber({0: l0_in_features}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features, 1: l1_in_features})

        self.se3 = SE3Transformer(
            num_layers   = num_layers_grid,
            num_heads    = 4,
            channels_div = 4,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: l0_out_features}), #1:l1_out_features}),
            fiber_edge=Fiber({0: num_edge_features}),
        )

        Cblock = []
        for i in range(ntypes):
            Cblock.append(nn.Linear(l0_out_features,1,bias=False)) 
            
        self.Cblock = nn.ModuleList(Cblock)
        self.dropoutlayer = nn.Dropout(dropout_rate)

    def forward(self, G, node_features, edge_features=None, drop_out=False):

        node_in, edge_in = {},{}
        for key in node_features:
            node_in[key] = node_features[key]
            if drop_out: node_in[key] = self.dropoutlayer(node_in[key])

        hs = self.se3(G, node_in, edge_features)

        hs0 = hs['0'].squeeze(2)
        if drop_out:
            hs0 = self.dropoutlayer(hs0)

        # hs0 as pre-FC embedding; N x num_channels
        cs = []
        for i,layer in enumerate(self.Cblock):
            c = layer(hs0)
            cs.append(c) # Nx2
        cs = torch.stack(cs,dim=0) #ntype x N x 2
        cs = cs.T.squeeze(0)

        return hs0, cs 

class Ligand_SE3(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers=2,
                 num_channels=32,
                 num_degrees=3,
                 n_heads_se3=4,
                 div=4,
                 l0_in_features=15,
                 l0_out_features=32,
                 l1_in_features=0,
                 l1_out_features=0, 
                 num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                 dropout_rate=0.1,
                 bias=True):
        super().__init__()

        self.l1_in_features = l1_in_features
        self.dropoutlayer = nn.Dropout(p=dropout_rate)
        
        fiber_in = Fiber({0: l0_in_features}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features, 1: l1_in_features})

        # processing ligands
        self.se3 = SE3Transformer(
            num_layers   = num_layers,
            num_heads    = n_heads_se3,
            channels_div = div,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: l0_out_features}),
            fiber_edge=Fiber({0: num_edge_features}),
        )

    def forward(self, Glig, drop_out=True):
        node_features = {'0':Glig.ndata['attr'][:,:,None].float()}
        edge_features = {'0':Glig.edata['attr'][:,:,None].float()}
        
        if drop_out:
            node_features['0'] = self.dropoutlayer( node_features['0'] )
        
        hs = self.se3(Glig, node_features, edge_features)['0'] # M x d x 1
        
        return hs.squeeze(-1)
        
class TrigonModule(nn.Module):
    def __init__(self,
                 n_trigonometry_module_stack,
                 m=16,
                 c=32,
                 dropout_rate=0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        

        self.Wrs = nn.Linear(m,c)
        self.Wls = nn.Linear(m,c)
        
        self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=c, c=c) for _ in range(n_trigonometry_module_stack)])
        self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=c, c=c) for _ in range(n_trigonometry_module_stack)])

        self.tranistion = Transition(embedding_channels=c, n=4)
        
    def forward(self, hs_rec, hs_lig, z_mask,
                D_rec, D_lig,
                use_checkpoint=True, drop_out=False):
        # hs_rec: B x Nmax x d
        # hs_lig: B x Mmax x d
        
        # process features
        # all inputs are batched
        hs_rec = self.Wrs(hs_rec)
        hs_lig = self.Wls(hs_lig)
        
        # shrink all lig atom -> key atom
        # 1nd, bnd -> bnmd? this works correctly anyways...
        # receptor dim grows to B  here...
        z = torch.einsum('bnd,bmd->bnmd', hs_rec, hs_lig )

        # trigonometry part
        for i_module in range(self.n_trigonometry_module_stack):
            if use_checkpoint:
                zadd = checkpoint.checkpoint(self.protein_to_compound_list[i_module], z, D_rec, D_lig, z_mask.unsqueeze(-1))
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                zadd = checkpoint.checkpoint(self.triangle_self_attention_list[i_module], z, z_mask)
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
            else:
                zadd = self.protein_to_compound_list[i_module](z, D_rec, D_lig, z_mask.unsqueeze(-1))
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                zadd = self.triangle_self_attention_list[i_module](z, z_mask)
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd

            # norm -> linear -> relu -> linear
            z = self.tranistion(z)
            
        return z
    
class ClassModule( nn.Module ):
    def __init__(self, m, c,
                 classification_mode='ligand',
                 n_lig_emb=4 ):
        super().__init__()
        # m: originally called "embedding_channels"
        self.classification_mode = classification_mode
        
        if self.classification_mode == 'ligand':
            self.lapool = nn.AdaptiveMaxPool2d((4,c)) #nn.Linear(l0_out_features,c)
            self.rapool = nn.AdaptiveMaxPool2d((100,c))
            self.lhpool = nn.AdaptiveMaxPool2d((20,c))
            self.linear_pre_aff = nn.Linear(c,1)
            self.linear_for_aff = nn.Linear(124,1)
            
        elif self.classification_mode in ['ligand_v2','ligand_v3','combo_v1']:
            self.wra = nn.Linear(m, m) #, bias=None)
            self.wrh = nn.Linear(m, m) #, bias=None)
            self.wla = nn.Linear(m, m)#, bias=None)
            self.wlh = nn.Linear(m, m)#, bias=None)
            self.linear_z1 = nn.Linear(m,1)
            M = {'ligand_v2':2*m,'ligand_v3':2*m+n_lig_emb,'combo_v1':2*m+n_lig_emb}[self.classification_mode]
            L = {'ligand_v2':3,'ligand_v3':5,'combo_v1':5}[self.classification_mode]

            if self.classification_mode == 'combo_v1':
                self.w_cR = nn.Linear( m, m )
                self.w_cl = nn.Linear( m, m )
                self.linear_kR = nn.Linear( m, m )
            
            self.map_to_L = nn.Linear(M,L)
            self.final_linear = nn.Linear(L, 1)

        elif self.classification_mode == 'former':
            self.linear_z1 = nn.Linear(c, 1)
            self.map_to_L = nn.Linear( 2*c+n_lig_emb , 8 )
            self.final_linear = nn.Linear( 8, 1 )
            
        elif self.classification_mode == 'tank':
            # attention matrix to affinity 
            self.linear1 = nn.Linear(m, 1)
            self.linear2 = nn.Linear(m, 1)
            self.linear1a = nn.Linear(m, m)
            self.linear2a = nn.Linear(m, m)
            self.bias = nn.Parameter(torch.ones(1))
            self.leaky = nn.LeakyReLU()
            
    def forward( self, z, hs_grid_batched, hs_key_batched,
                 lig_rep=None,
                 hs_rec_batched=None,
                 w_Rl=None, w_mask=None ):

        ## TODO
        # for classification
        if self.classification_mode == 'tank':
            #pair_energy = (self.linear1(z).sigmoid()).squeeze(-1) * z_mask #B x N x K
            
            pair_energy = ( self.linear1(F.relu(self.linear1a(z)).sigmoid()) * \
                            self.linear2(F.relu(self.linear2a(z))) ).squeeze(-1) * z_mask
            affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1,-2)))) # "NK energy sum"
            Aff = affinity_pred # 1
            
        elif self.classification_mode == 'former':
            ## simplified!
            att = self.linear_z1(z).squeeze(-1)
            #z = torch.einsum('bnh,bkh->bnk', hs_grid_batched, hs_key_batched) # b x N x k
            
            att_l = torch.nn.Softmax(dim=2)(att).sum(axis=1) # b x K
            att_r = torch.nn.Softmax(dim=1)(att).sum(axis=2) # b x N

            # "attention-weighted 1-D token"
            key_rep = torch.einsum('bk,bkl -> bl', att_l, hs_key_batched)
            grid_rep = torch.einsum('bi,bil -> bl',att_r, hs_grid_batched)

            #
            pair_rep = torch.cat([key_rep, grid_rep, lig_rep ],dim=1) # b x emb*2
            pair_rep = self.map_to_L(pair_rep) # b x L
            Aff = self.final_linear(pair_rep).squeeze(-1) # b x 1
            
        else:
            exp_z = torch.exp(z) 
            # soft alignment 
            # normalize each row of z for receptor counterpart
            zr_denom = exp_z.sum(axis=(-2)).unsqueeze(-2) # 1 x Nrec x 1 x c
            zr = torch.div(exp_z,zr_denom) # 1 x Nrec x K x c; "per-NK weight, receptor version"
            ra = zr*hs_key_batched.unsqueeze(1) # 1 x Nrec x K x c
            ra = ra.sum(axis=-2) # 1 x Nrec x c
            
            # normalize each row of z for ligand counterpart
            zl_denom = exp_z.sum(axis=(-3)).unsqueeze(-3) # 1 x Nrec x 1 x c
            zl = torch.div(exp_z,zl_denom) # 1 x Nrec x K x c; "per-NK weight, ligand version"
            zl_t = torch.transpose(zl, 1, 2) # 1 x K x Nrec x c

            la = zl_t*hs_grid_batched.unsqueeze(1) # 1 x K x Nrec x numchannel
            la = la.sum(axis=-2) # 1 x K x numchannel
            
            if self.classification_mode == 'ligand':
                # concat and then pool 
                la = self.lapool(la) # 1 x K x c
                lh = hs_key_batched # 1 x Nlig x c
                ra = self.rapool(ra) # 1 x 100 x c
                lh = self.lhpool(lh) # 1 x 20 x c

                cat = torch.cat([ra,la,lh],dim=1) # 1 x 124 x c
                
                Aff = self.linear_pre_aff(cat).squeeze(-1) # 1 x 124
                Aff = self.linear_for_aff(Aff).squeeze(-1) # b x 1

            elif self.classification_mode in ['ligand_v2', 'ligand_v3', 'combo_v1']:
                ra_rh = (self.wra(ra) + self.wrh(hs_grid_batched))
                la_lh = (self.wla(la) + self.wlh(hs_key_batched)) # b x K x emb

                att = (self.linear_z1(z)).squeeze(-1) # b x Ngrid x K
                att_l = torch.nn.Softmax(dim=2)(att).sum(axis=1) # b x K
                att_r = torch.nn.Softmax(dim=1)(att).sum(axis=2) # b x Ngrid

                key_rep = torch.einsum('bk,bkl -> bl', att_l, la_lh)
                grid_rep = torch.einsum('bk,bkl -> bl',att_r, ra_rh)

                if self.classification_mode == 'combo_v1':
                    Rh = self.w_cR( hs_rec_batched ) #1 x N x h; dimension preseved
                    lh = self.w_cl( hs_key_batched ) #b x K x h

                    z_Rl = torch.einsum('bnh,bkh->bnk', Rh, lh) # b x n x k 
                    w_mask = w_mask[:,None,:].repeat((1,z_Rl.shape[1],1))
                    z_Rl = masked_softmax(w_Rl*z_Rl, mask=w_mask, dim=1)

                    # actually bnk,1nh -> bkh: expansion to B
                    key_rep_from_R = torch.einsum( 'bnk,bnh -> bkh', z_Rl, hs_rec_batched )
                    key_rep_from_R = self.linear_kR( key_rep_from_R ) # b x k x h
                    # reuse att_l
                    key_rep_from_R = torch.einsum('bk,bkh -> bh', att_l, key_rep_from_R)
                    
                    key_rep = key_rep + key_rep_from_R
                    pair_rep = torch.cat([key_rep, lig_rep, grid_rep],dim=1) # b x emb*2 + L
                    
                elif self.classification_mode == 'ligand_v2':
                    pair_rep = torch.cat([key_rep, grid_rep],dim=1) # b x emb*2
                elif self.classification_mode == 'ligand_v3':
                    pair_rep = torch.cat([key_rep, lig_rep, grid_rep],dim=1) # b x emb*2 + L
                    
                pair_rep = self.map_to_L(pair_rep) # b x L
                Aff = self.final_linear(pair_rep).squeeze(-1) # b x 1

                
            
        return Aff
            
