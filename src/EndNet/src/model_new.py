import torch
import torch.nn as nn

from src.src_Grid.model import SE3TransformerWrapper as Grid_SE3
from src.src_TR.model_generic import HalfModel as TR_SE3
from src.other_utils import to_dense_batch, make_batch_vec

def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -1.0e10 #-float("inf")

    return torch.softmax(x_masked, **kwargs)

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
            
class StructModule( nn.Module ):
    def __init__(self, m ):
        super().__init__()
        self.lastlinearlayer = nn.Linear(m,1)
        self.scale = 1.0 # num_head #1.0/np.sqrt(float(d))
        
    def forward( self, z, z_mask, Grec, key_mask ):
        # for structureloss
        z = self.lastlinearlayer(z).squeeze(-1) #real17 ext15
        z = masked_softmax(self.scale*z, mask=z_mask, dim = 1)
        
        # final processing
        batchvec_rec = make_batch_vec(Grec.batch_num_nodes()).to(Grec.device)
        # r_coords_batched: B x Nmax x 3
        x = Grec.ndata['x'].squeeze()
        r_coords_batched, _ = to_dense_batch(x, batchvec_rec) # time consuming!!

        Ykey_s = torch.einsum("bij,bic->bjc",z,r_coords_batched) # "Weighted sum":  i x j , i x 3 -> j x 3
        Ykey_s = [l for l in Ykey_s]
        Ykey_s = torch.stack(Ykey_s,dim=0) # 1 x K x 3

        key_mask = key_mask[:,:,None].repeat(1,1,3).float() #b x K x 3; "which Ks are used for each b"
        Ykey_s = Ykey_s * key_mask # b x K x 3 * b x 

        return Ykey_s, z #B x ? x ?

class LigandModule( nn.Module ):
    def __init__(self, dropout_rate, n_input=19, n_out=4, c=16 ):
        super().__init__()
        self.dropoutlayer = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(n_input, c)
        self.layernorm = nn.LayerNorm(c)
        self.linear2 = nn.Linear(c, n_out )

    def forward(self, ligand_features ): # 19: 10(ntors) + 3(kapp) + 1(natm) + 3(nacc/don/aro) + 3(principal)
        ligand_features = self.dropoutlayer(ligand_features)
        h_lig = self.linear1(ligand_features)
        h_lig = self.layernorm(h_lig)
        h_lig = self.linear2(h_lig)
        return h_lig
            
class EndtoEndModel(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, args):
        super().__init__()

        self.dropout_rate = args.dropout_rate
        self.se3_Grid = Grid_SE3( **args.params_grid )
        self.se3_TR = TR_SE3( **args.params_TR )

        self.class_module = ClassModule( args.params_TR['m'],
                                         args.params_TR['c'],
                                         args.classification_mode,
                                         args.n_lig_emb )
        
        self.struct_module = StructModule( args.params_TR['m'] )
        self.classification_mode = args.classification_mode
        self.extract_ligand_embedding = LigandModule( args.dropout_rate, n_out=args.n_lig_emb )

        #only for combo
        self.sig_Rl = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, Grec, Glig, keyidx, grididx,
                gradient_checkpoint=False, drop_out=False,
                trim_receptor_embedding=False ):

        # 1) first process Grec to get h_rec -- "motif"-embedding
        node_features = {'0':Grec.ndata['attr'][:,:,None].float(), 'x': Grec.ndata['x'].float() }
        edge_features = {'0':Grec.edata['attr'][:,:,None].float()}
        
        h_rec, cs = self.se3_Grid(Grec, node_features, edge_features, drop_out)

        gridmap = torch.eye(h_rec.shape[0]).to(Grec.device)[grididx]
        h_grid = torch.matmul(gridmap, h_rec) # grid part
        
        # 1-1) trim to grid part of Grec
        Ggrid = Grec.subgraph( grididx )

        if (Ggrid.batch_num_nodes()==0).any():
            return None, None, None, None

        Ykey_s, z_norm, aff = None, None, None
        if Glig == None: # if no ligand info provided
            return Ykey_s, z_norm, cs, aff
        
        # 2) TRnet part -- structure & classification (pass if no ligand provided)
        # 2-1) structure module
        z, z_mask, hs_key_batched, key_mask = self.se3_TR( Ggrid, Glig, h_grid, keyidx,
                                                           gradient_checkpoint, drop_out )

        # Ykey: predicted position of keys
        # (0,0,0) at masked positions
        Ykey_s, z_norm = self.struct_module( z, z_mask, Ggrid, key_mask )

        # 2-2) screening module
        # not validated for minibatch >1
        batchvec_rec = make_batch_vec(Ggrid.batch_num_nodes()).to(Grec.device)
        hs_grid_batched, _ = to_dense_batch(h_grid, batchvec_rec)

        hs_rec_batched = None
        hs_lig_batched = None
        w_Rl = None
        if self.classification_mode in ['ligand_v3','combo_v1']:
            hs_lig_batched = self.extract_ligand_embedding( Glig.gdata.to(Glig.device) ) # gdata isn't std attr
            
        if self.classification_mode == 'combo_v1':
            batchvec_rec = make_batch_vec(Grec.batch_num_nodes() - Ggrid.batch_num_nodes()).to(Grec.device)
            recidx = torch.where(torch.sum(gridmap,dim=0)<0.5)[0] #zeros
            recmap = torch.eye(h_rec.shape[0]).to(Grec.device)[recidx]
            
            Grec = Grec.subgraph(recidx)
            h_rec  = torch.matmul(recmap, h_rec) # receptor atom part
            
            hs_rec_batched, _ = to_dense_batch(h_rec, batchvec_rec)
            x_rec = Grec.ndata['x'].unsqueeze(0) # 1 x N x 1 x 3
            x_key = Ykey_s.unsqueeze(1) # b x 1 x k x 3

            D_Rl = x_rec - x_key
            w_Rl = torch.exp(-torch.sum(D_Rl*D_Rl,dim=-1)/self.sig_Rl) # b x N x k
            

        aff = self.class_module( z, hs_grid_batched, hs_key_batched,
                                 hs_lig_batched,
                                 hs_rec_batched,
                                 w_Rl, key_mask )
            
        return Ykey_s, z_norm, cs, aff
