import sys
import torch
import torch.nn as nn

from src.module import Grid_SE3, Ligand_SE3, Ligand_GAT, TrigonModule, ClassModule
from src.other_utils import to_dense_batch, make_batch_vec

def get_pair_dis_one_hot(d, bin_size=2, bin_min=-1, bin_max=30, num_classes=32):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=num_classes)
    return pair_dis_one_hot

def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -1.0e10 #-float("inf")

    return torch.softmax(x_masked, **kwargs)

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
            
class XformModule( nn.Module ):
    def __init__(self, c, normalize=False ):
        super().__init__()
        self.linear1 = nn.Linear(c, c)
        self.layernorm = nn.LayerNorm(c)
        self.linear2 = nn.Linear(c, c)
        self.normalize = normalize

    def forward( self, V, Q, z, z_mask, dim ):
        # attention provided, not Q*K
        # z:  B x N x K x c
        # below assumes dim=2 (K-dimension)
        exp_z = torch.exp(z)
        # K-summed attention on i-th N (norm-over K)
        z_denom = exp_z.sum(axis=dim).unsqueeze(dim) # B x N x 1 x c 
        z = torch.div(exp_z,z_denom) # B x N x K x c #repeated over K

        Qa = self.linear1(Q) # B x K x c
        if dim == 1:
            Va = torch.einsum('bikc,bkc->bic', z, Qa) # B x N x K x c
        elif dim == 2:
            Va = torch.einsum('bikc,bic->bkc', z, Qa) # B x N x K x c

        #print(dim, Qa[0,:3,:10], Qa[1,:3,:10], Va[0,:3,:10], Va[1,:3,:10])
        
        #Va = self.layernorm( V + Va )
        V = V + self.linear2(Va)

        if self.normalize:
            V = nn.functional.layer_norm(V, V.shape)
        
        return V
    
class EndtoEndModel(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, args):
        super().__init__()

        self.dropout_rate = args.dropout_rate
        self.GridFeaturizer   = Grid_SE3( **args.params_grid )
        if args.ligand_model == 'se3':
            self.LigandFeaturizer = Ligand_SE3( **args.params_ligand )
        elif args.ligand_model == 'gat':
            self.LigandFeaturizer = Ligand_GAT( **args.params_ligand )
        else:
            sys.exit("unknown ligand_model: "+args.ligand_model)

        self.trigon_lig = TrigonModule( args.params_TR['n_trigon_lig_layers'],
                                        args.params_TR['m'],
                                        args.params_TR['c'],
                                        args.params_TR['dropout_rate'])

        self.n_trigon_key_layers = args.params_TR['n_trigon_key_layers']

        self.trigon_key = TrigonModule( 1,
                                        args.params_TR['m'],
                                        args.params_TR['c'],
                                        args.params_TR['dropout_rate'])

        self.class_module = ClassModule( args.params_TR['m'],
                                         args.params_TR['c'],
                                         args.classification_mode,
                                         args.n_lig_emb )
        
        self.struct_module = StructModule( args.params_TR['c'] )

        self.classification_mode = args.classification_mode
        self.d = args.params_TR['c']
        normalize = (self.classification_mode == "former_contrast2")
        
        self.XformKey = XformModule( args.params_TR['c'], normalize=normalize )
        self.XformGrid = XformModule( args.params_TR['c'], normalize=normalize )

        self.extract_ligand_embedding = LigandModule( args.dropout_rate,
                                                      n_input=args.n_lig_feat,
                                                      n_out=args.n_lig_emb )

        #only for combo
        self.sig_Rl = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, Grec, Glig, keyidx, grididx,
                gradient_checkpoint=True, drop_out=False):

        # 1) first process Grec to get h_rec -- "motif"-embedding
        node_features = {'0':Grec.ndata['attr'][:,:,None].float(), 'x': Grec.ndata['x'].float() }
        edge_features = {'0':Grec.edata['attr'][:,:,None].float()}
        h_rec, cs = self.GridFeaturizer(Grec, node_features, edge_features, drop_out)

        gridmap = torch.eye(h_rec.shape[0]).to(Grec.device)[grididx]

        h_grid = torch.matmul(gridmap, h_rec) # grid part
        
        # 1-1) trim to grid part of Grec
        Ggrid = Grec.subgraph( grididx )

        if (Ggrid.batch_num_nodes()==0).any():
            return None, None, None, None

        Ykey_s, z_norm, aff = None, None, None
        if Glig == None: # if no ligand info provided
            return Ykey_s, z_norm, cs, aff

        # 2) ligand embedding
        try:
            h_lig = self.LigandFeaturizer(Glig, drop_out=drop_out)
        except:
            return None, None, None, None

        # global embedding if needed
        h_lig_global = self.extract_ligand_embedding( Glig.gdata.to(Glig.device) ) # gdata isn't std attr

        # 3) Prep Trigon attention
        # 3-1) Grid part
        batchvec_grid = make_batch_vec(Ggrid.batch_num_nodes()).to(Ggrid.device)
        gridxyz = Ggrid.ndata['x'].squeeze().float()
        grid_x_batched, grid_mask = to_dense_batch(gridxyz, batchvec_grid)
        D_grid = get_pair_dis_one_hot(grid_x_batched, bin_size=2, bin_min=-1, num_classes=self.d).float()
        h_grid_batched, _ = to_dense_batch(h_grid, batchvec_grid)

        # 3-2) Ligand-> Key mapper (trim down ligand -> key)
        Kmax = max([idx.shape[0] for idx in keyidx])
        Nmax = max([idx.shape[1] for idx in keyidx])
        key_batched = torch.zeros((Glig.batch_num_nodes().shape[0],Kmax,Nmax)).to(Grec.device) #b x K x j
        for i,idx in enumerate(keyidx):
            key_batched[i,:idx.shape[0],:idx.shape[1]] = idx

            
        # 3-3) ligand part
        batchvec_lig = make_batch_vec(Glig.batch_num_nodes()).to(Grec.device)
        ligxyz = Glig.ndata['x'].squeeze().float()
        lig_x_batched, lig_mask = to_dense_batch(ligxyz, batchvec_lig)
        D_lig  = get_pair_dis_one_hot(lig_x_batched, bin_size=2, bin_min=-1, num_classes=self.d).float()
        h_lig_batched, _  = to_dense_batch(h_lig, batchvec_lig)
        # vars up to here

        # 3-3) trigon1 "pre-keying"
        z_mask = torch.einsum('bn,bm->bnm', grid_mask, lig_mask )

        z = self.trigon_lig( h_grid_batched, h_lig_batched, z_mask,
                             D_grid, D_lig,
                             drop_out=drop_out )
        
        # trim down to key after trigon
        # key_batched: B x K x M
        key_x_batched = torch.einsum('bik,bji->bjk', lig_x_batched, key_batched)
        h_key_batched = torch.einsum('bkj,bjd->bkd',key_batched,h_lig_batched)
        D_key  = get_pair_dis_one_hot(key_x_batched, bin_size=2, bin_min=-1, num_classes=self.d).float()
        
        #print(h_key_batched[0], h_key_batched[1])
        # vars up to here
        z = torch.einsum( 'bkj,bijd->bikd', key_batched, z)

        # 3-4) key-position-aware attn
        # shared params; sort of "recycle"
        key_mask = torch.einsum('bkj,bj->bk', key_batched, lig_mask.float()).bool()
        z_mask = torch.einsum('bn,bm->bnm', grid_mask, key_mask )

        #TODO
        h_grid_batched = h_grid_batched.repeat(h_key_batched.shape[0],1,1)
        for it in range(self.n_trigon_key_layers):
            
            # move from below to here so that h shares embedding w/ structure...
            # would it make difference?
            # update key/grid features using learned attention
            h_key_batched  = self.XformKey( h_key_batched, h_grid_batched, z, z_mask, dim=2 ) # key/query/attn
            h_grid_batched = self.XformGrid( h_grid_batched, h_key_batched, z, z_mask, dim=1 ) # key/query/attn

            z = self.trigon_key( h_grid_batched, h_key_batched, z_mask,
                                 D_grid, D_key,
                                 drop_out=drop_out )

            z_mask = torch.einsum('bn,bm->bnm', grid_mask, key_mask )
            # Ykey_s: B x K x 3; z_norm: B x N x K x d
            # z: B x N x K x d; z_norm: B x N x K (&softmaxed)
            Ykey_s, z_norm = self.struct_module( z, z_mask, Ggrid, key_mask )

        # 2-2) screening module
        aff = self.class_module( z, h_grid_batched, h_key_batched,
                                 lig_rep=h_lig_global, w_mask=key_mask )
            
        return Ykey_s, z_norm, cs, aff
