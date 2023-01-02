import sys,os
import numpy as np
import torch
import dgl
from torch.utils import data
from os import listdir
from os.path import join, isdir, isfile
from scipy.spatial import distance, distance_matrix
import scipy
import time

#sys.path.insert(0,'./')
from . import myutils
from . import motif

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,
                 targets,
                 root_dir        = "",
                 verbose         = False,
                 ball_radius     = 12.0,
                 displacement    = "",
                 randomize_lig   = 0.5,
                 randomize       = 0.0,
                 tag_substr      = [''],
                 upsample        = None,
                 num_channels    = 32,
                 sasa_method     = "sasa",
                 edgemode        = 'distT',
                 edgek           = (0,0),
                 edgedist        = (2.5,4.5),
                 ballmode        = 'all',
                 distance_feat   = 'std',
                 debug           = False,
                 nsamples_per_p  = 1,
                 CBonly          = False,
                 xyz_as_bb       = False, 
                 sample_mode     = 'random',
                 use_l1          = False,
                 origin_as_node  = False
    ):
        self.proteins = targets
        
        self.datadir = root_dir
        self.verbose = verbose
        self.ball_radius = ball_radius
        self.randomize_lig = randomize_lig
        self.randomize = randomize
        self.tag_substr = tag_substr
        self.sasa_method = sasa_method
        self.num_channels = num_channels
        self.ballmode = ballmode #["com","all"]
        self.edgemode = edgemode
        self.edgedist = edgedist
        self.edgek = edgek
        #self.dist_fn_atm = lambda x:get_dist_neighbors(x, mode=edgemode, top_k=edgek[1], dcut=edgedist[1])
        self.dist_fn_atm = get_dist_neighbors
        self.debug = debug
        self.distance_feat = distance_feat
        self.nsamples_per_p = nsamples_per_p
        self.xyz_as_bb = xyz_as_bb
        self.sample_mode = sample_mode
        self.CBonly = CBonly
        self.use_l1 = use_l1
        self.origin_as_node = origin_as_node

        if upsample == None:
            self.upsample = sample_uniform
        else:
            self.upsample = upsample

        self.nsamples = max(1,len(self.proteins)*nsamples_per_p)
        
    def __len__(self):
        'Denotes the total number of samples'
        return int(self.nsamples)
    
    def __getitem__(self, index):
        # Select a sample decoy
        t0 = time.time()
        info = {}
        info['sname'] = 'none'
        
        skip_this = False
        
        # "proteins" in a format of "protein.idx"
        ip = int(index/self.nsamples_per_p) #==index

        if self.labeled:
            pname = self.proteins[ip].split('.')[0]
            gridchain = self.proteins[ip].split('.')[1]
        else:
            pname = '.'.join(self.proteins[ip].split('.')[:-1])
            gridchain = None

        fname = self.datadir+'%s.grid.npz'%self.proteins[ip]
        if not os.path.exists(fname): 
            fname = self.datadir+'%s.lig.npz'%self.proteins[ip])
            
        if not os.path.exists(fname):
            print("file not exists:",fname)
            skip_this = True
            cats = []
        else:
            sample = np.load(fname,allow_pickle=True) #motif type & crd only
            labels = sample['labels'] # ngrid x nmotif
            if len(labels) == 0:
                skip_this = True
            else:
                mask  = np.sum(labels>0,axis=1) #0 or 1
                
        if skip_this:
            info['pname'] = 'none'
            print("failed to read input npz")
            return False, info

        info['pname'] = self.proteins[ip]
        info['sname'] = self.proteins[ip]

        grids  = sample['xyz'] #vector; set all possible motif positions for prediction

        # receptor features that go into se3        
        prop = np.load(self.datadir+pname+".prop.npz")
        xyz         = prop['xyz_rec']
        charges_rec = prop['charge_rec'] 
        atypes_rec  = prop['atypes_rec'] #1-hot
        anames      = prop['atmnames']
        aas_rec     = prop['aas_rec'] #rec only

        #mask self
        if gridchain != None:
            iexcl   = [i for i,a in enumerate(prop['atmres_rec']) if a[0].split('.') == gridchain]
        #iexcl = np.array(iexcl)
        xyz[iexcl] += 1000.0

        sasa_rec = prop['sasa_rec']
       
        # bond properties
        bnds    = prop['bnds_rec']
        atypes  = np.array([myutils.find_gentype2num(at) for at in atypes_rec]) # string to integers
            
        # randomize motif coordinate
        '''
        dxyz = np.zeros(3)
        if self.randomize_lig > 1e-3:
            dxyz = 2.0*self.randomize_lig*(0.5 - np.random.rand(3))
        '''

        ### grids with labels only
        #ilabeled = np.where(np.sum(labels,axis=0)>0)
        #grids_l = grids[ilabeled]
        #labels = 
        
        grid_com = np.mean(grids,axis=0)
        
        # orient around grid center
        xyz = xyz - grid_com
        grids = grids - grid_com

        # randomize the rest coordinate
        if self.randomize > 1e-3:
            randxyz = 2.0*self.randomize*(0.5 - np.random.rand(len(xyz),3))
            xyz = xyz + randxyz

        d2o = np.sqrt(np.sum(xyz*xyz,axis=1))

        ## append "virtual" nodes at the grid points
        for i,grid in enumerate(grids):
            anames = np.concatenate([['grid%04d'%i],anames])
            aas_rec = np.concatenate([[0], aas_rec]) # unk
            atypes = np.concatenate([[0],atypes]) #null
            sasa_rec = np.concatenate([[0.0],sasa_rec]) # 
            charges_rec = np.concatenate([[0.0],charges_rec])
            d2o = np.concatenate([[0.0], d2o])

        xyz = np.concatenate([grids,xyz])
        aas1hot = np.eye(myutils.N_AATYPE)[aas_rec]
        atypes  = np.eye(max(myutils.gentype2num.values())+1)[atypes] # convert integer to 1-hot
        sasa    = np.expand_dims(sasa_rec,axis=1)
        charges = np.expand_dims(charges_rec,axis=1)
        d2o = d2o[:,None]

        #natm = xyz.shape[0]-grids.shape[0] # real atoms
        
        #try:
        G_atm = self.make_atm_graphs(xyz, grids,
                                     [aas1hot,atypes,sasa,charges,d2o],
                                     bnds, anames, self.CBonly, self.use_l1)

        #except:
        #    G_atm  = self.make_atm_graphs(xyz, grids,
        #                                  [aas1hot,atypes,sasa,charges],
        #                                  bnds, anames, self.CBonly, self.use_l1)
        #    print("graphgen fail")
        #    return False, info
       
        # works up to 5layers 
        if isinstance(G_atm,int) or G_atm.number_of_nodes() > 5000 or G_atm.number_of_edges() > 50000:
            t1 = time.time()
            if isinstance(G_atm,int):
                print(" - Skip this (name/grid/node): ", info["pname"], G_atm, "%.1f sec"%(t1-t0))
            else:
                print(" - Skip this (name/grid/node/edge): ", info["pname"], grids.shape[0],
                      G_atm.number_of_nodes(), G_atm.number_of_edges(), "%.1f sec"%(t1-t0))
            return False, info

        info['grids'] = grids+grid_com # recover to untranslated 
        info['com']   = grid_com # where the origin is set
        info['numnode'] = G_atm.number_of_nodes()
        info['labels']  = labels
        info['mask']    = mask

        # Remove self edges
        # TODO: check
        G_atm = myutils.remove_self_edges(G_atm)

        t1 = time.time()
        if self.debug:
            print("ngrid/node/edge:", info['pname'], grids.shape[0], G_atm.number_of_nodes(), G_atm.number_of_edges(), t1-t0)
            
        return G_atm, info

    def report_xyz(self,outname, atypes, xyz, origin=[0,0,0]):
        out = open(outname,'w')
        form = '%1s %8.3f %8.3f %8.3f\n'
        out.write(form%("F",origin[0],origin[1],origin[2]))
        for i,a in enumerate(atypes):
            out.write(form%(a[0],xyz[i,0],xyz[i,1],xyz[i,2]))
        out.close()
            
    def get_a_sample(self,pname,index):
        pindices = []

        samples = np.load(self.datadir+pname+".lig.npz",allow_pickle=True)
        for substr in self.tag_substr:
            pindices += [i for i,n in enumerate(samples['name']) if substr in n]
        fnats = np.array([samples['fnat'][i] for i in pindices])

        if self.sample_mode == 'random':
            pindex  = np.random.choice(pindices,p=self.upsample(fnats))
        elif self.sample_mode == 'serial':
            pindex = index%len(pindices)
        
        return samples, pindex

    def make_res_graph(self, xyz, center_xyz, obt_fs, repsatm_idx, rsds_in_Gatm):
        # repsatm_idx: 166 vs reps_idx: 100

        # get xyz that goes into Gres
        xyz_reps = xyz[repsatm_idx]
        xyz_reps = torch.tensor(xyz_reps-center_xyz).float()

        # Grabbing a < dist neighbor
        kd      = scipy.spatial.cKDTree(xyz_reps)
        kd_ca   = scipy.spatial.cKDTree(center_xyz)

        indices = kd_ca.query_ball_tree(kd, 30.0)[0] # residue idx
        reps_idx = [repsatm_idx[i] for i in indices] # atm idx

        # which idx in reps_idx map to idx_in_Gatm; i.e. what becomes residue-embedding in G_atm
        # rsds_in_Gatm: resno following idx_ord
        r2amap = np.zeros(len(rsds_in_Gatm), dtype=int)
        for i,rsd in enumerate(rsds_in_Gatm):
            if rsd in indices: r2amap[i] = indices.index(rsd)
            else: sys.exit("unexpected resno %d"%(rsd)) #should not happen
        # many entries unused in Gatm may left as 0
        #r2amap = np.eye(max(indices)+1)[r2amap]
        r2amap = np.eye(len(indices))[r2amap]

        # renew xyz on picked indices
        xyz_reps = xyz[indices]
        xyz_reps = torch.tensor(xyz_reps-center_xyz).float()
        
        obt  = []
        for f in obt_fs:
            if len(f) > 0: obt.append(f[reps_idx])
        obt = np.concatenate(obt,axis=-1)

        u,v = self.dist_fn_res(xyz_reps[None,])
        D = torch.sqrt(torch.sum((xyz_reps[v] - xyz_reps[u])**2, axis=-1)+1e-6)[...,None]
        D1hot = distance_feature(self.distance_feat,D,1.0,10.0)

        G_res = dgl.graph((u,v))
        G_res.ndata['0'] = torch.tensor(obt).float()
        G_res.ndata['x'] = xyz_reps[:,None,:]
        G_res.edata['d'] = xyz_reps[v] - xyz_reps[u]
        G_res.edata['w'] = D1hot
        
        return G_res, r2amap

    def make_atm_graphs(self, xyz, grids, obt_fs, bnds, atmnames, CBonly=False, use_l1=False ):
        # Do KD-ball neighbour search -- grabbing a <dist neighbor from gridpoints
        t0 = time.time()
        kd      = scipy.spatial.cKDTree(xyz)
        indices = []
        kd_ca   = scipy.spatial.cKDTree(grids)
        indices = np.concatenate(kd_ca.query_ball_tree(kd, self.ball_radius))

        idx_ord = list(np.array(np.unique(indices),dtype=np.int16)) #atom indices close to any grid points
        if CBonly:
            idx_ord = [i for i in idx_ord if atmnames[i] in ['N','CA','C','O','CB']]

        if len(idx_ord) > 5000:
            return len(idx_ord)

        excl = np.isin(np.arange(len(xyz)),idx_ord) 
        excl = np.where(excl==False)[0]

        ## 1) Find edges that are chemical bonds
        isbnd = np.zeros((len(xyz),len(xyz)),dtype=np.int32)
        for i,j in bnds: isbnd[i,j] = 1
        isbnd[:,excl] = 0
        isbnd[excl,:] = 0

        ub,vb = np.where(isbnd==1)
            
        xyz     = xyz[idx_ord]
        natm = xyz.shape[0]-grids.shape[0] # real atoms
        
        bnds_bin = np.zeros((len(xyz),len(xyz)))
        newidx = {idx:i for i,idx in enumerate(idx_ord)}
        
        for i,j in zip(ub,vb):
            # Caution!! below was time consuming step
            #k,l = idx_ord.index(i),idx_ord.index(j) #list.index very inefficient
            k,l = newidx[i], newidx[j] 
            bnds_bin[k,l] = bnds_bin[l,k] = 1
        for i in range(len(xyz)): bnds_bin[i,i] = 1 #self
        
        # Concatenate coord & centralize xyz to ca.
        xyz = torch.tensor(xyz).float()

        ## 2) Connect by distance
        ## Graph connection: u,v are nedges & pairs to each other; i.e. (u[i],v[i]) are edges for every i
        t1 = time.time()
        u,v,dX = self.dist_fn_atm(xyz[None,], mode=self.edgemode, top_k=self.edgek[1],
                                  dcut=self.edgedist[1])
        t2 = time.time()

        ## reduce edge connections
        N = u.shape[0] # num edges
        D = torch.sqrt(torch.sum(dX[0]*dX[0],dim=-1))

        # prv: take if 1) real-X or 2) virtual-virtual but d < dcut
        incl = [i for i in range(N) if (u[i] < natm or v[i] < natm) or D[u[i],v[i]] < self.edgedist[0]]
        
        # for debug mode
        n1 = len([i for i in range(N) if (u[i] < natm or v[i] < natm)])
        n2 = len([i for i in range(N) if D[u[i],v[i]] < self.edgedist[0]])

        u = u[incl]
        v = v[incl]
        
        #if self.bndgraph_type == 'bonded':
        ub,vb = np.where(bnds_bin)
            
        # distance
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None]
        
        w1hot = distance_feature(self.distance_feat,w,0.5,5.0) # torch.tensor
        
        bnds_bin = torch.tensor(bnds_bin[v,u]).float() #replace first bin (0.0~0.5 Ang) to bond info
        w1hot[:,0] = bnds_bin # chemical bond

        grid_neighs = torch.tensor([float(i >= natm and j >= natm) for i,j in zip(u,v)]).float()
        w1hot[:,1] = grid_neighs # whether u,v are grid neighbors -- always 0 if split

        # concatenate all one-body-features
        obt  = []
        for i,f in enumerate(obt_fs):
            if len(f) > 0: obt.append(f[idx_ord])
        obt = np.concatenate(obt,axis=-1)

        ## Construct graphs
        # G_atm: graph for all atms
        G_atm = dgl.graph((u,v))
        G_atm.ndata['attr'] = torch.tensor(obt).float() 
        G_atm.edata['attr'] = w1hot #;  'd' previously
        G_atm.edata['rel_pos'] = dX[:,u,v].float()[0]

        #if use_l1:
        G_atm.ndata['x'] = xyz[:,None,:] # not a l1 feature; just info 
        #G_atm.edata['d'] = xyz[v] - xyz[u] #neccesary?
        #G_atm.edata['w'] = w1hot #neccesary? #unused

        te = time.time()
        print("took %.1f/%.1f/%.1f sec for processing "%(t1-t0,t2-t1,te-t2),
              G_atm.number_of_nodes(), G_atm.number_of_edges() )
        return G_atm
    
# Given a list of coordinates X, gets top-k neighbours based on eucledian distance
def get_dist_neighbors(X, mode="topk", top_k=16, dcut=4.5, eps=1E-6):
    """ Pairwise euclidean distances """
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = torch.sqrt(torch.sum(dX**2, 3) + eps)

    if mode in ['topk','mink']:
        D_neighbors, E_idx = torch.topk(D, top_k+1, dim=-1, largest=False)
        #exclude self-connection
        D_neighbor =  D_neighbors[:,:,1:]
        E_idx = E_idx[:,:,1:]
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)

        # append more than k that are within dcut
        if mode == 'mink':
            nprv = len(u)
            mask = torch.where(torch.tril(D)<1.0e-6,100.0,1.0)
            _,uD,vD = torch.where((mask*D)<dcut)
            uv = np.array(list(zip(u,v))+list(zip(uD,vD)))
            uv = np.unique(uv,axis=0)
            u = [a for a,b in uv]
            v = [b for a,b in uv]
            #print("nedge:",dcut,nprv,len(uD),len(u))
            
    elif mode == 'distT':
        # need to exclude self?
        mask = torch.where(torch.tril(D)<1.0e-6,100.0,1.0)
        _,u,v = torch.where((mask*D)<dcut)
        
    elif mode == 'dist':
        _,u,v = torch.where(D<dcut)

    return u,v,dX

def sample_uniform(fnats):
    return np.array([1.0 for _ in fnats])/len(fnats)

def upsample_category(cat):
    p = motif.sampling_weights[cat]
    return p/np.sum(p)

def get_affinity_1hot(affinity,digits,soften=0.5):
    # translate
    ibin = max(0,int(0.5*(affinity-2.0)))
    if ibin >= len(digits): ibin = len(digits)-1
    
    hot1 = np.eye(len(digits))[ibin]
    if soften > 0:
        hot1[:-1] += soften*np.copy(hot1[:-1])
        hot1[1:] += soften*np.copy(hot1[1:])
        hot1 /= 1.0 + 2.0*soften
    return hot1
    
def collate(samples):
    #samples should be G,info
    
    valid = [v[0] != None for v in samples]
    try:
        info  = [s[1] for s in samples]
        samples = [s[0] for s in samples]
        bG = dgl.batch(samples)

        # below contains l0 features only
        if 'x' in bG.ndata:
            node_feature = {"0": bG.ndata["attr"][:,:,None].float(), "1": bG.ndata["x"].float()} #1: 3D, N x 1 x 3
        else:
            node_feature = {"0": bG.ndata["attr"][:,:,None].float()}
        edge_feature = {"0": bG.edata["attr"][:,:,None].float()}
    
        return bG, node_feature, edge_feature, info
    except:
        print("failed collation")
        return None, {}, {}, {},

def distance_feature(mode,d,binsize=0.5,maxd=5.0):
    if mode == '1hot': 
        b = (d/binsize).long()
        nbin = int(maxd/binsize) # 0.0~5.0<
        b[b>=nbin] = nbin-1
        d1hot = torch.eye(nbin)[b].float()
        feat = d1hot.squeeze()
    elif mode == 'std': #sigmoid
        d0 = 0.5*(maxd-binsize) #center
        m = 5.0/d0 #slope
        feat = 1.0/(1.0+torch.exp(-m*(d-d0)))
        feat = feat.repeat(1,3) # hacky!! make as 3-dim
    return feat

def correlation_Pearson( pred, ans ):
    pred  = pred - torch.mean(pred)
    ans   = ans - torch.mean(ans)

    norm = torch.sum(pred*ans)
    denorm1 = torch.sqrt(torch.sum(pred*pred)+1e-6)
    denorm2 = torch.sqrt(torch.sum(ans*ans)+1e-6)
    
    return norm/denorm1/denorm2, denorm2

def load_dataset(set_params, generator_params, setsuffix):
    train_set = Dataset(np.load("data/train_proteins%s.npy"%setsuffix),
                        **set_params)
    
    val_set = Dataset(np.load("data/valid_proteins%s.npy"%setsuffix),
                      **set_params)

    train_generator = data.DataLoader(train_set,
                                      #worker_init_fn=lambda _: np.random.seed(),
                                      **generator_params)
    
    valid_generator = data.DataLoader(val_set,
                                      #worker_init_fn=lambda _: np.random.seed(),
                                      **generator_params)

    
    return train_generator,valid_generator
    
