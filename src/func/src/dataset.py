import sys
import numpy as np
import torch
import dgl
from torch.utils import data
from os import listdir
from os.path import join, isdir, isfile
from scipy.spatial import distance, distance_matrix
import scipy

#sys.path.insert(0,'./')
from . import myutils

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,
                 targets,
                 root_dir        = "",
                 verbose         = False,
                 ball_radius     = 10,
                 displacement    = "",
                 randomize_lig   = 0.5,
                 randomize       = 0.0,
                 tag_substr      = [''],
                 upsample        = None,
                 num_channels    = 32,
                 sasa_method     = "sasa",
                 edgemode        = 'topk',
                 edgek           = (12,12),
                 edgedist        = (8.0,4.5),
                 ballmode        = 'com',
                 distance_feat   = 'std',
                 debug           = False,
                 nsamples_per_p  = 1,
                 CBonly          = False,
                 xyz_as_bb       = False, 
                 sample_mode     = 'random',
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
        self.dist_fn_res = lambda x:get_dist_neighbors(x, mode=edgemode, top_k=edgek[0], dcut=edgedist[0])
        self.dist_fn_atm = lambda x:get_dist_neighbors(x, mode=edgemode, top_k=edgek[1], dcut=edgedist[1])
        self.debug = debug
        self.distance_feat = distance_feat
        self.nsamples_per_p = nsamples_per_p
        self.xyz_as_bb = xyz_as_bb
        self.sample_mode = sample_mode
        self.CBonly = CBonly

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
        info = {}
        info['sname'] = 'none'
        
        skip_this = False
        if self.nsamples_per_p == 0:
            skip_this = True
        else:
            ip = int(index/self.nsamples_per_p)
            if ip >= len(self.proteins): skip_this = True

        # "proteins" in a format of "protein.idx"
        pname = self.proteins[ip].split('.')[0]
        pindex = int(self.proteins[ip].split('.')[-1])
            
        samples = np.load(self.datadir+pname+'.lig.npz',allow_pickle=True) #motif type & crd only
        if pindex >= len(samples['name']): skip_this = True

        #print("?",self.proteins[ip],pindex,len(samples['name']),skip_this)
        if skip_this:
            info['pname'] = 'none'
            print("failed to read input npz")
            return False, False, False, info

        info['pindex'] = pindex
        info['pname'] = pname
        info['sname'] = samples['name'][pindex]

        rot_motif,motifidx = None,-1
        xyz_motif  = samples['xyz'][pindex] #vector; set motif position at the origin
        if self.xyz_as_bb:
            if 'xyz_bb' in samples:
                #take CA only for consistency -- should figure out how to use the full bb crds
                #print(samples['xyz_bb'][pindex].shape)
                xyz_pred   = samples['xyz_bb'][pindex][1] #vector
            else:
                sys.exit("xyz_bb requested but not exist in npz, failed!")
        else:
            xyz_pred   = xyz_motif
        #print(samples['xyz_bb'][pindex][1], samples['xyz'][pindex])

        if 'rot' in samples: rot_motif   = samples['rot'][pindex] #quaternion"s" (n,4)
        if 'cat' in samples: motifidx    = samples['cat'][pindex] #integer
        
        # receptor features that go into se3        
        prop = np.load(self.datadir+pname+".prop.npz")
        xyz         = prop['xyz_rec']
        charges_rec = prop['charge_rec'] 
        atypes_rec  = prop['atypes_rec'] #1-hot
        anames      = prop['atmnames']
        aas_rec     = prop['aas_rec'] #rec only

        #mask neighbor res
        if 'exclude' in samples:
            reschains   = [a[0] for a in prop['atmres_rec']]
            exclrc = samples['exclude'][pindex]
            if len(exclrc) > 0:
                iexcl = np.array([i for i,rc in enumerate(reschains) if rc in exclrc],dtype=int)
                # send away excl atms not to screw up indexing...
                xyz[iexcl] += 100.0
                #print("exclude!", info['sname'], iexcl, exclrc, xyz[iexcl[0]])

        '''
        if CBonly:
            ibbs = [i for i,a in enumerate(anames) if a in ['N','CA','C','CB','O']]
            xyz = xyz[ibbs]
            charges_rec = charges_rec[ibbs]
            atypes_rec  = atypes_rec[ibbs]
            aas_rec     = aas_rec[ibbs]
        '''
        
        # per-res
        repsatm_idx = prop['repsatm_idx'] #representative atm idx for each residue (e.g. CA); receptor only
        repsatm_idx = np.array(repsatm_idx,dtype=int)
        #print(len(repsatm_idx))
        if len(repsatm_idx) > 3000:
            return False, False, False, info
        
        # representative atoms
        r2a         = np.array(prop['residue_idx'],dtype=int)
        r2a1hot = np.eye(max(r2a)+1)[r2a]
        aas1hot = np.eye(myutils.N_AATYPE)[aas_rec]

        sasa_rec = prop['sasa_rec']
        #sasa_rec = sasa_rec[ibbs]
        sasa    = np.expand_dims(sasa_rec,axis=1)
        
        # bond properties
        bnds    = prop['bnds_rec']
        charges = np.expand_dims(charges_rec,axis=1)
        atypes  = np.array([myutils.find_gentype2num(at) for at in atypes_rec]) # string to integers
        atypes  = np.eye(max(myutils.gentype2num.values())+1)[atypes] # convert integer to 1-hot
            
        # randomize motif coordinate
        dxyz = np.zeros(3)
        if self.randomize_lig > 1e-3:
            dxyz = 2.0*self.randomize_lig*(0.5 - np.random.rand(3))

        # orient around motif + delta
        xyz = xyz - xyz_motif + dxyz[None,:]
        xyz_pred = xyz_pred - xyz_motif + dxyz[None,:] #==dxyz if not-bb
        
        # randomize the rest coordinate
        if self.randomize > 1e-3:
            randxyz = 2.0*self.randomize*(0.5 - np.random.rand(len(xyz),3))
            xyz = xyz + randxyz

        center_xyz = np.zeros((1,3))
        ball_xyzs = [center_xyz] #origin

        try:
            G_bnd, G_atm, idx_ord = self.make_atm_graphs(xyz, ball_xyzs,
                                                         [aas1hot,atypes,sasa,charges],
                                                         bnds, anames, self.CBonly)

            rsds_ord = r2a[idx_ord]
            G_res, r2amap = self.make_res_graph(xyz, center_xyz, [aas1hot,sasa],
                                                repsatm_idx, rsds_ord)
            #if G_atm.number_of_nodes() > 
            #print("Size:", len(xyz), len(repsatm_idx), G_atm.number_of_nodes(), G_res.number_of_nodes())

        except:
            if self.debug:
                G_bnd, G_atm, idx_ord = self.make_atm_graphs(xyz, ball_xyzs,
                                                             [aas1hot,atypes,sasa,charges],
                                                             bnds, anames, self.CBonly )

                rsds_ord = r2a[idx_ord]
                G_res, r2amap = self.make_res_graph(xyz, center_xyz, [aas1hot,sasa],
                                                    repsatm_idx, rsds_ord)
                
            print("graphgen fail")
            return False, False, False, info

        if G_atm.number_of_nodes() > 500:
            return False, False, False, info
            
        
        info['dxyz']  = xyz_pred 
        info['xyz']   = xyz_motif # where the origin is set
        info['rot']   = rot_motif # may potentially perturb later
        info['r2a'] = torch.tensor(r2amap).float() #originally stored as r2amap

        # for testing w/o answer
        #if motifidx != -1: info['motifidx']   = motifidx
        info['motifidx']   = motifidx
        
        #info['r2a']   = torch.tensor(r2a1hot).float()
        #info['r2a']   = r2a1hot
        info['repsatm_idx'] = torch.tensor(repsatm_idx).float()

        return G_bnd, G_atm, G_res, info
            
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
        #print(len(repsatm_idx))
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

    def make_atm_graphs(self, xyz, ball_xyzs, obt_fs, bnds, atmnames, CBonly=False ):
        # Do KD-ball neighbour search -- grabbing a <dist neighbor from ligcom
        kd      = scipy.spatial.cKDTree(xyz)
        indices = []
        for ball_xyz in ball_xyzs:
            kd_ca   = scipy.spatial.cKDTree(ball_xyz)
            indices += kd_ca.query_ball_tree(kd, self.ball_radius)[0]
        idx_ord = list(np.unique(indices))
        if CBonly:
            idx_ord = [i for i in idx_ord if atmnames[i] in ['N','CA','C','O','CB']]

        # concatenate all one-body-features
        obt  = []
        for f in obt_fs:
            if len(f) > 0: obt.append(f[idx_ord])
        obt = np.concatenate(obt,axis=-1)
        
        xyz     = xyz[idx_ord]
        bnds    = [bnd for bnd in bnds if (bnd[0] in idx_ord) and (bnd[1] in idx_ord)]
        bnds_bin = np.zeros((len(xyz),len(xyz)))
        for i,j in bnds:
            k,l = idx_ord.index(i),idx_ord.index(j)
            bnds_bin[k,l] = bnds_bin[l,k] = 1
        for i in range(len(xyz)): bnds_bin[i,i] = 1 #self
        
        # Concatenate coord & centralize xyz to ca.
        xyz = torch.tensor(xyz).float()
        
        ## Graph connection: u,v are nedges & pairs to each other; i.e. (u[i],v[i]) are edges for every i
        # for G_atm
        u,v = self.dist_fn_atm(xyz[None,])

        #if self.bndgraph_type == 'bonded':
        ub,vb = np.where(bnds_bin)
        
        '''
        elif self.bndgraph_type == 'mink':
            # delete protein-ligand connection
            mono_edges = [i for i,a in enumerate(u) if (a < nlig and v[i] < nlig) or (a >= nlig and v[i] >= nlig)]
            ub = u[mono_edges]
            vb = v[mono_edges]
        '''
            
        # distance
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None]
        w1hot = distance_feature(self.distance_feat,w,0.5,5.0)
        bnds_bin = torch.tensor(bnds_bin[v,u]).float() #replace first bin (0.0~0.5 Ang) to bond info
        w1hot[:,0] = bnds_bin

        ## Construct graphs
        # G_atm: graph for all atms
        G_atm = dgl.graph((u,v))
        G_atm.ndata['x'] = xyz[:,None,:]
        G_atm.edata['d'] = xyz[v] - xyz[u]
        G_atm.edata['w'] = w1hot

        # G_bnd: graph for only bonded atms
        G_bnd = dgl.graph((ub,vb))
        G_bnd.ndata['0'] = torch.tensor(obt).float()
        G_bnd.edata['d'] = xyz[vb] - xyz[ub]
        w = torch.sqrt(torch.sum((xyz[vb] - xyz[ub])**2, axis=-1)+1e-6)[...,None]
        w1hot = distance_feature(self.distance_feat,w,0.5,5.0)
        G_bnd.edata['w'] = w1hot #bond info not necessary
 
        return G_bnd, G_atm, idx_ord
    
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

    return u,v

def sample_uniform(fnats):
    return np.array([1.0 for _ in fnats])/len(fnats)

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
    graphs_bnd, graphs_atm, graphs_res, info = map(list, zip(*samples))
    B = len(samples)
    binfo = {'pname':[],'sname':[],'ligidx':[],'motifidx':[],'dxyz':[],'rot':[],'r2a':[]}

    try:
    #if True:
        if not graphs_bnd[0]:
            return False, False, False, {'pname':"graphfail",'sname':"graphfail"}
            
        #for k in range(B):
        #    print(k, info[k]['sname'], graphs_bnd[k], graphs_atm[k])
        
        bgraph_bnd = dgl.batch(graphs_bnd)
        bgraph_atm = dgl.batch(graphs_atm)
        bgraph_res = dgl.batch(graphs_res)

        #info part
        batch_nress = bgraph_res.batch_num_nodes()
        batch_natms = bgraph_atm.batch_num_nodes()

        #r2a = np.array([[]])
        ligidx = np.zeros((B,batch_natms))
        shift = 0
        for k in range(B):
            binfo['sname'].append(info[k]['sname'])
            binfo['pname'].append(info[k]['pname'])
            binfo['motifidx'].append(info[k]['motifidx'])
            binfo['dxyz'].append(info[k]['dxyz'])
            binfo['rot'].append(info[k]['rot'])
            
            ligidx[k,shift] = 1.0
            shift += batch_natms[k]

            # TODO: r2a
            #r2a_k = info[k]['r2a'] #size natm, value resno
            #if k > 0: r2a_k += batch_nress[k]
            #r2a = np.concatenate([r2a,r2a_k])

        # tmp
        r2a = info[0]['r2a'] #size natm, value resno
        #r2a = r2a.astype(int)
        #r2a1hot = np.eye(sum(batch_nress))[r2a]
        
        binfo['r2a'] = r2a #torch.tensor(r2a).float()
        binfo['motifidx'] = torch.tensor(binfo['motifidx']).float()
        binfo['ligidx'] = torch.tensor(ligidx).float()
        binfo['dxyz'] = torch.tensor(binfo['dxyz']).float()
        binfo['rot'] = torch.tensor(binfo['rot']).float()
    except:
    #else:
        print("Warning; failed to collate graph")
        bgraph_bnd,bgraph_atm,bgraph_res = False,False,False

    return bgraph_bnd, bgraph_atm, bgraph_res, binfo

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
        feat = feat.repeat(1,2) # hacky!! make as 2-dim
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
                                      worker_init_fn=lambda _: np.random.seed(),
                                      **generator_params)
    
    valid_generator = data.DataLoader(val_set,
                                      worker_init_fn=lambda _: np.random.seed(),
                                      **generator_params)

    
    return train_generator,valid_generator
    
