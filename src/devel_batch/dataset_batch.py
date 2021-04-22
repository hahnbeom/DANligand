import sys
import numpy as np
import torch
import dgl
from .utils import *
from torch.utils import data
from os import listdir
from os.path import join, isdir, isfile
from scipy.spatial import distance, distance_matrix

#from torch.utils import data
from os import listdir
from os.path import join, isdir, isfile
from scipy.spatial import distance, distance_matrix
import scipy

sys.path.insert(0,'./')

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,
                 targets,
                 root_dir        = "./",
                 verbose         = False,
                 ball_radius     = 9.0,
                 displacement    = "",
                 randomize       = 0.0,
                 tag_substr      = [''],
                 upsample        = None,
                 num_channels    = 32,
                 affinity_digits = np.array([2,4,6,8,10,12,14]),
                 sasa_method     = "sasa",
                 bndgraph_type   = 'bonded',
                 edgemode        = 'dist',
                 edgek           = (  0,  0),
                 edgedist        = (8.0,4.5),
                 ballmode        = 'all',
                 distance_feat   = 'std',
                 debug           = False,
                 nsamples_per_p  = 1,
                 sample_mode     = 'random'
    ):
        
        self.proteins = targets
        
        self.datadir = root_dir
        self.verbose = verbose
        self.ball_radius = ball_radius
        self.randomize = randomize
        self.tag_substr = tag_substr
        self.sasa_method = sasa_method
        self.num_channels = num_channels
        self.bndgraph_type = bndgraph_type
        self.ballmode = ballmode #["com","all"]
        self.dist_fn_res = lambda x:get_dist_neighbors(x, mode=edgemode, top_k=edgek[0], dcut=edgedist[0])
        self.dist_fn_atm = lambda x:get_dist_neighbors(x, mode=edgemode, top_k=edgek[1], dcut=edgedist[1])
        self.debug = debug
        self.distance_feat = distance_feat
        self.nsamples_per_p = nsamples_per_p
        self.nsamples = max(1,len(self.proteins)*nsamples_per_p)
        self.sample_mode = sample_mode

        if upsample == None:
            self.upsample = sample_uniform
        else:
            self.upsample = upsample

        self.affinity_digits = affinity_digits

    def __len__(self):
        'Denotes the total number of samples'
        return int(self.nsamples)
    
    def __getitem__(self, index):
        # Select a sample decoy
        featuref = self.proteins[index]
        info = {}
        info['pname'] = featuref
        
        # New style
        '''
        try:
            samples = np.load(self.datadir+featuref,allow_pickle=True)
        except:
            print("BAD npz!", self.datadir+featuref)
            return False, False, False, info
        pindex = index%len(samples['name'])
        
        # ligand atms
        ligatms = samples['ligatms'][pindex]
        
        # coordinate
        xyz_lig = samples['xyz_lig'][pindex]
        xyz_rec = samples['xyz_rec'][pindex]
        xyz = np.concatenate([xyz_lig, xyz_rec])

        # residue representatives
        #repsatm_idx = samples['repsatm_idx'] #representative atm idx for each residue (e.g. CA); receptor only
        #repsatm_lig = samples['repsatm_lig']
        #repsatm_idx = np.concatenate([np.array([repsatm_lig]),np.array(repsatm_idx,dtype=int)+len(xyz_lig)])
        #r2a         = np.array(samples['residue_idx'],dtype=int) + 1 #add ligand as the first residue
        #r2a = np.concatenate([np.array([0 for _ in ligatms]),r2a])
        repsatm_idx = samples['repsatm_idx'][pindex]
        r2a = samples['r2a'][pindex]

        # residue features
        #sasa_lig = np.array([0.5 for _ in xyz_lig]) #neutral value
        #sasa = np.expand_dims(np.concatenate([sasa_lig,samples['sasa_rec']]),axis=1)
        sasa = np.expand_dims(samples['sasa'][pindex],axis=1)
        
        naas = len(residues_and_metals) + 1 #add "ligand type"
        aas = samples['aas'][pindex]
        aas1hot = np.eye(naas)[aas]
        
        # atom features
        #atypes = np.concatenate([samples['atypes_lig'], samples['atypes_rec']])
        atypes = samples['atypes'][pindex]
        atypes = np.array([gentype2num[at] for at in atypes]) # string to integers
        atypes = np.eye(max(gentype2num.values())+1)[atypes]  # convert integer to 1-hot
        
        #charges = np.expand_dims(np.concatenate([samples['charge_lig'], samples['charge_rec']]),axis=1)
        charges = np.expand_dims(samples['charge'][pindex],axis=1)
    
        # edge features
        #bnds_rec    = prop['bnds_rec'] + len(xyz_lig) #shift index;
        #bnds_lig    = samples['bnds_lig']
        #bnds    = np.concatenate([bnds_lig,bnds_rec])
        bnds = samples['bnds'][pindex]
        '''

        ## old but working style -- from deepaccnet_graph/dataset.py
        samples = np.load(self.datadir+'ref.lig.npz',allow_pickle=True)
        pindex = index
        sname   = samples['name'][pindex]
        info['pindex'] = pindex
        info['sname'] = sname
        
        prop = np.load(self.datadir+"ref.prop.npz")
        charges_rec = prop['charge_rec'] 
        atypes_rec  = prop['atypes_rec'] #1-hot
        aas         = prop['aas'] #rec only
        repsatm_idx = prop['repsatm_idx'] #representative atm idx for each residue (e.g. CA); receptor only
        r2a         = np.array(prop['residue_idx'],dtype=int) + 1 #add ligand as the first residue

        sasa_rec,cbcounts_rec = 0,0
        if 'sasa_rec' in prop: sasa_rec = prop['sasa_rec']
        if 'cbcounts_rec' in prop: cbcounts_rec = prop['cbcounts_rec']
        
        # get per-lig features
        xyz_lig = samples['xyz'][pindex]
        xyz_rec = samples['xyz_rec'][pindex]
        lddt    = samples['lddt'][pindex] # natm
        fnat    = samples['fnat'][pindex] # 1

        atypes_lig  = samples['atypes_lig'][pindex]
        bnds_lig    = samples['bnds_lig'][pindex]
        charges_lig = samples['charge_lig'][pindex]
        if 'repsatm_lig' in samples:
            repsatm_lig = samples['repsatm_lig'][pindex]
        else:
            repsatm_lig = 0

        # shift indices for ligands on receptor-only properties
        naas = len(residues_and_metals) + 1 #add "ligand type"
        aas = [naas-1 for _ in xyz_lig] + list(aas)
        aas1hot = np.eye(naas)[aas]

        r2a = np.concatenate([np.array([0 for _ in xyz_lig]),r2a])
        r2a1hot = np.eye(max(r2a)+1)[r2a]
        repsatm_idx = np.concatenate([np.array([repsatm_lig]),np.array(repsatm_idx,dtype=int)+len(xyz_lig)])
    
        #affinity = -1.0
        #if 'affinity' in samples:    affinity    = samples['affinity'][pindex]
        #affinity1hot = get_affinity_1hot(affinity, self.affinity_digits)

        # bond properties
        bnds_rec    = prop['bnds_rec'] + len(xyz_lig) #shift index;
        
        # concatenate rec & ligand: ligand comes first
        charges = np.expand_dims(np.concatenate([charges_lig, charges_rec]),axis=1)
            
        xyz = np.concatenate([xyz_lig, xyz_rec])
        atypes = np.concatenate([atypes_lig, atypes_rec])
        atypes = np.array([find_gentype2num(at) for at in atypes]) # string to integers
        atypes = np.eye(max(gentype2num.values())+1)[atypes] # convert integer to 1-hot
        
        sasa = []
        sasa_lig = np.array([0.5 for _ in xyz_lig]) #neutral value

        if self.sasa_method == 'cbcounts':
            sasa = np.concatenate([sasa_lig,cbcounts_rec])
            sasa = np.expand_dims(sasa,axis=1)
        elif self.sasa_method == 'sasa':
            sasa = np.concatenate([sasa_lig,sasa_rec])
            sasa = np.expand_dims(sasa,axis=1)
            
        bnds    = np.concatenate([bnds_lig,bnds_rec])
        
        # orient around ligand-COM
        center_xyz = np.mean(xyz_lig, axis=0)[None,:] #2-D required...
        xyz = xyz - center_xyz
        xyz_lig = xyz_lig - center_xyz
        center_xyz[:,:] = 0.0
        ball_xyzs = [a[None,:] for a in xyz_lig]

        #try:
        if True:
            G_bnd, G_atm, idx_ord = self.make_atm_graphs(xyz, ball_xyzs,
                                                         [aas1hot,atypes,sasa,charges],
                                                         bnds, len(xyz_lig) )

            rsds_ord = r2a[idx_ord]
            G_res, r2amap = self.make_res_graph(xyz, center_xyz, [aas1hot,sasa],
                                                repsatm_idx, rsds_ord)

            # store which indices go to ligand atms
            ligidx = np.zeros((len(idx_ord),len(xyz_lig)))
            for i in range(len(xyz_lig)): ligidx[i,i] = 1.0
            info['ligidx'] = torch.tensor(ligidx).float()

        #except:
        else:
            return False, False, False, info

        info['sname'] = samples['name'][pindex]
        info['r2amap'] = torch.tensor(r2amap).float()
        info['r2a']   = torch.tensor(rsds_ord).float()
        info['repsatm_idx'] = torch.tensor(repsatm_idx).float()
        info['ligatms'] = ['A' for _ in xyz_lig]#samples['ligatms'][pindex]
        
        return G_bnd, G_atm, G_res, info
            
    def make_res_graph(self, xyz, center_xyz, obt_fs, repsatm_idx, rsds_in_Gatm):
        xyz_reps = xyz[repsatm_idx]
        xyz_reps = torch.tensor(xyz_reps-center_xyz).float()

        # Grabbing a < dist neighbor
        kd      = scipy.spatial.cKDTree(xyz_reps)
        kd_ca   = scipy.spatial.cKDTree(center_xyz)
        indices = kd_ca.query_ball_tree(kd, 1000.0)[0] #any huge number to cover entire protein
        reps_idx = [repsatm_idx[i] for i in indices]

        # which idx in reps_idx map to idx_in_Gatm; i.e. what becomes residue-embedding in G_atm
        # rsds_in_Gatm: resno following idx_ord
        r2amap = np.zeros(len(rsds_in_Gatm), dtype=int)
        for i,rsd in enumerate(rsds_in_Gatm):
            if rsd in indices: r2amap[i] = indices.index(rsd)
            else: sys.exit("unexpected resno %d"%(rsd)) #should not happen
        # many entries unused in Gatm may left as 0
        r2amap = np.eye(max(indices)+1)[r2amap]

        #print(r2amap.shape)
        #for i,v in enumerate(r2amap): print(i,np.where(v==1))
                
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
        #G_res.edata['w'] = D[...,None].repeat(1,2) #for multihead
        G_res.edata['w'] = D1hot
        
        return G_res, r2amap

    def make_atm_graphs(self, xyz, ball_xyzs, obt_fs, bnds, nlig ):
        # Do KD-ball neighbour search -- grabbing a <dist neighbor from ligcom
        kd      = scipy.spatial.cKDTree(xyz)
        indices = []
        for ball_xyz in ball_xyzs:
            kd_ca   = scipy.spatial.cKDTree(ball_xyz)
            indices += kd_ca.query_ball_tree(kd, self.ball_radius)[0]
        indices = np.unique(indices)

        # make sure ligand atms are ALL INCLUDED
        idx_ord = [i for i in indices if i < nlig]
        idx_ord += [i for i in indices if i >= nlig]

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

        if self.bndgraph_type == 'bonded':
            ub,vb = np.where(bnds_bin)
        elif self.bndgraph_type == 'mink':
            # delete protein-ligand connection
            mono_edges = [i for i,a in enumerate(u) if (a < nlig and v[i] < nlig) or (a >= nlig and v[i] >= nlig)]
            ub = u[mono_edges]
            vb = v[mono_edges]
            
        # distance
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None]
        w1hot = distance_feature(self.distance_feat,w,0.5,5.0)
        bnds_bin = torch.tensor(bnds_bin[v,u]).float() #replace first bin (0.0~0.5 Ang) to bond info
        w1hot[:,0] = bnds_bin

        ## Construct graphs
        # G_atm: graph for all atms
        G_atm = dgl.graph((u,v))
        # uninitialize?
        #G_atm.ndata['0'] = torch.zeros((len(xyz),self.num_channels)).float() #empty spot
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
        #_,u,v = torch.where(D<dcut)
        
    elif mode == 'dist':
        _,u,v = torch.where(D<dcut)
    #print(dcut,len(u),len(u2))

    return u,v

def sample_uniform(fnats):
    return np.array([1.0 for _ in fnats])/len(fnats)

def get_affinity_1hot(affinity,digits,soften=0.5):
    hot1 = np.eye(len(digits))[affinity]
    if soften > 0:
        hot1[:-1] += soften*np.copy(hot1[:-1])
        hot1[1:] += soften*np.copy(hot1[1:])
        hot1 /= 1.0 + 2.0*soften
    return hot1
    
def collate(samples):
    graphs_bnd, graphs_atm, graphs_res, info = map(list, zip(*samples))
    nbatch = len(samples)
    binfo = {'pname':[],'sname':[],'ligidx':[],'ligatms':[]}
    
    if True:
        bgraph_bnd = dgl.batch(graphs_bnd)
        bgraph_atm = dgl.batch(graphs_atm)
        bgraph_res = dgl.batch(graphs_res)

        #info part
        batch_nress = bgraph_res.batch_num_nodes()
        batch_natms = bgraph_atm.batch_num_nodes()
        
        r2a = np.array([])
        for k in range(nbatch):
            binfo['pname'].append(info[k]['pname'])
            binfo['sname'].append(info[k]['sname'])
            binfo['ligidx'].append(info[k]['ligidx'])
            binfo['ligatms'].append(info[k]['ligatms'])
            
            r2a_k = info[k]['r2a'] #size natm, value resno
            if k > 0: r2a_k += batch_nress[k]
            r2a = np.concatenate([r2a,r2a_k])
            
        r2a = r2a.astype(int)
        r2a1hot = np.eye(sum(batch_nress))[r2a]
        binfo['r2a'] = torch.tensor(r2a1hot).float()
        
    else:
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
