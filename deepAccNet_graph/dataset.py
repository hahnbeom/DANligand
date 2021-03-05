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
                 dist_fn, # lambda x:get_dist_neighbors(x, top_k=N)
                 root_dir        = "/projects/ml/ligands/v4/",
                 verbose         = False,
                 useTipNode      = False,
                 ball_radius     = 10,
                 displacement    = "",
                 randomize       = 0.0,
                 tag_substr      = [''],
                 upsample        = None,
                 num_channels    = 32,
                 affinity_digits = np.array([2,4,6,8,10,12,14]),
                 sasa_method     = "none"
    ):
        
        self.dist_fn = dist_fn 
        self.datadir = root_dir
        self.verbose = verbose
        self.proteins = targets
        self.ball_radius = ball_radius
        self.randomize = randomize
        self.tag_substr = tag_substr
        self.sasa_method = sasa_method
        self.num_channels = num_channels
        
        if upsample == None:
            self.upsample = sample_uniform
        else:
            self.upsample = upsample

        self.affinity_digits = affinity_digits

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.proteins)
    
    def __getitem__(self, index):
        #index = 76 #debugging
        
        # Select a sample decoy 
        pname = self.proteins[index]

        info = {}
        info['pname'] = pname
        info['sname'] = 'none'

        try:
            samples, pindex = self.get_random_sample(pname)
        except:
            print("BAD npz!", self.datadir+pname+".lignpz")
            return False, False, False, info
        
        sname   = samples['name'][pindex]
        info['sname'] = sname
        
        # receptor features that go into se3        
        prop = np.load(self.datadir+pname+".prop.npz")
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

        # shift indices for ligands on receptor-only properties
        naas = len(residues_and_metals) + 1 #add "ligand type"
        aas = [naas-1 for _ in xyz_lig] + list(aas)
        aas1hot = np.eye(naas)[aas]

        r2a = np.concatenate([np.array([0 for _ in xyz_lig]),r2a])
        r2a1hot = np.eye(max(r2a)+1)[r2a]
        repsatm_idx = np.concatenate([np.array([0]),np.array(repsatm_idx,dtype=int)+len(xyz_lig)])
    
        #affinity = -1.0
        #if 'affinity' in samples:    affinity    = samples['affinity'][pindex]
        #affinity1hot = get_affinity_1hot(affinity, self.affinity_digits)

        # bond properties
        bnds_rec    = prop['bnds_rec'] + len(xyz_lig) #shift index;
        
        # concatenate rec & ligand: ligand comes first
        charges = np.expand_dims(np.concatenate([charges_lig, charges_rec]),axis=1)
        xyz = np.concatenate([xyz_lig, xyz_rec])
        atypes = np.concatenate([atypes_lig, atypes_rec])
        atypes = np.array([gentype2num[at] for at in atypes]) # string to integers
        atypes = np.eye(max(gentype2num.values())+1)[atypes] # convert integer to 1-hot
        
        sasa = []
        sasa_lig = np.array([0.5 for _ in xyz_lig]) #neutral value

        if self.sasa_method == 'cbcounts':
            sasa = np.concatenate([sasa_lig,cbcounts_rec])
            sasa = np.expand_dims(sasa,axis=1)
        elif self.sasa_method == 'sasa':
            sasa = np.concatenate([sasa_lig,sasa_rec])
            sasa = np.expand_dims(sasa,axis=1)
            
        center_xyz = np.mean(xyz_lig, axis=0)[None,:] #2-D required...
        bnds    = np.concatenate([bnds_lig,bnds_rec])

        try:
            G_bnd, G_atm, idx_ord = self.make_atm_graphs(xyz, center_xyz,
                                                         [aas1hot,atypes,sasa,charges],
                                                         bnds, len(xyz_lig))

            rsds_ord = r2a[idx_ord]
            G_res, r2amap = self.make_res_graph(xyz, center_xyz, [aas1hot,sasa],
                                                repsatm_idx, rsds_ord)

            # store which indices go to ligand atms
            ligidx = np.zeros((len(idx_ord),len(xyz_lig)))
            for i in range(len(xyz_lig)): ligidx[i,i] = 1.0
            info['ligidx'] = torch.tensor(ligidx).float()

        except:
            return False, False, False, info

        
        '''
        try:
            G_bnd, G_atm, idx_ord = self.make_atm_graphs(xyz, center_xyz,
                                                         [aas1hot,atypes,charges,sasa],
                                                         bnds, len(xyz_lig))

            rsds_ord = r2a[idx_ord]
            G_res, r2amap = self.make_res_graph(xyz, center_xyz, [aas1hot,sasa],
                                                repsatm_idx, rsds_ord)

            # store which indices go to ligand atms
            ligidx = np.zeros((len(idx_ord),len(xyz_lig)))
            for i in range(len(xyz_lig)): ligidx[i,i] = 1.0
            info['ligidx'] = torch.tensor(ligidx).float()
            
        except:
            return False, False, False, info
        '''
        
        info['fnat']  = torch.tensor(fnat).float()
        info['lddt']  = torch.tensor(lddt).float()
        info['r2amap'] = torch.tensor(r2amap).float()
        info['r2a']   = torch.tensor(r2a1hot).float()
        info['repsatm_idx'] = torch.tensor(repsatm_idx).float()
        
        return G_bnd, G_atm, G_res, info
            
    def get_random_sample(self,pname):
        pindices = []
        
        samples = np.load(self.datadir+pname+".lig.npz",allow_pickle=True)
        for substr in self.tag_substr:
            pindices += [i for i,n in enumerate(samples['name']) if substr in n]
        fnats = np.array([samples['fnat'][i] for i in pindices])
        pindex  = np.random.choice(pindices,p=self.upsample(fnats))
        
        return samples, pindex

    def make_res_graph(self, xyz, center_xyz, obt_fs, repsatm_idx, rsds_in_Gatm):
        xyz_reps = xyz[repsatm_idx]
        xyz_reps = torch.tensor(xyz_reps-center_xyz).float()

        # Grabbing a < dist neighbor
        kd      = scipy.spatial.cKDTree(xyz_reps)
        kd_ca   = scipy.spatial.cKDTree(center_xyz)
        indices = kd_ca.query_ball_tree(kd, 1000.0)[0] #any huge number to cover entire protein
        reps_idx = [repsatm_idx[i] for i in indices]

        # which idx in reps_idx map to idx_in_Gatm; i.e. what becomes residue-embedding in G_atm
        r2amap = np.zeros(len(rsds_in_Gatm), dtype=int)
        for i,rsd in enumerate(rsds_in_Gatm):
            if rsd in indices: r2amap[i] = indices.index(rsd)
        r2amap = np.eye(max(indices)+1)[r2amap]
                
        #for k in range(30,50): print(indices[k],rsds_in_Gatm[k],r2amap[k])
        obt  = []
        for f in obt_fs:
            if len(f) > 0: obt.append(f[reps_idx])
        obt = np.concatenate(obt,axis=-1)
        
        D_neighbors, E_idx = self.dist_fn(xyz_reps[None,])
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)

        G_res = dgl.graph((u,v))
        G_res.ndata['0'] = torch.tensor(obt).float()
        G_res.ndata['x'] = xyz_reps[:,None,:]
        G_res.edata['d'] = xyz_reps[v] - xyz_reps[u]
        G_res.edata['w'] = torch.sqrt(torch.sum((xyz_reps[v] - xyz_reps[u])**2, axis=-1)+1e-6)[...,None].repeat(1,2) #for multihead
        
        return G_res, r2amap
    
    def make_atm_graphs(self, xyz, center_xyz,
                        obt_fs, bnds, nlig):
        # Do KD-ball neighbour search -- grabbing a <dist neighbor from ligcom
        kd      = scipy.spatial.cKDTree(xyz)
        kd_ca   = scipy.spatial.cKDTree(center_xyz)
        indices = kd_ca.query_ball_tree(kd, self.ball_radius)

        # make sure ligand atms are ALL INCLUDED
        idx_ord = [i for i in indices[0] if i < nlig]
        idx_ord += [i for i in indices[0] if i >= nlig]

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
        xyz = xyz - center_xyz
        xyz = torch.tensor(xyz).float()

        # randomize coordinate
        if self.randomize > 1e-3:
            randxyz = 2.0*self.randomize*(0.5 - torch.rand((len(xyz),3)))
            xyz = xyz + randxyz
        
        ## Graph connection: u,v are nedges & pairs to each other; i.e. (u[i],v[i]) are edges for every i
        # for G_atm
        D_neighbors, E_idx = self.dist_fn(xyz[None,])
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)
        ub,vb = np.where(bnds_bin)
        
        # distance
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None].repeat(1,2)

        ## Construct graphs
        # G_atm: graph for all atms
        G_atm = dgl.graph((u,v))
        # uninitialize?
        #G_atm.ndata['0'] = torch.zeros((len(xyz),self.num_channels)).float() #empty spot
        G_atm.ndata['x'] = xyz[:,None,:]
        G_atm.edata['d'] = xyz[v] - xyz[u]
        G_atm.edata['w'] = w
        G_atm.edata['w'][:,1] = torch.tensor(bnds_bin[v,u]).float() #hack to fill in 2nd column

        # G_bnd: graph for only bonded atms
        G_bnd = dgl.graph((ub,vb))
        G_bnd.ndata['0'] = torch.tensor(obt).float()
        G_bnd.edata['d'] = xyz[vb] - xyz[ub]
        w = torch.sqrt(torch.sum((xyz[vb] - xyz[ub])**2, axis=-1)+1e-6)[...,None].repeat(1,2)
        G_bnd.edata['w'] = w #bond info not necessary
 
        return G_bnd, G_atm, idx_ord
    
# Given a list of coordinates X, gets top-k neighbours based on eucledian distance
def get_dist_neighbors(X, top_k=16, eps=1E-6):
    """ Pairwise euclidean distances """
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = torch.sqrt(torch.sum(dX**2, 3) + eps)

    D_neighbors, E_idx = torch.topk(D, top_k+1, dim=-1, largest=False)
    #exclude self-connection
    return D_neighbors[:,:,1:], E_idx[:,:,1:]

def parse_pdbfile(pdbfile):
    file = open(pdbfile, "r")
    lines = file.readlines()
    file.close()
    
    lines = [l for l in lines if l.startswith("ATOM")]
    output = {}
    for line in lines:
        if line[13] != "H": 
            aidx = int(line[6:11])
            aname = line[12:16].strip()
            rname = line[17:20].strip()
            cname = line[21].strip()
            rindex = int(line[22:26])
            xcoord = float(line[30:38])
            ycoord = float(line[38:46])
            zcoord = float(line[46:54])
            occupancy = float(line[54:60])

            temp = dict(aidx = aidx,
                        aname = aname,
                        rname = rname,
                        cname = cname,
                        rindex = rindex,
                        x = xcoord,
                        y = ycoord,
                        z = zcoord,
                        coord = (xcoord,ycoord,zcoord),
                        occupancy = occupancy)

            residue = output.get(rindex, {})
            residue[aname] = temp
            output[rindex] = residue
        
    output2 = []
    keys = [i for i in output.keys()]
    keys.sort()
    for k in keys:
        temp = output[k]
        temp["rindex"] = k
        temp["rname"] = temp["CA"]["rname"]
        output2.append(temp)
        
    return output2

def positional_embedding(length, dup=False, d=20, dmax=80):
    if not dup: 
        index = np.arange(length)
    else:
        index = np.floor(np.arange(0, length, 0.5))
    
    output = []
    for i in range(d):
        coef = (1/(10000**(2*i/dmax)))
        output.append(np.sin(coef*index))
        output.append(np.cos(coef*index))
                      
    return np.array(output)

def get_lddt(decoy, ref, cutoff=15, threshold=[0.5, 1, 2, 4]):
   
    # only use parts that are less than 15A in ref structure
    mask = ref < cutoff
    for i in range(mask.shape[0]):
        mask[i,i]=False
   
    # Get interactions that are conserved
    conservation = []
    for th in threshold:
        temp = np.multiply((np.abs(decoy-ref) < th), mask)
        conservation.append(np.sum(temp, axis=0)/np.sum(mask, axis=0))
    return np.mean(conservation, axis=0)

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
    try:
        bgraph_bnd = dgl.batch(graphs_bnd)
        bgraph_atm = dgl.batch(graphs_atm)
        bgraph_res = dgl.batch(graphs_res)
    except:
        bgraph_bnd,bgraph_atm,bgraph_res = False,False,False
    
    return bgraph_bnd, bgraph_atm, bgraph_res, info

def correlation_Pearson( pred, ans ):
    pred  = pred - torch.mean(pred)
    ans   = ans - torch.mean(ans)

    norm = torch.sum(pred*ans)
    denorm1 = torch.sqrt(torch.sum(pred*pred)+1e-6)
    denorm2 = torch.sqrt(torch.sum(ans*ans)+1e-6)
    
    return norm/denorm1/denorm2, denorm2
