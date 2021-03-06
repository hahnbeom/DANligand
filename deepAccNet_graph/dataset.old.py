import sys
import numpy as np
import torch
import dgl
from .utils import *
from torch.utils import data
from os import listdir
from os.path import join, isdir, isfile
from scipy.spatial import distance, distance_matrix

from torch.utils import data
from os import listdir
from os.path import join, isdir, isfile
from scipy.spatial import distance, distance_matrix
import scipy

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,
                 targets,
                 dist_fn, # lambda x:get_dist_neighbors(x, top_k=N)
                 root_dir        = "/projects/ml/ligands/v3/",
                 verbose         = False,
                 useTipNode      = False,
                 ball_radius     = 10,
                 displacement    = "",
                 randomize       = 0.0,
                 tag_substr      = [''],
                 upsample        = None,
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
        
        if upsample == None:
            self.upsample = sample_uniform
        else:
            self.upsample = upsample

        self.affinity_digits = affinity_digits

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.proteins)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select a sample decoy 
        pname = self.proteins[index]
        
        info = {}
        info['stat'] = True
        info['pname'] = pname
        info['sname'] = 'none'
        pindices = []
        stat = True
        
        try:
            samples = np.load(self.datadir+pname+".lig.npz",allow_pickle=True)
            for substr in self.tag_substr:
                pindices += [i for i,n in enumerate(samples['name']) if substr in n]
        except:
            stat = False
        
        if len(pindices) == 0 or not stat:
            print("BAD npz!", self.datadir+pname+".lignpz")
            return False, 0.0, 0.0, info
        
        fnats   = np.array([samples['fnat'][i] for i in pindices])
        pindex  = np.random.choice(pindices,p=self.upsample(fnats))
        #print(pindices, [samples['name'][i] for i in pindices])
        
        # receptor features that go into se3
        prop = np.load(self.datadir+pname+".prop.npz")
        charges_lig = prop['charge_lig'] 
        atypes_lig  = prop['atypes_lig'] #1-hot
        charges_rec = prop['charge_rec'] 
        atypes_rec  = prop['atypes_rec'] #1-hot
        #xyz_rec     = prop['xyz_rec'] #receptor xyz -- read from per-lig instead
        aas         = prop['aas'] #1hot, 0~27 (last is ligand, follow utils.residues_and_metals

        sasa_rec,cbcounts_rec = 0,0
        if 'sasa_rec' in prop: sasa_rec = prop['sasa_rec']
        if 'cbcounts_rec' in prop: cbcounts_rec = prop['cbcounts_rec']
        
        # get per-lig features
        xyz_lig = samples['xyz'][pindex]
        xyz_rec = samples['xyz_rec'][pindex]
        sname   = samples['name'][pindex]
        lddt    = samples['lddt'][pindex] # natm
        fnat    = samples['fnat'][pindex] # 1
        #rmsd    = samples['rmsd'][pindex] # 1
        # get ligand properties IF defined in the lig.npz
        #print('atypes_lig' in samples, 'bnds_lig' in samples,
        #      'charge_lig' in samples, 'aas' in samples)

        if 'atypes_lig' in samples:  atypes_lig  = samples['atypes_lig'][pindex]
        if 'bnds_lig' in samples:    bnds_lig    = samples['bnds_lig'][pindex]
        if 'charge_lig' in samples:  charges_lig = samples['charge_lig'][pindex]
        if 'aas' in samples:         aas         = samples['aas'][pindex]
        
        #affinity = -1.0
        #if 'affinity' in samples:    affinity    = samples['affinity'][pindex]
        #affinity1hot = get_affinity_1hot(affinity, self.affinity_digits)

        # bond properties
        bnds_lig    = prop['bnds_lig']
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
        
        # Do KD-ball neighbour search.
        # We are also saving node features.
        # I think there is more efficient way to do this, but for now we are settling with this.
        # Grabbing a <dist neighbor from ligcom
        
        try:
            center_xyz = np.mean(xyz_lig, axis=0)[None,:] #2-D required...
            dist    = self.ball_radius
            kd      = scipy.spatial.cKDTree(xyz)
            kd_ca   = scipy.spatial.cKDTree(center_xyz)
            indices = kd_ca.query_ball_tree(kd, dist)

            # make sure ligand atms are ALL INCLUDED
            idx_ord = [i for i in indices[0] if i < len(xyz_lig)]
            # append rest atms
            idx_ord += [i for i in indices[0] if i >= len(xyz_lig)]
         
            # Get only the node features that we need.
            atype_f = atypes[idx_ord]
            aas_f   = aas[idx_ord]
            charges_f = charges[idx_ord]
            sasa_f = []
            if len(sasa) > 0: sasa_f  = sasa[idx_ord]
            xyz     = xyz[idx_ord]
            bnds    = np.concatenate([bnds_lig,bnds_rec])
            bnds    = [bnd for bnd in bnds if (bnd[0] in idx_ord) and (bnd[1] in idx_ord)]

            ligidx = np.zeros((len(idx_ord),len(xyz_lig)))
            for i in range(len(xyz_lig)): ligidx[i,i] = 1.0
            info['ligidx'] = torch.tensor(ligidx).float()
        
            # Concatenate coord & centralize xyz to ca.
            xyz = xyz - center_xyz
            xyz = torch.tensor(xyz).float()

            # randomize coordinate
            if self.randomize > 1e-3:
                randxyz = 2.0*self.randomize*(0.5 - torch.rand((len(xyz),3)))
                xyz = xyz + randxyz
        
            # Edge index and neighbour funtions.
            D_neighbors, E_idx = self.dist_fn(xyz[None,])
        
            # Construct the graph
            u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
            #print(atype_f.shape, aas_f.shape, charges_f.shape, xyz.shape, E_idx.shape)

            obt = [atype_f, aas_f, charges_f]
            if len(sasa_f) > 0: obt.append(sasa_f)
            obt = np.concatenate(obt,axis=-1)
            
            v = E_idx[0,].reshape(-1)
            G = dgl.graph((u,v))
        
            # Save x 
            G.ndata['x'] = xyz[:,None,:]
            G.ndata['0'] = torch.tensor(obt).float()
        
            # Add edge features to graph
            # u,v are nedges & pairs to each other; i.e. (u[i],v[i]) are edges for every i
            bnds_bin = np.zeros((len(xyz),len(xyz)))
            for i,j in bnds:
                k = idx_ord.index(i)
                l = idx_ord.index(j)
                bnds_bin[k,l] = bnds_bin[l,k] = 1

            w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None].repeat(1,2)
            w[:,1] = torch.tensor(bnds_bin[v,u]).float()

            G.edata['d'] = xyz[v] - xyz[u]
            G.edata['w'] = w
            
        except:
            stat = False

        info['stat'] = stat
        info['pname'] = pname
        info['sname'] = sname
        #info['affinity'] = affinity1hot

        if not stat:
            return False, 0.0, 0.0, info
        
        return G, lddt, fnat, info 
    
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
    hot1 = np.eye(len(digits)+1)[affinity]
    if soften > 0:
        hot1[:-1] += soften*np.copy(hot1[:-1])
        hot1[1:] += soften*np.copy(hot1[1:])
        hot1 /= 1.0 + 2.0*soften
    return hot1
    
def collate(samples):
    graphs, lddt, fnat, info = map(list, zip(*samples))
    try:
        batched_graph = dgl.batch(graphs)
    except:
        batched_graph = False
    
    return batched_graph, torch.tensor(lddt), torch.tensor(fnat), info

def correlation_Pearson( pred, ans ):
    pred  = pred - torch.mean(pred)
    ans   = ans - torch.mean(ans)

    norm = torch.sum(pred*ans)
    denorm1 = torch.sqrt(torch.sum(pred*pred)+1e-6)
    denorm2 = torch.sqrt(torch.sum(ans*ans)+1e-6)
    
    return norm/denorm1/denorm2, denorm2
