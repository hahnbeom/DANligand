import sys
import numpy as np
import torch
import dgl
from .utils import *
from .peratom_lddt import *
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
                 dist_fn_ha,
                 dist_fn_ca,
                 root_dir        = "/projects/casp/pdbs/",
                 verbose         = False,
                 useTipNode      = False,
                 ball_radius     = 10,
                 displacement    = "",
                 encodeBonds     = True,
                 encodeBackboneConnectivity = True):
        
        self.dist_fn_ha = dist_fn_ha
        self.dist_fn_ca = dist_fn_ca
        self.datadir = root_dir
        self.verbose = verbose
        self.proteins = targets
        self.ball_radius = ball_radius
        self.encodeBonds = encodeBonds
        self.encodeBackboneConnectivity = encodeBackboneConnectivity
        
        # These are pulled from deepAccNet2.utils
        self.residuemap = dict([(residues[i], i) for i in range(len(residues))])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.proteins)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select a sample decoy 
        pname = self.proteins[index]
        samples = [i for i in listdir(join(self.datadir, pname)) if i.endswith(".pdb")]
        pindex = np.random.choice(np.arange(len(samples)))
        sname = samples[pindex]
        
        pdbfilename = join(self.datadir, pname, sname)
        pose = parse_pdbfile(pdbfilename)
        
        # Grab the corresponding native structure
        native_filename = join(self.datadir, pname, "native.pdb")
        native_pose = parse_pdbfile(native_filename)
        
        # Choose with residue index to look at.
        residue_index = np.random.choice(np.arange(len(pose)))
        
        # Get CAlddt:
        decoy_dist = np.array([pose[i]["CA"]["coord"] for i in range(len(pose))])
        native_dist = np.array([native_pose[i]["CA"]["coord"] for i in range(len(native_pose))])
        decoy_coords = distance_matrix(decoy_dist, decoy_dist)
        native_coords = distance_matrix(native_dist, native_dist)
        res_ca_lddt = get_lddt(decoy_coords, native_coords)
        
        # Calculate full_atom lddt based on Frank's code.
        fullatom_lddt = get_fullatom_lddt(native_filename, pdbfilename)
        
        # Positinal embedding
        pos_emb_temp = positional_embedding(len(pose), dup=False).T
        
        #######################
        # HA graph generation #
        #######################
        
        # Do KD-ball neighbour search.
        # We are also saving node features.
        # I think there is more efficient way to do this, but for now we are settling with this.
        pos_emb_source   = []
        atype_emb_source = []
        aas_emb_source   = []
        xyz_source       = []
        atom_lddt        = []
        res_mask         = []
        
        # Very bad implementation to get bonds. I need to revisit later.
        AAforBond = []
        AnameforBond = []
        IndsForBond = []
        
        for i in range(len(pose)):
            aa = pose[i]["rname"]
            aa_1hot = np.zeros((20))
            aa_1hot[residuemap[aa]] = 1
            for k in pose[i].keys():
                if not k in ["rname", "rindex", "OXT"]:
                    pos_emb_source.append(pos_emb_temp[i])
                    aas_emb_source.append(aa_1hot)
                    atype_1hot = np.zeros((21))
                    atype_1hot[atype2num.get(atypes.get((aa, k), "dmy"), -1)] = 1
                    atype_emb_source.append(atype_1hot)
                    xyz_source.append(pose[i][k]["coord"])
                    atom_lddt.append(fullatom_lddt[i, k])
                    res_mask.append(1 if i==residue_index else 0)
                    
                    # For bonds
                    AAforBond.append(aa)
                    AnameforBond.append(k)
                    IndsForBond.append(i)
                    
        pos_emb_source   = np.array(pos_emb_source)
        atype_emb_source = np.array(atype_emb_source)
        aas_emb_source   = np.array(aas_emb_source)
        xyz_source       = np.array(xyz_source)
        atom_lddt        = np.array(atom_lddt)
        res_mask         = np.array(res_mask)
        
        # For bonds
        AAforBond        = np.array(AAforBond)
        AnameforBond        = np.array(AnameforBond)
        IndsForBond        = np.array(IndsForBond)
        
        # Get the target CA position
        center_ca_xyz = np.array(pose[residue_index]["CA"]["coord"])[None, :]
        
        # Grabbing a < dist neighubour from CA.
        dist    = self.ball_radius
        kd      = scipy.spatial.cKDTree(xyz_source)
        kd_ca   = scipy.spatial.cKDTree(center_ca_xyz)
        indices = kd_ca.query_ball_tree(kd, dist)
        
        # Get only the node features that we need.
        pos_f     = pos_emb_source[indices[0]]
        atype_f   = atype_emb_source[indices[0]]
        aas_f     = aas_emb_source[indices[0]]
        xyz       = xyz_source[indices[0]]
        atom_lddt = atom_lddt[indices[0]]
        res_mask  = res_mask[indices[0]]
        AAforBond     = AAforBond[indices[0]]
        AnameforBond  = AnameforBond[indices[0]]
        IndsForBond   = IndsForBond[indices[0]]
        
        # Centralize xyz to ca.
        xyz = xyz - center_ca_xyz
        xyz = torch.tensor(xyz).float()
        
        # Edge index and neighbour funtions.
        D_neighbors, E_idx = self.dist_fn_ha(xyz[None,])
        # Construct the graph
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)
        G_ha = dgl.DGLGraph((u, v))
        
        # Save x 
        G_ha.ndata['x'] = xyz
        G_ha.ndata['0'] = torch.tensor(np.concatenate([pos_f, atype_f, aas_f, res_mask[:, None]], axis=-1)).float()
        G_ha.ndata['mask'] = torch.tensor(res_mask).float()
        G_ha.ndata['lddt'] = torch.tensor(atom_lddt).float()
        G_ha.edata['d'] = xyz[v] - xyz[u]
        
        # Add edge features to graph
        # See if two atoms are bonded
        isBonded = []
        for x in zip(AAforBond[u], AnameforBond[u], IndsForBond[u], AAforBond[v], AnameforBond[v], IndsForBond[v]):
            aa1, at1, i1, aa2, at2, i2 = x
            # atoms needs to be from the same residue to be connected
            if i1 == i2:
                if (at1, at2) in bonds[aa1]: isBonded.append(1)
                else: isBonded.append(0)
            else:
                # Adjacent residues are connected between C and A
                if i2-i1 == 1:
                    if at1=="C" and at2=="N": isBonded.append(1)
                    else: isBonded.append(0)
                elif i1-i2 == 1:
                    if at1=="N" and at2=="C": isBonded.append(1)
                    else: isBonded.append(0)
                else:
                    isBonded.append(0)
        isBonded = torch.tensor(np.array(isBonded)[:, None]).float()
        original_edge_feature = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None].repeat(1,4)
        
        if self.encodeBonds:
            G_ha.edata['w'] = torch.cat([isBonded, original_edge_feature], axis=1)
        else:
            G_ha.edata['w'] = original_edge_feature
            
        #######################
        # Ca graph generation #
        #######################
        ca_xyz = torch.tensor(np.array([pose[i]["CA"]["coord"] for i in range(len(pose))])).float()
        
        D_neighbors, E_idx = self.dist_fn_ca(ca_xyz[None,])
        # Construct the graph
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)
        G_ca = dgl.DGLGraph((u, v))
        
        # AA feature
        aa_1hot = np.zeros((20, len(pose)))
        for i in range(len(pose)):
            aa = pose[i]["rname"]
            aa_1hot[residuemap[aa], i] = 1
        
        G_ca.ndata['x'] = ca_xyz
        G_ca.ndata['0'] = torch.tensor(np.concatenate([pos_emb_temp, aa_1hot.T], axis=-1)).float()
        G_ca.ndata['ca_lddt'] = torch.tensor(res_ca_lddt).float()
        G_ca.edata['d'] = ca_xyz[v] - ca_xyz[u]
        
        isConnected = []
        for i, j in zip(u, v):
            if np.abs(i-j) == 1:isConnected.append(1)
            else:isConnected.append(0)
        isConnected = torch.tensor(np.array(isConnected)[:, None]).float()
        original_edge_feature = torch.sqrt(torch.sum((ca_xyz[v] - ca_xyz[u])**2, axis=-1)+1e-6)[...,None].repeat(1,4)
        
        if self.encodeBackboneConnectivity:
            G_ca.edata['w'] = torch.cat([isConnected, original_edge_feature], axis=1)
        else:
            G_ca.edata['w'] = original_edge_feature
        
        return G_ha, G_ca
    
def get_fullatom_lddt(native_pdb, decoy_pdb):
    nxyz,nseq = readpdb(native_pdb)
    mask = aamask[nseq,:]
    xyz,seq = readpdb(decoy_pdb)
    assert(torch.equal(nseq,seq))

    lddt_i = lddt(xyz,nxyz,mask)
    xyzalt = alternate_coords(xyz, seq)
    lddt_i = torch.max(lddt_i, lddt(xyzalt, nxyz,mask))

    lddt_dict = {}
    for i,s in enumerate(seq):
        for j,a in enumerate(aa2longalt[s]):
            if a is not None:
                lddt_dict[i, a.strip()] = float(lddt_i[i,j])
                
    return lddt_dict
    
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

def collate(samples):
    g_ha, g_ca = map(list, zip(*samples))
    batched_g_ha = dgl.batch(g_ha)
    batched_g_ca = dgl.batch(g_ca)
    return batched_g_ha, batched_g_ca