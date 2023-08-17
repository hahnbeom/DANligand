import torch
import numpy as np
import dgl
import sys
import scipy
from scipy.spatial.transform import Rotation
import time

ELEMS = ['Null','H','C','N','O','Cl','F','I','Br','P','S'] #0 index goes to "empty node"

def collate(samples):
    #samples should be G,info
    valid = [v for v in samples if v != None]
    valid = [v for v in valid if (v[0] != False and v[0] != None)]
    if len(valid) == 0:
        return None, None, None

    G = []
    label = []
    info = []
    for s in valid:
        G.append(s[0])
        label.append(torch.tensor(s[1]))
        info.append(s[2])
        
    label = torch.stack(label,dim=0)
    #if len(label.shape) == 1: label = label.unsqueeze(2)

    bG = dgl.batch(G)

    return bG, label, info

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets,
                 orgpath='/ml/CSD',
                 datapath='data',
                 datatype = 'CSD',
                 neighmode='dist',
                 maxT=5.0,
                 dcut=5.0,
                 topk=8,
                 ntypes=128,
                 max_nodes=300,
                 pert=False,
                 debug=False):
        
        self.targets = targets
        self.datatype = datatype
        self.datapath = datapath
        self.orgpath = orgpath
        self.neighmode = neighmode
        self.dcut = dcut
        self.maxT = maxT
        self.pert = pert
        self.topk = topk
        self.ntypes = ntypes
        self.debug = debug
        self.max_nodes = max_nodes
        self.labelidx, self.connect_to = read_label(datapath+"/label.txt")

        self.targets = [target for target in targets if self.labelidx[target] < ntypes]
        print("reduced target list %d -> %d with labelidx < %d"%(len(targets),len(self.targets),ntypes))
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index): #N: maximum nodes in batch
        target = self.targets[index] #should be formatted as AAAAAA_0001_0

        if self.datatype == 'CSD':
            CSDID = target.split('/')[-1].split('_')[0]
            symmpdb  = self.orgpath+'/%s/%s.symm.pdb'%(CSDID[0],CSDID)
            inputpdb = self.datapath+'/%s/%s.pdb'%(CSDID[0],target)
            mol2  = self.orgpath+'/%s/%s.mol2'%(CSDID[0],CSDID)
            
        elif self.datatype == 'ZINC':
            symmpdb, inputpdb = None, None
            ZINCID = target.split('/')[-1].split('_')[0] # temporary
            mol2  = self.orgpath+'/%s/%s.mol2'%(target[0],ZINCID)

        labelidx = self.labelidx[target]
        if labelidx >= self.ntypes:
            print("pass %s due to label idx (%d>%d)"%(target,labelidx,self.ntypes))
            return 
        
        G = False
        try:
        #if True:
            elems,bnds,nHs = read_ligand_property(mol2, self.connect_to[target])
            G = build_graph(elems, bnds, nHs, self.connect_to[target],
                            contextpdb=symmpdb, selfpdb=inputpdb, 
                            dcut=self.dcut, mode=self.neighmode, topk=self.topk)

        #else:
        except:
            if self.debug:
                print("failed to read %s"%target)
            return 
        

        if G.number_of_nodes() > self.max_nodes:
            return
        
        if self.pert:
            com = torch.mean(G.ndata['x'],axis=0)
            q = torch.rand(4) # random rotation
            R = torch.tensor(Rotation.from_quat(q).as_matrix()).float()
            t = self.maxT*(2.0*torch.rand(3)-1.0)
            xyz = torch.matmul(G.ndata['x']-com,R) + com + t
            G.ndata['x'] = xyz

        info = {'name':target}

        return G, labelidx, info

def read_label(labeltxt):
    label = {}
    connected_to = {}
    for l in open(labeltxt):
        words = l[:-1].split()
        target,_,labelidx,con_atm = words
        try:
            label[target] = int(labelidx)
        except:
            print("invalid line:", l[:-1])
            continue
        connected_to[target] = con_atm
        
    return label, connected_to

def read_pdb(pdb, ignore_chain=None):
    xyz = []
    atms = []
    chainidx = []
    for l in open(pdb):
        if not l.startswith('HETATM'): continue
        chain = l[21]
        if chain == ignore_chain: continue
        xyz.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        atms.append(l[12:16].strip())
        chainidx.append(chain)

    return np.array(xyz), np.array(atms), np.array(chainidx)

def build_graph(elems, bnds, nHs, connected_to, contextpdb=None, selfpdb=None,
                mode="dist", dcut=8.0, topk=-1):

    if contextpdb != None and selfpdb != None:
        xyz_symm,atms_symm,chainidx = read_pdb( contextpdb, ignore_chain='A' )
        xyz_self,atms_self,_ = read_pdb( selfpdb )

        # select nodes nearby 
        kd_self = scipy.spatial.cKDTree(xyz_self)
        kd_symm = scipy.spatial.cKDTree(xyz_symm)
        ineigh = np.concatenate(kd_self.query_ball_tree(kd_symm,dcut)).astype(int)

        dcut_ext = dcut + 3.0 # bigger radius around connected atom
        xyz_con = xyz_self[np.where(atms_self==connected_to)]
        kd_con = scipy.spatial.cKDTree(xyz_con)
        ineigh_ext = np.concatenate(kd_con.query_ball_tree(kd_symm,dcut_ext)).astype(int)
        ineigh = np.unique(np.concatenate([ineigh, ineigh_ext]))

        # re index self so that connected atom goes to the last
        iself = [i for i,a in enumerate(atms_self) if a != connected_to]+list(np.where(atms_self==connected_to)[0])

        xyz = np.concatenate([xyz_symm[ineigh],xyz_self[iself]],axis=0)
        atms_n = np.concatenate([atms_symm[ineigh], atms_self[iself]],axis=0)
        chainidx = np.concatenate([chainidx[ineigh],['A' for _ in atms_self]],axis=0)
        
    else:
        xyz = None
        atms_n = list(elems.keys())
        chainidx = np.zeros(len(elems),dtype=int)
        
    # stores bond order
    bnd_matrix = np.zeros((len(elems),len(elems)),dtype=int)
    for i,a1 in enumerate(atms_n[:-1]):
        for j,a2 in enumerate(atms_n[i+1:]):
            if chainidx[i] != chainidx[j+i+1]: continue
            if a1 in bnds and a2 in bnds[a1]:
                bnd_matrix[i,j+i+1] = bnds[a1][a2]
    bnds_n = np.where(bnd_matrix)
    bnds_n = [[i,j] for i,j in zip(bnds_n[0],bnds_n[1])]

    # node info
    elems_n = [elems[atm] for atm in atms_n]
    elems_n = np.eye(len(ELEMS))[elems_n]
    hasbonds_n = np.max(bnd_matrix,axis=0) # single < double < triple < aro
    hasbonds_n = np.eye(5)[hasbonds_n]
    predsite_n = np.zeros((len(atms_n),1))
    predsite_n[-1] = 1.0
    nHs_n = [nHs[atm] for atm in atms_n]
    nHs_n = np.eye(4)[nHs_n]

    obt = [elems_n, hasbonds_n, predsite_n, nHs_n]
    nodef = np.concatenate(obt,axis=-1)
    nodef = torch.tensor(nodef)

    # edge info
    if mode == 'conn': #based on connectivity only
        tors_matrix = (bnd_matrix > 0).astype(int)
        # enumerate for torsions... should be not the time determining step, I guess
        for (i,j) in bnds_n:
            for (k,l) in bnds_n:
                if bnd_matrix[i,k]+bnd_matrix[j,k] > 0: tors_matrix[j,l] = tors_matrix[i,l] = 1 
                if bnd_matrix[i,l]+bnd_matrix[j,l] > 0: tors_matrix[j,k] = tors_matrix[i,k] = 1 
                if   (i==k): tors_matrix[j,l] = tors_matrix[l,j] = 2
                elif (j==l): tors_matrix[i,k] = tors_matrix[k,i] = 2
                elif (i==l): tors_matrix[j,k] = tors_matrix[k,j] = 2
                elif (j==k): tors_matrix[i,l] = tors_matrix[l,i] = 2

            tors_matrix[i,j] = tors_matrix[j,i] = 3
            tors_matrix[i,i] = 4
            
        u,v = np.where(tors_matrix > 0) # connect all edges within 1-4

    else:
        X = torch.tensor(xyz[None,]) #expand dimension
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-6)
        
        if mode == 'topk':
            top_k_var = min(xyz.shape[0],topk+1) # consider tiny ones
            D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
            D_neighbor =  D_neighbors[:,:,1:]
            E_idx = E_idx[:,:,1:]
            u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
            v = E_idx[0,].reshape(-1)
        
        elif mode == 'dist':
            _,u,v = torch.where(D<dcut)

    edge_order = np.eye(5)[bnd_matrix[u,v]] #0: none 1: single 2:double 3:triple 4:aromatic
    edge_sep   = np.eye(5)[tors_matrix[u,v]] # 0:far 1:tors 2:angle 3:bond 4:self
    edgef = torch.tensor(np.concatenate([edge_order,edge_sep],axis=1))

    # construct graph
    G = dgl.graph((u,v))
    G.ndata['attr'] = nodef
    G.edata['attr'] = edgef
    if xyz != None:
        xyz = torch.tensor(xyz)[:,None,:]
        G.ndata['x'] = xyz
        G.edata['rel_pos'] = dX[:,u,v].float()[0]
    
    return G

def read_ligand_property(mol2, connected_to):
    read_cont = 0
    elems = []
    bonds = {}
    atms = []
    
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        if l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            break

        words = l[:-1].split()
        if read_cont == 1:

            idx = words[0]
            if words[1].startswith('BR'): words[1] = 'Br'
            if words[1].startswith('Br') or  words[1].startswith('Cl') :
                elem = words[1][:2]
            else:
                elem = words[1][0]

            if elem == 'A' or elem == 'B' :
                elem = words[5].split('.')[0]
            
            if elem not in ELEMS:
                elem = 'Null'
            
            atms.append(words[1])
            elems.append(elem)
            
        elif read_cont == 2:
            bondtypes = {'1':1,'2':2,'3':3,'ar':4,'am':2, 'du':0, 'un':0}
            a1 = atms[int(words[1])-1]
            a2 = atms[int(words[2])-1]
            if a1 not in bonds: bonds[a1] = {}
            bonds[a1][a2] = bondtypes[words[3]]
            if a2 not in bonds: bonds[a2] = {}
            bonds[a2][a1] = bondtypes[words[3]]

    nHs = {atm:0 for atm in atms}
    for a1 in bonds:
        for a2 in bonds[a1]:
            i = atms.index(a1)
            j = atms.index(a2)
            if elems[i] == 'H': nHs[a2] += 1
            if elems[j] == 'H': nHs[a1] += 1

    nHs = {a:int(nHs[a]/2) for a in nHs}
    elems = {atm:ELEMS.index(elem) for atm,elem in zip(atms,elems)}

    return elems, bonds, nHs
