import torch
import numpy as np
import dgl
import sys,copy
from scipy.spatial.transform import Rotation

ELEMS = ['Null','H','C','N','O','Cl','F','I','Br','P','S'] #0 index goes to "empty node"

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets, K, datapath='data',
                 n=1, maxT=5.0, dcut_lig=5.0, pert=False ):
        
        self.targets = targets
        self.datapath = datapath
        self.n = n
        self.K = K
        self.dcut_lig = dcut_lig
        self.maxT = maxT
        self.pert = pert
        
        self.Grecs = []
        self.Gligs = []
        for target in targets:
            motifnetnpz = self.datapath+'/'+target+'.score.npz'
            Grec = receptor_graph_from_motifnet(motifnetnpz)
            self.Grecs.append(Grec)

            mol2 = self.datapath+'/'+target+'.ligand.mol2'
            Glig,atms = ligand_graph_from_mol2(mol2,self.K,dcut=self.dcut_lig)
            keyidx = identify_keyidx(target, Glig, atms)[:self.K]
            self.Gligs.append((Glig,keyidx))
            
    def __len__(self):
        return self.n*len(self.targets)
    
    def __getitem__(self, index): #N: maximum nodes in batch
        imol = index%len(self.targets)
        target = self.targets[imol]
        
        #motifnetnpz = self.datapath+'/'+target+'.score.npz'
        #Grec = receptor_graph_from_motifnet(motifnetnpz)
        Grec = self.Grecs[imol]

        # why preprocessing Glig much important for speed?
        
        #Glig,atms = ligand_graph_from_mol2(mol2,self.K,dcut=self.dcut_lig)
        #keyidx = identify_keyidx(target, Glig, atms)[:self.K]
        Glig,keyidx = self.Gligs[imol]

        if target in ['bace1r']:
            natmol2 = '%s/bace1.ligand.mol2'%self.datapath
            Gnat,_ = ligand_graph_from_mol2(natmol2,self.K,dcut=self.dcut_lig)
        else:
            Gnat = Glig
            
        # hard-coded
        keyidx_nat = keyidx
        xyzlig_nat = Gnat.ndata['x']
            
        #label = np.random.permutation(keyidx)[:self.K]
        label = keyidx_nat[:self.K]
        labelxyz = xyzlig_nat[label] # K x 3

        if self.pert:
            com = torch.mean(Glig.ndata['x'],axis=0)
            q = torch.rand(4) # random rotation
            R = torch.tensor(Rotation.from_quat(q).as_matrix()).float()
            t = self.maxT*(2.0*torch.rand(3)-1.0)

            xyz = torch.matmul(Glig.ndata['x']-com,R) + com + t
            Glig.ndata['x'] = xyz

        info = {'name':target}

        return Grec, Glig, labelxyz, keyidx, info #label

def receptor_graph_from_motifnet(npz,dcut=1.8,debug=False): # max 26 edges
    data = np.load(npz,allow_pickle=True)
    grids = data['grids']
    prob = data['P']

    sel = []
    #criteria = [0.25 for k in range(14)] #uniform
    criteria = np.array([0.5,0.5,0.5,0.9,0.5,0.5,0.3,0.4,0.3,1.0,0.3,0.3,0.3]) #per-motif
    for i,p in enumerate(prob):
        #imax = np.argmax(p[1:])
        #if p[imax] > criteria[imax]: sel.append(i)
        diff = p[1:]-criteria
        if (diff>0.0).any(): sel.append(i)
            
    if debug:
        print("selected %d points from %d"%(len(sel),len(grids)))
    xyz = grids[sel]
    P_sel = prob[sel]

    X = torch.tensor(xyz[None,]) #expand dimension
    nodef = torch.tensor(P_sel)

    # distace matrix
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-6)

    # trick to take upper triagonal
    mask = torch.where(torch.tril(D)<1.0e-6,100.0,1.0)

    # disallow self connection
    #maskedD = mask*D + 100*torch.eye(len(nodef))[None,:,:]

    #allow self connection
    maskedD = mask*D
    _,u,v = torch.where(maskedD<dcut)

    #print(max(u), max(v), maskedD.shape)

    # construct graph
    G = dgl.graph((u,v))

    # extract "normalized" distance
    normD = normalize_distance(D,maxd=dcut)
    edgef = torch.tensor([normD[0,i,j] for i,j in zip(u,v)])[:,None]
    #edgef = torch.tensor([normD[0,u,v]])[:,None]
    #print(f"extracted {edgef.shape} edge & {nodef.shape} node features")

    G.ndata['attr'] = nodef
    G.ndata['x'] = torch.tensor(xyz)[:,None,:]
    
    G.edata['attr'] = edgef
    
    G.edata['rel_pos'] = dX[:,u,v].float()[0]
    return G
    
def read_mol2(mol2):
    read_cont = 0
    qs = []
    elems = []
    xyzs = []
    bonds = []
    borders = []
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
            if words[1].startswith('Br') or  words[1].startswith('Cl'):
                elem = words[1][:2]
            else:
                elem = words[1][0]
            
            if elem not in ELEMS:
                sys.exit('unknown elem type: %s'%elem)
            atms.append(words[1])
            elems.append(elem)
            qs.append(float(words[-1]))
            xyzs.append([float(words[2]),float(words[3]),float(words[4])]) 
                
        elif read_cont == 2:
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'1':1,'2':2,'3':3,'ar':3,'am':2}
            borders.append(bondtypes[words[3]])

    nneighs = [[0,0,0,0] for _ in qs]
    for i,j in bonds:
        if elems[i] in ['H','C','N','O']:
            k = ['H','C','N','O'].index(elems[i])
            nneighs[j][k] += 1.0
        if elems[j] in ['H','C','N','O']:
            l = ['H','C','N','O'].index(elems[j])
            nneighs[i][l] += 1.0

    # drop hydrogens
    #nonHid = [i for i,a in enumerate(elems) if a != 'H']
    nonHid = [i for i,a in enumerate(elems)]

    bonds = [[i,j] for i,j in bonds if i in nonHid and j in nonHid]
    borders = [b for b,ij in zip(borders,bonds) if ij[0] in nonHid and ij[1] in nonHid]
    
    return np.array(elems)[nonHid], np.array(qs)[nonHid], bonds, borders, np.array(xyzs)[nonHid], np.array(nneighs)[nonHid], atms #np.array(atms)[nonHid]

def normalize_distance(D,maxd=5.0):
    d0 = 0.5*maxd #center
    m = 5.0/d0 #slope
    feat = 1.0/(1.0+torch.exp(-m*(D-d0)))
    return feat

def find_dist_neighbors(dX,dcut):
    D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-6)
    
    # Trick to take upper triagonal
    mask = torch.where(torch.tril(D)<1.0e-6,100.0,1.0)
    #maskedD = mask*D + 100*torch.eye(len(X))[None,:,:]
    #_,u,v = torch.where(maskedD<dcut)
    _,u,v = torch.where(mask*D<dcut)

    return u,v,D[0]
    
def ligand_graph_from_mol2(mol2,K,dcut):
    elems, qs, bonds, borders, xyz, nneighs, atms = read_mol2(mol2)

    X = torch.tensor(xyz[None,]) #expand dimension
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    u,v,d = find_dist_neighbors(dX,dcut=dcut)
    obt  = []
    elems = [ELEMS.index(a) for a in elems]

    G = dgl.graph((u,v))

    # normalize nneigh
    ns = np.sum(nneighs,axis=1)[:,None]+0.01
    ns = np.repeat(ns,4,axis=1)
    nneighs *= 1.0/ns
        
    elems1hot = np.eye(len(ELEMS))[elems] # 1hot encoded
    obt.append(elems1hot)
    obt.append(nneighs)
    obt = np.concatenate(obt,axis=-1)

    ## edge features
    # 1-hot encode bond order
    border_ = np.zeros(len(u), dtype=np.int64)
    d_ = torch.zeros(len(u))
    for k,(i,j) in enumerate(zip(u,v)):
        if [i,j] in bonds:
            ib = bonds.index([i,j])
            border_[ib] = borders[ib]
        d_[k] = d[i,j]

    edata = torch.eye(5)[border_] #0~3
    edata[:,-1] = normalize_distance(d_) #4
     
    G.ndata['attr'] = torch.tensor(obt).float()
    G.ndata['x'] = torch.tensor(xyz).float()[:,None,:]
    G.edata['attr'] = edata
    G.edata['rel_pos'] = dX[:,u,v].float()[0]

    G  = dgl.add_self_loop(G)

    return G,atms    

def identify_keyidx(target, Glig, atms):
    ## find key idx as below -- TODO
    # 1,2 2 max separated atoms along 1-st principal axis
    # 3:

    ## temporary: hard-coded
    # perhaps revisit as 3~4 random fragment centers -- once fragmentization logic implemented

    keyatoms = np.load('data/keyatom.def.npz',allow_pickle=True)['keyatms'].item()

    keyidx = [atms.index(a) for a in keyatoms[target]] 

    return keyidx
