import torch
import numpy as np
import dgl
import sys,copy
from scipy.spatial.transform import Rotation
import time

ELEMS = ['Null','H','C','N','O','Cl','F','I','Br','P','S'] #0 index goes to "empty node"

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets, K, datapath='data', neighmode='dist',
                 keyatmdef='data/keyatom.def.npz',
                 batchsize=10,
                 real_data_every_minibatch=2,
                 n=1, maxT=5.0, dcut_lig=5.0, pert=False ):
        
        self.datapath = datapath
        self.neighmode = neighmode
        self.keyatmdef = keyatmdef
        self.batchsize = batchsize
        self.real_data_every_minibatch = real_data_every_minibatch
        self.K = K
        self.dcut_lig = dcut_lig
        self.maxT = maxT
        self.pert = pert
        self.topk = 8
        
        self.Grecs = []
        self.Gligs = []

        self.targets_real = [a for a in targets if 'simulated' not in a]
        self.targets_simul = [a for a in targets if 'simulated' in a]
            
    def __len__(self):
        return 1000 #len(self.targets_real+self.targets_simul)
    
    def __getitem__(self, index): #N: maximum nodes in batch
        batchidx = int(index/self.batchsize)
        #print(index,batchidx,batchidx%self.real_data_every_minibatch)

        if batchidx%self.real_data_every_minibatch == 0:
            targets = self.targets_real
            datatype = 'real'
        else:
            targets = self.targets_simul
            datatype = 'simulated'

        # take random sample and not shuffle
        imol = np.random.randint(len(targets))
        target = targets[imol]

        motifnetnpz = self.datapath+'/'+target+'.score.npz'
        mol2 = self.datapath+'/'+target+'.ligand.mol2'

        Grec,Glig = False,False
        try:
            Grec = receptor_graph_from_motifnet(motifnetnpz, self.K, mode=self.neighmode, top_k=self.topk)
            Glig,atms = ligand_graph_from_mol2(mol2,self.K,dcut=self.dcut_lig,mode=self.neighmode,top_k=self.topk)
        except:
            print("failed to read %s"%target)
            return 

        # use shorter name from here on
        target = target.split('/')[-1]
        keyidx = identify_keyidx(target, Glig, atms, self.keyatmdef, self.K)
        if not keyidx:
            #print("%4d/%4d: key name violation %s"%(index,len(self.targets),target))
            return 

        keyidx = keyidx[:self.K]
        
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

        info = {'name':target, 'datatype':datatype }

        return Grec, Glig, labelxyz, keyidx, info #label

def receptor_graph_from_motifnet(npz,K,dcut=1.8,mode='dist',top_k=8,debug=False): # max 26 edges
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

    #dummy placeholder for attention bias feature
    Abias = np.zeros((len(P_sel),K))
    
    nodef = np.concatenate([P_sel,Abias],axis=-1)

    X = torch.tensor(xyz[None,]) #expand dimension
    nodef = torch.tensor(nodef)

    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-6)

    if mode == 'topk':
        top_k_var = min(xyz.shape[0],top_k+1) # consider tiny ones
        D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
        #exclude self-connection
        D_neighbor =  D_neighbors[:,:,1:]
        E_idx = E_idx[:,:,1:]
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)
        
    elif mode == 'distT':
        # trick to take upper triagonal
        mask = torch.where(torch.tril(D)<1.0e-6,100.0,1.0)
        maskedD = mask*D
        _,u,v = torch.where(maskedD<dcut)
        
    elif mode == 'dist':
        _,u,v = torch.where(D<dcut)

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
                #print('ERROR: %s, unknown elem type: %s'%(mol2,elem))
                #return False
                elem = 'Null'
            
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

def find_dist_neighbors(dX,dcut,mode='dist',top_k=8):
    D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-6)

    if mode == 'dist':
        # Trick to take upper triagonal
        _,u,v = torch.where(D<dcut)
    elif mode == 'distT':
        mask = torch.where(torch.tril(D)<1.0e-6,100.0,1.0)
        _,u,v = torch.where(mask*D<dcut)
    elif mode == 'topk':
        top_k_var = min(D.shape[1],top_k+1) # consider tiny ones
        D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
        D_neighbor =  D_neighbors[:,:,1:]
        E_idx = E_idx[:,:,1:]
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)
        
    return u,v,D[0]
    
def ligand_graph_from_mol2(mol2,K,dcut,mode='dist',top_k=8):
    t0 = time.time()
    try:
        args = read_mol2(mol2)
    except:
        return False, False
        
    if not args:
        return False, False
    elems, qs, bonds, borders, xyz, nneighs, atms = args

    X = torch.tensor(xyz[None,]) #expand dimension
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    u,v,d = find_dist_neighbors(dX,dcut=dcut,mode=mode,top_k=top_k)
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

    #dummy placeholder
    key1hot = np.zeros((len(elems),K))
    obt.append(key1hot)
    
    obt = np.concatenate(obt,axis=-1)

    ## edge features
    # 1-hot encode bond order
    ib = np.zeros((xyz.shape[0],xyz.shape[0]),dtype=int)
    for k,(i,j) in enumerate(bonds): ib[i,j] = k
    
    border_ = np.zeros(u.shape[0], dtype=np.int64)
    d_ = torch.zeros(u.shape[0])
    t1 = time.time()

    for k,(i,j) in enumerate(zip(u,v)):
        border_[k] = borders[ib[i,j]]
        d_[k] = d[i,j]
    t2 = time.time()

    edata = torch.eye(5)[border_] #0~3
    edata[:,-1] = normalize_distance(d_) #4

    G.ndata['attr'] = torch.tensor(obt).float()
    G.ndata['x'] = torch.tensor(xyz).float()[:,None,:]
    G.edata['attr'] = edata
    G.edata['rel_pos'] = dX[:,u,v].float()[0]
    
    #print(f"extracted {edata.shape} edge & {obt.shape} node features")
    
    G.ndata['Y'] = torch.zeros(obt.shape[0],3) # placeholder

    #G  = dgl.add_self_loop(G)
    #G = dgl.remove_self_loop(G)
    t3 = time.time()
    #print("data loading %3d %.1f %.1f %.1f"%(G.number_of_nodes(), t1-t0, t2-t1, t3-t2))

    return G,atms    

def identify_keyidx(target, Glig, atms, keyatmdef, K):
    ## find key idx as below -- TODO
    # 1,2 2 max separated atoms along 1-st principal axis
    # 3:

    ## temporary: hard-coded
    # perhaps revisit as 3~4 random fragment centers -- once fragmentization logic implemented
    
    keyatoms = np.load(keyatmdef,allow_pickle=True)
    if 'keyatms' in keyatoms:
        keyatoms = keyatoms['keyatms'].item()
    keyidx = [atms.index(a) for a in keyatoms[target] if a in atms]
        
    if len(keyidx) < K: return False
    return keyidx
