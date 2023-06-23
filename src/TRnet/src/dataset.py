import torch
import numpy as np
import dgl
import sys,copy,os
from scipy.spatial.transform import Rotation
import time
import random
import src.myutils as myutils

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets, K, datapath='data', neighmode='dist',
                 n=1, maxT=5.0, dcut_lig=5.0, pert=False, noiseP = 0.8, topk=8, maxnode=1500,
                 max_subset=5,
                 mixkey=False, dropH=False, version=1, motifnetnpz=None):
        
        self.targets = targets
        self.datapath = datapath
        self.neighmode = neighmode
        self.n = n
        self.K = K
        self.dcut_lig = dcut_lig
        self.maxT = maxT
        self.pert = pert
        self.topk = topk
        self.noiseP = noiseP
        self.mixkey = mixkey
        self.max_subset = max_subset
        #self.motifnetnpz = motifnetnpz

        self.maxnode = maxnode
        self.dropH = dropH
        self.version = version
            
    def __len__(self):
        return self.n*len(self.targets)
    
    def __getitem__(self, index): #N: maximum nodes in batch
        t0 = time.time()
        imol = index%len(self.targets)
        target = self.targets[imol][0]
        target_lig_list = self.targets[imol][1]

        if len(target_lig_list) > self.max_subset:
            target_lig_list = [target]+list(np.random.choice(target_lig_list[1:],self.max_subset-1,replace=False))

        info = {'name':target}

        if self.version == 2: # Biolip-Combo
            if '.' in target:
                motifnetnpz = self.datapath+'/biolip/'+target+'.score.npz'
            else:
                motifnetnpz = self.datapath+'/PDBbind/'+target+'.score.npz'
        else: #1/3
            motifnetnpz = self.datapath+'/'+target+'.score.npz'
                
        if self.version <= 2:
            keyatomf = self.datapath+'/keyatom.def.npz'
            
        elif self.version == 3:
            mol2 = target+'.ligands.mol2'
            keyatomf = target+'.keyatom.def.npz'
            
        if not os.path.exists(motifnetnpz):
            print(f"{motifnetnpz} does not exist")
            return info
        if not os.path.exists(keyatomf):# or (not os.path.exists(mol2):
            print(f"{keyatomf} does not exist")
            return info

        if True:
        #try:
            t1 = time.time()
            Grec = receptor_graph_from_motifnet(motifnetnpz, 
                                                mode=self.neighmode, top_k=self.topk,
                                                maxnode=self.maxnode)

            t2 = time.time()
            if self.version <= 2:
                blabel_list, Glig_list, keyidx_list, atms_list, Gnat, keyxyz = self.read_by_single_mol2(target, target_lig_list, keyatomf, self.K)
                tags = target_lig_list
            elif self.version == 3:
                blabel_list, Glig_list, keyidx_list, atms_list, Gnat, keyxyz, tags = read_by_batch_mol2(mol2, keyatomf, self.K, target_lig_list)
            t3 = time.time()

            if not Grec:
                print("skip %s for size cut %d"%(target, self.maxnode))
                return info
                
        #except:
        else:
            print("failed to read %s"%target)
            return info

        info['atms'] = atms_list
        info['lig']  = tags


        return Grec, Glig_list, keyxyz, keyidx_list, blabel_list, info #label

    def read_by_single_mol2(self, target, target_lig_list, keyatomf, K):
        Glig_list = [] 
        blabel_list = []
        keyidx_list = []
        atms_list = []
        Gnat = None
        keyxyz = None
        
        for lig in target_lig_list:
            if self.version == 1:
                mol2 = self.datapath+'/../mol2s/'+lig+'.ligand.mol2'
                confmol2 = self.datapath+'/../mol2s/'+lig+'.conformers.mol2'
            elif self.version == 2:
                mol2 = self.datapath+'/mol2s/'+lig+'.ligand.mol2'
                confmol2 = self.datapath+'/conformers/'+lig+'.conformers.mol2'
                    
            if not os.path.exists(confmol2):
                confmol2 = None
                    
            Gnat,Glig,atms = ligand_graph_from_mol2(mol2,
                                                    dcut=self.dcut_lig,
                                                    read_alt_conf=confmol2,
                                                    mode=self.neighmode,top_k=self.topk,
                                                    drop_H=self.dropH)
        
            if Glig == None or not atms:
                print("skip %s for ligand read failure %s"%(target,mol2),file=sys.stderr)
                continue
            
            keyidx = identify_keyidx(lig, atms, keyatomf, K)
            #keyatms = np.array(atms)[keyidx]
                
            if not isinstance(keyidx, np.ndarray):
                print("key name violation %s\n"%(lig),file=sys.stderr)
                continue
                
            keyidx_list.append(keyidx)
                
            if random.random() < self.noiseP:
                Glig, randoms = give_noise_to_lig(Glig)
                
            blabel_list.append(int(lig == target))
            Glig_list.append(Glig)
            atms_list.append(atms)
                
            if lig == target:
                keyxyz = Gnat.ndata['x'][keyidx] # K x 3
                    
        return blabel_list, Glig_list, keyidx_list, atms_list, Gnat, keyxyz

def read_by_batch_mol2(mol2, keyatomf, K, tags=None):
    Glig_list = [] 
    blabel_list = []
    keyidx_list = []
    atms_list = []
        
    t0 = time.time()
    (elems,_,bonds,borders,xyz,nneighs,atms,tags) = myutils.read_mol2_batch(mol2, tags)
    t1 = time.time()

    tags_processed = []
    for e,b,o,x,n,a,tag in zip(elems,bonds,borders,xyz,nneighs,atms,tags):
        try:
        #if True:
            keyidx = identify_keyidx(tag, a, keyatomf, K)
            if not isinstance(keyidx, np.ndarray):
                print("key name violation %s"%(tag))
                continue
                
            keyidx_list.append(keyidx)
            tags_processed.append(tag)
            
            Glig = generate_ligand_graph(e,_,b,o,x,n,a)
            Glig_list.append(Glig)
            atms_list.append(a)
            #print(Glig.number_of_nodes(), Glig, keyidx, tag)
                
        #else:
        except:
            continue
    t2 = time.time()
    print(t1-t0, t2-t1)
            
    # assume this is inferrence 
    Gnat, keyxyz = None, None
    blabel_list = [0 for _ in Glig_list]
    
    return blabel_list, Glig_list, keyidx_list, atms_list, Gnat, keyxyz, tags_processed

def receptor_graph_from_motifnet(npz,dcut=1.8,mode='dist',top_k=8,maxnode=1500,debug=False): # max 26 edges
    data = np.load(npz,allow_pickle=True)
    grids = data['grids']
    if grids.shape[0] > maxnode:
        return False
    prob = data['P']

    sel = []
    criteria = np.array([0 for k in range(13)]) #uniform
    # criteria = np.array([0.5,0.5,0.5,0.9,0.5,0.5,0.3,0.4,0.3,1.0,0.3,0.3,0.3]) #per-motif
    for i,p in enumerate(prob):
        diff = p[1:]-criteria
        if (diff>0.0).any(): sel.append(i)
            
    if debug:
        print("%s, selected %d points from %d"%(npz, len(sel),len(grids)))
    xyz = grids[sel]
    P_sel = prob[sel]

    #dummy placeholder for trsf learning
    Abias = np.zeros((len(P_sel),4))
    #nodef = np.concatenate([P_sel,Abias],axis=-1)
    
    nodef = np.concatenate([P_sel],axis=-1)

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

    G.ndata['attr'] = nodef
    G.ndata['x'] = torch.tensor(xyz)[:,None,:]
    
    G.edata['attr'] = edgef
    
    G.edata['rel_pos'] = dX[:,u,v].float()[0]
    return G
    
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
    
def ligand_graph_from_mol2(mol2,dcut,
                           read_alt_conf=None,mode='dist',
                           top_k=16,drop_H=True):
    t0 = time.time()
    xyz_alt = []
    try:
        if read_alt_conf != None:
            xyz_alt,atms = myutils.read_mol2s_xyzonly(read_alt_conf)
            
        args = myutils.read_mol2(mol2,drop_H=drop_H)
    except:
        args = myutils.read_mol2(mol2,drop_H=drop_H)
        print("failed to read mol2", mol2)
        return False, False, False
        
    if not args:
        return False, False, False

    elems, qs, bonds, borders, xyz_nat, nneighs, atms = args
    
    if len(xyz_alt) > 1:
        ialt = min(np.random.randint(len(xyz_alt)),len(xyz_alt)-1)
        xyz = torch.tensor(xyz_alt[ialt]).float()
    else:
        xyz = xyz_nat
        
    Glig = generate_ligand_graph(elems, qs, bonds, borders, xyz, nneighs, atms)
    
    Gnat = copy.deepcopy(Glig)
    Gnat.ndata['x'] = torch.tensor(xyz_nat).float()[:,None,:]
    
    return Glig, Gnat, atms

def generate_ligand_graph(elems, qs, bonds, borders, xyz, nneighs, atms, top_k=16, dcut=5.0, mode='dist'):
    X = torch.tensor(xyz[None,]) #expand dimension
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    u,v,d = find_dist_neighbors(dX,dcut=dcut,mode=mode,top_k=top_k)
    obt  = []
    elems = [myutils.ELEMS.index(a) for a in elems]

    Glig = dgl.graph((u,v))

    # normalize nneigh
    ns = np.sum(nneighs,axis=1)[:,None]+0.01
    ns = np.repeat(ns,4,axis=1)
    nneighs *= 1.0/ns
        
    elems1hot = np.eye(len(myutils.ELEMS))[elems] # 1hot encoded
    obt.append(elems1hot)
    obt.append(nneighs)
    
    # just dummy placeholder for trsf learning
    #key1hot = np.zeros((len(elems),4))
    #obt.append(key1hot)

    obt = np.concatenate(obt,axis=-1)

    ## edge features
    # 1-hot encode bond order
    ib = np.zeros((xyz.shape[0],xyz.shape[0]),dtype=int)
    for k,(i,j) in enumerate(bonds): ib[i,j] = k
    
    border_ = np.zeros(u.shape[0], dtype=np.int64)
    d_ = torch.zeros(u.shape[0])
    t1 = time.time()

    #print(mol2, bonds, len(borders), border_.shape)
    for k,(i,j) in enumerate(zip(u,v)):
        border_[k] = borders[ib[i,j]]
        d_[k] = d[i,j]
    t2 = time.time()

    edata = torch.eye(5)[border_] #0~3
    edata[:,-1] = normalize_distance(d_) #4

    Glig.ndata['attr'] = torch.tensor(obt).float()
    Glig.ndata['x'] = torch.tensor(xyz).float()[:,None,:]
    Glig.edata['attr'] = edata
    Glig.edata['rel_pos'] = dX[:,u,v].float()[0]
    
    Glig.ndata['Y'] = torch.zeros(obt.shape[0],3) # placeholder

    return Glig

def identify_keyidx(target, atms, keyatomf, K):
    keyatoms = np.load(keyatomf,allow_pickle=True)
    if 'keyatms' in keyatoms:
        keyatoms = keyatoms['keyatms'].item()

    if target not in keyatoms: return None
    keyidx = np.array([atms.index(a) for a in keyatoms[target] if a in atms],dtype=int)
        
    if len(keyidx) < 4: return None

    if K > 0:
        if self.mixkey:
            keyidx = np.random.choice(keyidx,self.K,replace=False)
        else:
            keyidx = keyidx[:self.K]
    else: # else use max 8; always mix if K < 0 (dynamic-K)
        if len(keyidx) > 8:
            keyidx = np.random.choice(keyidx,8,replace=False)
            
    return keyidx

def give_noise_to_lig(G_lig, random_in_lig = 0.1, noise_scale = 1):
    xyz = G_lig.ndata['x']
    xyz = xyz.squeeze()

    at_num = xyz.shape[0]
    rand_num = int(at_num*random_in_lig)

    mask = torch.zeros_like(xyz)
    randoms = random.choices([i for i in range(at_num)],k=rand_num)
    random_scale = (torch.rand_like(mask)*2-1)*noise_scale

    for i in randoms:
        mask[int(i)] +=1

    mask = random_scale*mask
    xyz = xyz + mask

    G_lig.ndata['x'] =xyz.float()[:,None,:]

    return G_lig, randoms

def collate(samples):
    if len(samples) > 1:
        sys.exit("batch should be set to 1")

    args = samples[0]
    if len(args) == 1: # info
        return None, None, None, None, None, args
        
    (G1,G2s,keyxyz,keyidx_list,blabel,info) = samples[0]

    if G1 == None: #or keyxyz == None:
        return None, None, None, None, None, info

    b = len(keyidx_list)
    G2s = dgl.batch(G2s)

    ## separate batch here...
    # not very memory efficient
    G1s = dgl.batch([G1 for _ in range(b)])
    keyidx = [torch.eye(n)[idx] for n,idx in zip(G2s.batch_num_nodes(),keyidx_list)]
    blabel = torch.tensor(blabel,dtype=float)

    if keyxyz != None:
        keyxyz = keyxyz.squeeze() # K x 3
    
    # info
    return G1s, G2s, keyxyz, keyidx, blabel, info

