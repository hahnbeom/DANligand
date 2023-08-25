## modified from TRnet/src/dataset_eff.py

import torch
import numpy as np
import dgl
import sys,copy,os
import scipy
from scipy.spatial.transform import Rotation
import time
import random
import src.types as types
import src.myutils as myutils
import src.kappaidx as kappaidx

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets,
                 is_ligand,
                 ligands=[],
                 datapath='data',
                 ball_radius=8.0,
                 edgemode='dist', edgek=(8,16), edgedist=(2.2,4.5),
                 pert=False, randomize=0.5,
                 ntype = 6,
                 max_subset = 5,
                 maxedge = 100000,
                 maxnode = 3000,
                 debug=False):
        
        self.targets = targets
        self.is_ligand = is_ligand
        
        if ligands == []:
            self.ligands = [[a] for a in targets]
        else:
            self.ligands = ligands
            
        assert(len(is_ligand) == len(targets) == len(self.ligands))

        self.ball_radius = ball_radius
        self.datapath = datapath
        self.edgemode = edgemode
        self.edgedist = edgedist
        self.edgek = edgek
        self.ntype = ntype
        self.pert = pert
        self.randomize = randomize
        self.maxedge = maxedge
        self.maxnode = maxnode
        self.max_subset = max_subset
            
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index): #N: maximum nodes in batch
        t0 = time.time()

        # target: GridNet.biolip/3pnf.8BX
        # pname: 3pnf.8BX
        
        target = self.targets[index]
        pname = target.split('/')[-1]
        
        gridinfo = self.datapath + target + '.grid.npz'
        parentpath = '/'.join(gridinfo.split('/')[:-1])+'/' # includes all absolute path
        propnpz = parentpath+pname+".prop.npz"
        
        is_ligand = self.is_ligand[index]
        ligands = self.ligands[index]
        
        if ligands[0] == 'batch':
            # randomly select from mol2
            mol2type = 'batch'
            mol2f_a = parentpath+'/%s.active.mol2'%pname
            active = myutils.read_mol2_batch(mol2f_a,tags_read=ligands[1:],tag_only=True)[-1]
            active = [np.random.choice(active)] #one
            
            mol2f = parentpath+'/%s.ligand.mol2'%pname #active+decoy concat
            ligands = myutils.read_mol2_batch(mol2f,tag_only=True)[-1] # override ligand
            keyatomf = parentpath+'/%s.keyatom.def.npz'%pname #GridNet.ChemBl/{target}.keyatom.def.npz
                
        else:
            # place target-ligand at the first
            # likely to place target twice but low chance
            mol2type = 'single'
            mol2f = self.datapath #actual mol2s are datapath+/+ligand+'.mol2'
            active = [target]
            keyatomf = parentpath+'/keyatom.def.npz' #GridNet.ligand/keyatom.def.npz
            
        if len(ligands) > self.max_subset:
            ligands = active+list(np.random.choice(ligands,self.max_subset-1,replace=False))
                
        info= {'pname': pname}
        Grec, Glig_list, cats, mask, keyxyz, keyidx_list, blabel_list = None, None, None, None, None, None, None
        NullArgs = (None, None, None, None, None, None, info)

        # 0. read grid info
        if not os.path.exists(gridinfo) or not os.path.exists(propnpz):
            print(f"no such file exists {gridinfo} or {propnpz}")
            return NullArgs
        
        # 1. condition on inferrence <-> train; ligand-info <-> protein only
        sample = np.load(gridinfo, allow_pickle=True)
        grids  = sample['xyz'] #vector; set all possible motif positions for prediction
        
        #if self.labeled:
        cats, mask = None, None
        if 'labels' in sample:
            cats = sample['labels'] # ngrid x nmotif
            if cats.shape[1] > self.ntype: cats = cats[:,:self.ntype]
            mask = np.sum(cats>0,axis=1) #0 or 1
            
            cats = torch.tensor(cats).float()
            mask = torch.tensor(mask).float()
       
        try:
        #if True:
            t1 = time.time()
            if is_ligand:
                gridchain = None
                
                #if len(ligands) > 0:
                #blabel_list, Glig_list, keyidx_list, atms_list, Gnat, keyxyz, origin = self.read_by_single_mol2(target, ligands, keyatomf, mol2path=self.datapath)
                
                blabel_list, Glig_list, keyidx_list, atms_list, Gnat, keyxyz, origin = \
                    self.read_ligands(target, ligands, keyatomf, mol2path=mol2f, actives=active, mol2type=mol2type )

                info['nK'] = [len(idx) for idx in keyidx_list]
                    
            else:
                gridchain = pname.split('.')[1]
                gridname = pname.split('.')[2]
                info['pname'] = pname+'.'+gridchain+'.'+gridname # overwrite
                
                origin = torch.tensor(np.mean(grids,axis=0)) # orient around grid center

            t2 = time.time()
            grids = grids - origin.squeeze().numpy()
            Grec, grids = receptor_graph_from_structure(propnpz, grids, origin,
                                                        edgemode=self.edgemode,
                                                        edgedist=self.edgedist[1],
                                                        edgek=self.edgek[1],
                                                        ball_radius=self.ball_radius,
                                                        gridchain=gridchain,
                                                        randomize=self.randomize)
            t3 = time.time()

            if Grec == None:
                print(f"Receptor num nodes exceeds max cut 3000")
                return NullArgs 
            
            elif Grec.number_of_edges() > self.maxedge or Grec.number_of_nodes() > self.maxnode:
                print(f"Receptor num edges {Grec.number_of_edges()} exceeds max cut {self.maxedge}")
                return NullArgs

        #else:
        except:
            print("failed to read %s"%target)
            return NullArgs
        
        info['name'] = target
        info['com'] = origin
        info['is_ligand'] = is_ligand
        info['grididx'] = np.where(Grec.ndata['attr'][:,0]==1)[0] #aa=unk type
        info['grid'] = grids
        info['eval_struct'] = (is_ligand and mol2type == 'single')

        t4 = time.time()
        #if Glig == None:
        #    print(Grec.number_of_nodes(), 0, t4-t0, t1-t0, t2-t1, t3-t2)
        #else:
        #    print(Grec.number_of_nodes(), Glig.number_of_nodes(), t4-t0, t1-t0, t2-t1, t3-t2)
        
        return Grec, Glig_list, cats, mask, keyxyz, keyidx_list, blabel_list, info #label

    def read_ligands(self, target, ligands, keyatomf, mol2path, actives=[], mol2type='single'):
        if mol2type == 'batch':
            Glig_list, Gnat_list, keyidx_list, atms_list, tags_read = self.read_by_batch_mol2(mol2path, keyatomf, tags=ligands)
            origin = []

        else:
            Glig_list, Gnat_list, keyidx_list, atms_list, tags_read = self.read_by_single_mol2(target, ligands, keyatomf, mol2path=mol2path)
            tags_read = ligands
            
        blabel_list = [(1 if tag in actives else 0) for tag in tags_read]
        Gnat, keyxyz = None, None
        origin = np.zeros(3)
        
        if len(actives) > 0:
            inat = ligands.index(actives[0])
            Gnat = Gnat_list[inat]
            
            origin = torch.mean(Gnat.ndata['x'],axis=0) # 1,3
            Gnat.ndata['x'] = Gnat.ndata['x'] - origin # move lig to origin
            keyxyz = Gnat.ndata['x'][keyidx_list[inat]] # K x 3

        return blabel_list, Glig_list, keyidx_list, atms_list, Gnat, keyxyz, origin

    def read_by_single_mol2(self, target, ligands, keyatomf, mol2path):
        Glig_list = [] 
        Gnat_list = [] 
        keyidx_list = []
        atms_list = []
        tags_read = []                       
        origin = None
        Gnat = None
        keyxyz = None
        
        for lig in ligands:
            mol2 = mol2path+lig+'.ligand.mol2'
            confmol2 = mol2path+lig+'.conformers.mol2'
            if not self.pert or not os.path.exists(confmol2):
                confmol2 = None
                
            Gnat,Glig,atms = ligand_graph_from_mol2(mol2,
                                                    dcut=self.edgedist[1], #unused
                                                    read_alt_conf=confmol2,
                                                    mode=self.edgemode,top_k=self.edgek[0],
                                                    drop_H=False)

            if Glig == None or not atms:
                print("skip %s for ligand read failure %s"%(target,mol2),file=sys.stderr)
                continue
            
            #if random.random() < self.noiseP:
            #    Glig, randoms = give_noise_to_lig(Glig)

            com = torch.mean(Glig.ndata['x'],axis=0) # 1,3
            Glig.ndata['x'] = Glig.ndata['x'] - com # move lig to origin
                
            keyidx = identify_keyidx(lig, atms, keyatomf)
                
            if not keyidx: continue
            keyidx_list.append(keyidx)
            Glig_list.append(Glig)
            Gnat_list.append(Gnat)
            atms_list.append(atms)
            tags_read.append(lig)                

        return Glig_list, Gnat_list, keyidx_list, atms_list, tags_read

    def read_by_batch_mol2(self, mol2, keyatomf, tags=[], randrot=False, random_sample=-1):
        Glig_list = [] 
        blabel_list = []
        keyidx_list = []
        atms_list = []
        #keyxyz_list = []

        (elems,_,bonds,borders,xyz,nneighs,atms,atypes,tags) = myutils.read_mol2_batch(mol2, tags_read=tags)

        tags_processed = []
        for e,b,o,x,n,a,at,tag in zip(elems,bonds,borders,xyz,nneighs,atms,atypes,tags):
            try:
            #if True:
                keyidx = identify_keyidx(tag, a, keyatomf)
                if not isinstance(keyidx, list):
                    print("key name violation %s"%(tag))
                    continue
                
                keyidx_list.append(keyidx)
                tags_processed.append(tag)

                args = e,_,b,o,x,n,a,at
                Glig = generate_ligand_graph(args, mode=self.edgemode, top_k=self.edgek[0])
            
                atms_list.append(a)
                
                #keyxyz = Glig.ndata['x'][keyidx] # K x 3
                #keyxyz_list.append(keyxyz)
                #x = Glig.ndata['x']
                #if randrot:
                #    Glig = myutils.RandRG(Glig)
                
                Glig_list.append(Glig)

            #else:
            except:
                continue
        t2 = time.time()

        # assume no conformer
        Gnat_list = Glig_list
        
        return Glig_list, Gnat_list, keyidx_list, atms_list, tags_processed
    
def receptor_graph_from_structure(npz, grids, origin, edgemode, edgedist, edgek, ball_radius=8.0, gridchain=None, randomize=0.0):
    prop = np.load(npz) #parentpath+target+".prop.npz"
    xyz         = prop['xyz_rec']
    charges_rec = prop['charge_rec'] 
    atypes_rec  = prop['atypes_rec'] #1-hot
    anames      = prop['atmnames']
    aas_rec     = prop['aas_rec'] #rec only
    sasa_rec    = prop['sasa_rec']
    bnds    = prop['bnds_rec']
    atypes  = np.array([types.find_gentype2num(at) for at in atypes_rec]) # string to integers

    # mask self
    if gridchain != None: 
        iexcl   = [i for i,a in enumerate(prop['atmres_rec']) if a[0].split('.')[0] == gridchain]
        xyz[iexcl] += 1000.0
            
    # orient around ligand center
    origin = origin.squeeze().numpy()
    xyz = xyz - origin # also move receptor & grid to origin
    #grids = grids - origin #already moved

    '''
    out = open('test.xyz','w')
    for x in xyz:
        out.write("C %8.3f %8.3f %8.3f\n"%tuple(x))
    for x in grids:
        out.write("G %8.3f %8.3f %8.3f\n"%tuple(x))
    out.close()'''

    # randomize the receptor coordinates
    if randomize > 1e-3:
        randxyz = 2.0*randomize*(0.5 - np.random.rand(len(xyz),3))
        xyz = xyz + randxyz
        
    ## 4. append grid info to receptor graph: grid points as "virtual" nodes
    ngrids = len(grids)
    anames = np.concatenate([anames,['grid%04d'%i for i in range(ngrids)]])
    aas_rec = np.concatenate([aas_rec, [0 for _ in grids]]) # unk
    atypes = np.concatenate([atypes, [0 for _ in grids]]) #null
    sasa_rec = np.concatenate([sasa_rec, [0.0 for _ in grids]]) # 
    charges_rec = np.concatenate([charges_rec, [0.0 for _ in grids]])
    xyz = np.concatenate([xyz,grids])
    d2o = np.sqrt(np.sum(xyz*xyz,axis=1))

    aas1hot = np.eye(types.N_AATYPE)[aas_rec]
    atypes  = np.eye(max(types.gentype2num.values())+1)[atypes] # convert integer to 1-hot
    sasa    = np.expand_dims(sasa_rec,axis=1)
    charges = np.expand_dims(charges_rec,axis=1)
    d2o = d2o[:,None]
        
    # Make receptor graph
    G_atm = make_atm_graphs(xyz, grids,
                            [aas1hot,atypes,sasa,charges,d2o],
                            bnds, anames,
                            edgemode=edgemode, edgek=edgek, edgedist=edgedist, ball_radius=ball_radius )

    return G_atm, grids
    
def make_atm_graphs(xyz, grids, obt_fs, bnds, atmnames,
                    edgemode, edgek, edgedist, ball_radius, 
                    CBonly=False, use_l1=False, maxnode=3000 ):
    # Do KD-ball neighbour search -- grabbing a <dist neighbor from gridpoints
    t0 = time.time()
    kd      = scipy.spatial.cKDTree(xyz)
    indices = []
    kd_ca   = scipy.spatial.cKDTree(grids)
    indices = np.concatenate(kd_ca.query_ball_tree(kd, ball_radius))
    nxyz0 = len(xyz)

    idx_ord = list(np.array(np.unique(indices),dtype=np.int16)) #atom indices close to any grid points
    if CBonly:
        idx_ord = [i for i in idx_ord if atmnames[i] in ['N','CA','C','O','CB']]

    if len(idx_ord) > maxnode: return 

    xyz     = xyz[idx_ord]
    natm = xyz.shape[0]-grids.shape[0] # real atoms
        
    t1 = time.time()
    bnds_bin = np.zeros((len(xyz),len(xyz)))
    newidx = {idx:i for i,idx in enumerate(idx_ord)}
        
    # new way -- just iter through bonds
    for i,j in bnds:
        if i not in newidx or j not in newidx: continue # excluded node by kd
        k,l = newidx[i], newidx[j] 
        bnds_bin[k,l] = bnds_bin[l,k] = 1
    for i in range(len(xyz)): bnds_bin[i,i] = 1 #self
    ub,vb = np.where(bnds_bin)
    t2 = time.time()

    # Concatenate coord & centralize xyz to ca.
    xyz = torch.tensor(xyz).float()
    dX = torch.unsqueeze(xyz[None,],1) - torch.unsqueeze(xyz[None,],2)

    ## 2) Connect by distance
    ## Graph connection: u,v are nedges & pairs to each other; i.e. (u[i],v[i]) are edges for every i
    u,v,D = find_dist_neighbors(dX, edgemode, top_k=edgek, dcut=edgedist)

    # reselect edges between grids
    t3 = time.time()

    ## Edge index for grid-grid
    N = u.shape[0] # num edges
    '''
    incl_e = np.zeros(N,dtype=np.int16)+1
    for i,(a,b) in enumerate(zip(u,v)):
        if a < natm or b < natm: continue
        if D[a,b] > edgedist: incl_e[i] = -1
    '''

    # new logic -- prv one connected all receptor-receptor topk
    within = (D<edgedist)
    within[:natm,:] = within[:,:natm] = 1 # allow topK edges whatsoever
    incl_e = within[u,v]

    t3a = time.time()
            
    ii = np.where(incl_e>=0)
    u = u[ii]
    v = v[ii]

    t4 = time.time()

    ## reduce edge connections
    N = u.shape[0] # num edges

    # distance
    t4a = time.time()
    w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None]
    w1hot = distance_feature('std',w,0.5,5.0) # torch.tensor
       
    bnds_bin = torch.tensor(bnds_bin[v,u]).float() #replace first bin (0.0~0.5 Ang) to bond info
    w1hot[:,0] = bnds_bin # chemical bond

    t4b = time.time()
    #grid_neighs = torch.tensor([float(i >= natm and j >= natm) for i,j in zip(u,v)]).float()
    grid_neighs = ((u>=natm)*(v>=natm)).float()
    w1hot[:,1] = grid_neighs # whether u,v are grid neighbors -- always 0 if split

    t4c = time.time()
    # concatenate all one-body-features 
    obt  = []
    for i,f in enumerate(obt_fs):
        if len(f) > 0: obt.append(f[idx_ord])
    obt = np.concatenate(obt,axis=-1)

    ## Construct graphs
    # G_atm: graph for all atms
    G_atm = dgl.graph((u,v))
    G_atm.ndata['attr'] = torch.tensor(obt).float() 
    G_atm.edata['attr'] = w1hot 
    G_atm.edata['rel_pos'] = dX[:,u,v].float()[0]
    G_atm.ndata['x'] = xyz[:,None,:] # not a feature, just info to TRnet

    te = time.time()
    # time consuming: t4~te & t3~t4
    #print(t1-t0,t2-t1,t3-t2,t4-t3,te-t4,t4a-t4,t4b-t4a,t4c-t4b,te-t4c)
    
    return G_atm
    
def receptor_graph_from_motifnet(npz,K,dcut=1.8,mode='dist',top_k=8,debug=False): # max 26 edges
    data = np.load(npz,allow_pickle=True)
    grids = data['grids']
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
    
def normalize_distance(D,maxd=5.0):
    d0 = 0.5*maxd #center
    m = 5.0/d0 #slope
    feat = 1.0/(1.0+torch.exp(-m*(D-d0)))
    return feat

def find_dist_neighbors(dX,mode='dist',top_k=8,dcut=4.5):
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
            
        #args = myutils.read_mol2(mol2,drop_H=drop_H)
        args = read_mol2(mol2)
    except:
        #args = myutils.read_mol2(mol2,drop_H=drop_H)
        print("failed to read mol2", mol2)
        return False, False, False

    if not args:
        return False, False, False

    #if '3asx' in mol2: print(args)
    xyz_nat, atms = args[4], args[6]
    
    if len(xyz_alt) > 1:
        ialt = min(np.random.randint(len(xyz_alt)),len(xyz_alt)-1)
        xyz = torch.tensor(xyz_alt[ialt]).float()
        args = list(args)
        args[4] = xyz
    
    Glig = generate_ligand_graph(args,
                                 mode=mode, top_k=top_k)
    
    Gnat = copy.deepcopy(Glig)
    Gnat.ndata['x'] = torch.tensor(xyz_nat).float()[:,None,:]
    
    return Glig, Gnat, atms

def read_mol2(mol2):
    read_cont = 0
    qs = []
    elems = []
    xyzs = []
    bonds = []
    borders = []
    atms = []
    atypes = []
    
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = -1
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
            
            if elem not in myutils.ELEMS:
                elem = 'Null'
            
            atms.append(words[1])
            atypes.append(words[5])
            elems.append(elem)
            qs.append(float(words[-1]))
            xyzs.append([float(words[2]),float(words[3]),float(words[4])]) 
                
        elif read_cont == 2:
            # if words[3] == 'du' or 'un': print(mol2)
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
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
    
    return np.array(elems)[nonHid], np.array(qs)[nonHid], bonds, borders, np.array(xyzs)[nonHid], np.array(nneighs)[nonHid], atms, atypes #np.array(atms)[nonHid]

def generate_ligand_graph(args,
                          top_k=16, dcut=5.0, mode='dist'):

    elems, qs, bonds, borders, xyz, nneighs, atms, atypes = args
    X = torch.tensor(xyz[None,]) #expand dimension
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    u,v,d = find_dist_neighbors(dX,dcut=dcut,mode=mode,top_k=top_k)

    #print(mode, u.shape, d.shape, top_k)
    Glig = dgl.graph((u,v))

    ## 1. node features
    obt  = []
    elems = [myutils.ELEMS.index(a) for a in elems]
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

    ## 2. edge features
    # 1-hot encode bond order
    ib = np.zeros((xyz.shape[0],xyz.shape[0]),dtype=int)
    for k,(i,j) in enumerate(bonds): ib[i,j] = ib[j,i] = k
    
    border_ = np.zeros(u.shape[0], dtype=np.int64)
    d_ = torch.zeros(u.shape[0])
    t1 = time.time()

    for k,(i,j) in enumerate(zip(u,v)):
        border_[k] = borders[ib[i,j]]
        d_[k] = d[i,j]
    t2 = time.time()

    edata = torch.eye(5)[border_] #0~3
    edata[:,-1] = normalize_distance(d_) #4

    ## 3. ligand global properties
    nflextors = 0
    for i,j in bonds:
        if elems[i] != 1 and elems[j] != 1 and ib[i,j] > 1:
            nflextors += 1
    nflextors = np.eye(10)[min(9,nflextors)]
    
    kappa = kappaidx.calc_Kappaidx(atypes, bonds, False)
    kappa = [(kappa[0]-40.0)/40.0, (kappa[1]-15.0)/15.0, (kappa[2]-10.0)/10.0]

    com = np.mean(xyz,axis=0)
    _xyz = xyz - com
    inertia = np.dot(_xyz.transpose(), _xyz)
    eigv,_ = np.linalg.eig(inertia)
    principal_values = (np.sqrt(np.sort(eigv)) - 20.0)/20.0 #normalize b/w -1~1
    natm = (len([a for a in elems if a > 1])-25.0)/25.0 #norm
    naro = (len([a for a in atypes if a in ['C.ar','C.aro','N.ar']])-6.0)/6.0
    
    nacc,ndon = -1.0, -1.0
    for i,e in enumerate(elems):
        if e not in [3,4]: continue
        js = np.where(ib[i,:]>0)[0]
        hasH = np.sum(np.array(elems)[js] == 1)
        if hasH:
            ndon += 0.2 #sp3 donor here but not rec
            #print("don", atms[i], ndon)
        else:
            nacc += 0.2
            #print("rec", atms[i], nccc)

    # concatenate
    # nfeat: 19
    gdata = np.concatenate([nflextors,kappa,[nacc,ndon,naro],list(principal_values)])
    Glig.ndata['attr'] = torch.tensor(obt).float()
    Glig.ndata['x'] = torch.tensor(xyz).float()[:,None,:]
    Glig.edata['attr'] = edata
    Glig.edata['rel_pos'] = dX[:,u,v].float()[0]
    setattr(Glig, "gdata", torch.tensor(gdata).float())
    
    Glig.ndata['Y'] = torch.zeros(obt.shape[0],3) # placeholder

    return Glig

def identify_keyidx(target, atms, keyatomf):
    if '/' in target: target = target.split('/')[-1]
    
    ## find key idx as below -- TODO
    # 1,2 2 max separated atoms along 1-st principal axis
    # 3:

    ## temporary: hard-coded
    # perhaps revisit as 3~4 random fragment centers -- once fragmentization logic implemented

    keyatoms = np.load(keyatomf,allow_pickle=True)
    if 'keyatms' in keyatoms:
        keyatoms = keyatoms['keyatms'].item()

    if target not in keyatoms: return False
    
    keyidx = [atms.index(a) for a in keyatoms[target] if a in atms]

    if len(keyidx) > 10:
        keyidx = list(np.random.choice(keyidx,10,replace=False))
        
    return keyidx

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

def collate(samples):
    #samples should be G,info

    # check Gatm only  -- Glig, keyxyz 
    valid = [v for v in samples if v != None]
    valid = [v for v in valid if (v[0] != False and v[0] != None)]
    if len(valid) == 0:
        return

    is_ligand = [s[-1]['is_ligand'] for s in valid]
    if sum(is_ligand) == 0:
        is_ligand = False
    elif sum(is_ligand) == len(is_ligand):
        is_ligand = True
    else:
        sys.exit("ligand/non-ligand entry cannot be mixed together")
        return

    Grec = []
    Glig = []
    cats = []
    masks = []
    keyxyz = []
    keyidx = []
    blabel = []
    _info = []
    for s in valid: #Grec, Glig, keyxyz, keyidx, info
        Grec.append(s[0])
        cats.append(s[2])
        masks.append(s[3])
        _info.append(s[-1])
        
    Grec = dgl.batch(Grec)

    if None in cats:
        cats = None
        masks = None
    else:
        cats = torch.stack(cats,dim=0).squeeze()
        masks = torch.stack(masks,dim=0).float().squeeze()
        if len(cats.shape) == 2: cats = cats[None,] # B x N x T
        if len(masks.shape) == 1: masks = masks[None,:] # B x M

    # concat info into torch
    info = {key:[] for key in _info[0]}
    for args in _info:
        for key in args:
            info[key].append(args[key])
    info['grid'] = torch.tensor(np.array(info['grid']))

    grididx = []
    i = 0
    for n,idx in zip(Grec.batch_num_nodes(),info['grididx']):
        grididx.append(torch.tensor(idx,dtype=int)+i)
        i += n
    info['grididx'] = torch.cat(grididx,dim=0) #idx in bGrec

    if is_ligand:
        Glig = dgl.batch(s[1])
        #print([g.gdata.shape for g in s[1]])
        gdata = torch.stack([g.gdata for g in s[1]])
        setattr( Glig, "gdata", gdata ) #batched
        
        try:
            keyxyz = s[4].squeeze()[None,:,:]
        except:
            return 
        
        keyidx = [torch.eye(n)[idx] for n,idx in zip(Glig.batch_num_nodes(),s[5])]
        blabel = torch.tensor(s[6],dtype=float)
        info['nK'] = torch.tensor(info['nK'])

    else:
        Glig = None
        keyxyz = torch.tensor(0.0)
        keyidx = torch.tensor(0.0)

    # below contains l0 features only
    #if Glig != None:
    #    print("!", Glig.batch_num_nodes())
    
    #print("size", Grec.number_of_nodes(), Glig.number_of_nodes(), s[5])
    return Grec, Glig, cats, masks, keyxyz, keyidx, blabel, info
    
