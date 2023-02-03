import torch
import numpy as np
import dgl
import sys,copy

def collate(samples):
    #samples should be G,info
    valid = [v[0] for v in samples]

    #try:
    if True:
        Gs = []
        info = {key:[] for key in samples[0][1].keys()}
        
        for s in samples:
            Gs.append(s[0])
            for key in s[1]:
                info[key].append(s[1][key])

        info['xyz_lig'] = torch.stack(info['xyz_lig'],dim=0)
        #info['pepidx'] = torch.stack(info['pepidx'],dim=0)
        #info['pepidx'] = torch.tensor(info['pepidx'],dtype=torch.uint8)
        bG = dgl.batch(Gs)
        
        return bG, info

    #except:
    else:
        print("failed collation")
        return None, {}

class DataSet(torch.utils.data.Dataset):
    def __init__(self, fs, datapath='/ml/pepbdb/',
                 neighmode='dist',
                 dcut=12.0, top_k=8,
                 peplen=5):
        
        self.fs = fs
        self.datapath = datapath
        self.neighmode = neighmode
        self.dcut = dcut
        self.top_k = 8
        self.peplen = peplen
        self.pepshift = int((peplen-1)/2)
            
    def __len__(self):
        return len(self.fs)
    
    def __getitem__(self, index): #N: maximum nodes in batch
        ipdb = index%len(self.fs)
        prefix = self.fs[ipdb].split('.')[0]
        cenres = int(self.fs[ipdb].split('.')[1])
        
        pdb = self.datapath+'/'+prefix+'.complex.pdb'
        #try:
        if True:
            G, xyz_lig, pepidx = self.read_pdb(pdb, cenres)
            info = {'tag':self.fs[ipdb],
                    'xyz_lig':xyz_lig.float(),
                    'pepidx':pepidx}
        #except:
        else:
            print("failed reading", pdb)
            return False, {}

        return G, info

    def read_pdb(self,pdb, pepcen):
        # make receptor as non-X, ligand as chain X
        xyz = []
        aas = []
        seqsep = []
        pepidx = []
    
        crs = []
        seqidx = -1
        for l in open(pdb):
            if not l.startswith('ATOM'): continue
            aname = l[12:16].strip()
            if aname != 'CA': continue
            
            chain = l[21]
            resno = l[22:26]
            
            if resno[-1] != ' ' or crs == []:
                seqidx += 1
            elif chain != crs[-1].split('.')[0]:
                seqidx += 200
            else:
                prvres = crs[-1].split('.')[-1]
                seqidx += int(resno[:-1])-int(prvres[:-1])

            aaindex = aa3toindex(l[17:20])
            if aaindex < 0: continue # 

            seqsep.append(seqidx)
            crs.append(chain+'.'+resno)
        
            crd = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            xyz.append(crd)
            aas.append(aaindex)
        
            if chain == 'X':
                pepidx.append(len(aas)-1)

        xyz = np.array(xyz)
        nrec = len(xyz) - len(pepidx)

        # store as label
        istart = nrec+pepcen-self.pepshift
        iend = nrec+pepcen+self.pepshift+1
        pepidx = pepidx[pepcen-self.pepshift:pepcen+self.pepshift+1]

        # take pep info at :nrec & pep
        xyz = np.concatenate([xyz[:nrec],xyz[pepidx]])
        xyz_lig = xyz[nrec:]
        pepidx = np.arange(nrec,nrec+self.peplen)

        # reorient coord at COM
        xyz_com = xyz_lig[self.pepshift]
        xyz_lig = xyz_lig - xyz_com # label
        xyz = xyz - xyz_com
        xyz[nrec:,:] = 0.0 # initialize at origin

        aas    = [a for i,a in enumerate(aas) if i < nrec or i in pepidx]
        seqsep = [a for i,a in enumerate(seqsep) if i < nrec or i in pepidx]

        X = torch.tensor(xyz[None,]) #expand dimension
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        u,v,d_ = find_dist_neighbors(dX, dcut=self.dcut, mode=self.neighmode, top_k=self.top_k)

        d_ = d_[u,v]
        aas1hot = torch.eye(20)[aas] # 1hot encoded

        sep2cen = torch.zeros(len(aas),dtype=float)
        sep2cen[:nrec] = 0
        sep2cen[nrec:] = torch.arange(-self.pepshift,self.pepshift+1)
        sep2cen = torch.tanh(0.3*sep2cen)

        is_pep = torch.zeros(len(aas),dtype=int)
        is_pep[nrec:] = 1

        is_pep = is_pep[:,None]
        sep2cen = sep2cen[:,None]
        nodefeats = torch.cat([aas1hot,sep2cen,is_pep],dim=1)

        seqsep = torch.tensor(seqsep)
        seqsep[nrec:] += 200
        seqsep = abs(seqsep[:,None] - seqsep[None,:])
        seqsep = seqsep[u,v]

        edata = torch.zeros((u.shape[0],2))
        edata[:,0] = torch.tanh( 0.03*seqsep )  #positional_embedding( seqsep )
        edata[:,1] = normalize_distance(d_) #2

        G = dgl.graph((u,v))
        G.ndata['attr'] = torch.tensor(nodefeats).float()
        G.ndata['x'] = torch.tensor(xyz).float()[:,None,:]
        G.edata['attr'] = edata
        G.edata['rel_pos'] = dX[:,u,v].float()[0]

        return G, torch.tensor(xyz_lig), pepidx

def aa3toindex(aa3):
    aa3list = ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU',
               'MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
    
    if aa3 in aa3list:
        return aa3list.index(aa3)
    else:
        return -1

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

### unused
def positional_embedding( x ):
    P = torch.zeros((1, 300, d))
    X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
    X = X / torch.pow(10000, torch.arange(0, d, 2, dtype=torch.float32) / d)
    P[:,:,0::2] = torch.sin(X)
    P[:,:,1::2] = torch.cos(X)


