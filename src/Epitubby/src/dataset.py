import torch
import numpy as np
import dgl
import sys
import os

def collate(samples):
    #samples should be G,info
    valid = [v for v in samples if v[1] and v[1] != None]
    if len(valid) == 0:
        return False, False, {}
    
    try:
    #if True:
        Gs = []
        frag_emb = []
        info = {key:[] for key in samples[0][2].keys()}
        
        for s in valid:
            frag_emb.append(s[0])
            Gs.append(s[1])
            for key in s[2]:
                info[key].append(s[2][key])

        bG = dgl.batch(Gs)
        frag_emb = torch.stack(frag_emb,dim=0)
        
        #info['label'] = torch.stack(info['label'],dim=0)
        
        return frag_emb, bG, info

    except:
        #print(Gs)
    #else:
        print("failed collation")
        return None, None, {}

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets, datapath='/ml/motifnet/HmapPPDB/trainable/',
                 neighmode='topk',
                 dcut=12.0, top_k=12,
                 same_frag=False,
                 debug=False):
        
        self.targets = targets
        self.datapath = datapath
        self.neighmode = neighmode
        self.dcut = dcut
        self.top_k = 8
        self.same_frag = same_frag
        self.debug = debug
            
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index): #N: maximum nodes in batch
        interface_npz = self.datapath+self.targets[index]+'.interface.npz'
        pdbid = self.targets[index][:4]

        # 1. random pick single entry from the interface
        data = np.load(interface_npz,allow_pickle=True)
        frags = data['frags']
        if len(frags) == 0:
            return False, False, {}

        if self.same_frag:
            ifrag = 0
        else:
            ifrag = np.random.randint(len(frags))

        frag = frags[ifrag] #rc
        aas_frag = data['frag_aas'][ifrag]
        
        aa1index = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        aas_frag = np.array([aa1index.index(aa) for aa in aas_frag],dtype=int)
        
        rec_interface = data['interface'][ifrag] #rc; label
        frag_chain = frag[0][0]
        rec_chain = rec_interface[0][0]

        # sanity check for files
        fs = [self.datapath+'../ESMembedding/%s%s.embedding.npz'%(pdbid,rec_chain),
              self.datapath+'../ESMembedding/%s%s.embedding.npz'%(pdbid,frag_chain)]
        for f in fs:
            if not os.path.exists(f):
                print(f"missing {f} for {self.targets[index]}")
                return False, False, {}
            
        ## 2. retrieve ESMfold embedding
        rec_emb_data = np.load(fs[0],allow_pickle=True)
        rec_s = rec_emb_data['s']
        #rec_z = rec_emb_data['z'] #pairwise
        frag_emb_data = np.load(fs[1],allow_pickle=True)
        frag_s = frag_emb_data['s']
        #frag_z = frag_emb_data['z']

        # add dummy embedding at the end
        rec_s = np.concatenate([rec_s,np.zeros((1,rec_s.shape[-1]))])
        #rec_z = np.concatenate([rec_z,np.zeros((1,rec_s.shape[-1]))])
        frag_s = np.concatenate([frag_s,np.zeros((1,frag_s.shape[-1]))])
        #frag_z = np.concatenate([frag_z,np.zeros((1,frag_z.shape[-1]))])

        ## 3. retrieve interface label definition
        prop_npz = self.datapath+pdbid+'.prop.npz'
        propdata = np.load(prop_npz,allow_pickle=True)
         
        xyz_rec = propdata['xyz'].item()[rec_chain]
        aas_rec = propdata['aas'].item()[rec_chain]
        rc_rec = propdata['rc'].item()[rec_chain]

        iemb_frag = [int(rc.split('.')[-1])-1 for rc in frag]
        iemb_rec = [int(rc.split('.')[-1])-1 for rc in rc_rec]

        aas_rec = np.array([aa1index.index(aas_rec[rc]) for rc in rc_rec],dtype=int)
        ## 4. build rec_graph & frag_feature

        if self.debug:
            print("info: ", self.targets[index], frag_chain, rec_chain, frag, frag_s.shape)
            
        try:
            G_rec = self.make_graph(xyz_rec, aas_rec, rec_s[iemb_rec]) #
            
            aas_frag = np.eye(20)[aas_frag] # 1hot encoded
            pos_enc = np.zeros((aas_frag.shape[0],4)) #4-dim positional encoding
            pos_enc[:,0] = np.sin(np.arange(1000,1000+aas_frag.shape[0]))
            pos_enc[:,1] = np.cos(np.arange(1000,1000+aas_frag.shape[0]))
            pos_enc[:,2] = np.sin(np.arange(1000,1000+aas_frag.shape[0])/10000**(1.0/32.0))
            pos_enc[:,3] = np.cos(np.arange(1000,1000+aas_frag.shape[0])/10000**(1.0/32.0))

            frag_emb = np.concatenate([aas_frag,pos_enc,frag_s[iemb_frag]],axis=1)
            frag_emb = torch.tensor(frag_emb)
            
            label = torch.tensor([rc_rec.index(rc) for rc in rec_interface],dtype=int)
            
            info = {'tag':self.targets[index]+'.'+str(ifrag),
                    'label':label}
        except:
            print("failed reading", self.targets[index])
            
            if self.debug:
                G_rec = self.make_graph(xyz_rec, aas_rec, rec_s[iemb_rec]) #

                aas_frag = np.eye(20)[aas_frag] # 1hot encoded
                frag_emb = np.concatenate([aas_frag,frag_s[iemb_frag]],axis=1)
                frag_emb = torch.tensor(frag_emb)
            
                label = torch.tensor([rc_rec.index(rc) for rc in rec_interface],dtype=int)
            
                info = {'tag':self.targets[index]+'.'+str(ifrag),
                        'label':label}
                
            return False, False, {}

        return frag_emb, G_rec, info

    def make_graph(self, xyz, aas, s):
        X = torch.tensor(xyz[None,]) #expand dimension
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        u,v,d_ = find_dist_neighbors(dX, dcut=self.dcut, mode=self.neighmode, top_k=self.top_k)

        d_ = d_[u,v]
        aas1hot = np.eye(20)[aas] # 1hot encoded
        pos_enc = np.zeros((aas.shape[0],4)) #4-dim positional encoding
        pos_enc[:,0] = np.sin(np.arange(aas.shape[0]))
        pos_enc[:,1] = np.cos(np.arange(aas.shape[0]))
        pos_enc[:,2] = np.sin(np.arange(aas.shape[0])/10000**(1.0/32.0))
        pos_enc[:,3] = np.cos(np.arange(aas.shape[0])/10000**(1.0/32.0))

        nodefeats = np.concatenate([aas1hot,pos_enc,s],axis=1)
        
        G = dgl.graph((u,v))
        G.ndata['attr'] = torch.tensor(nodefeats).float()
        G.ndata['x'] = torch.tensor(xyz).float()[:,None,:]
        G.edata['attr'] = d_[:,None].float()
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

### unused
def positional_embedding( x ):
    P = torch.zeros((1, 300, d))
    X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
    X = X / torch.pow(10000, torch.arange(0, d, 2, dtype=torch.float32) / d)
    P[:,:,0::2] = torch.sin(X)
    P[:,:,1::2] = torch.cos(X)


