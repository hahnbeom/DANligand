import torch
import sys,os
import time
import dgl
#import dgl.function as fn
import torch.nn as nn
import random
#import torch.nn.functional as F
import numpy as np
from src.model_trigon_2 import SE3TransformerWrapper
import src.dataset as dataset
from src.myutils import generate_pose, report_attention, show_how_attn_moves
from os import listdir
from os.path import join, isdir, isfile

import warnings
warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

## cutline ---- ligandsize : 150 gridsize : 2200
MAXEPOCH = 500
BATCH = 2 # ~3time faster w/ B=10, ~4 times w/ B=20
LR = 1.0e-4 #2.0e-3
K = 4 # num key points
datapath = '/ml/motifnet/TRnet.ligand'
VERBOSE = '-v' in sys.argv
modelname = sys.argv[1]

w_reg = 1.e-4
w_spread = 5.0

setparams = {'K':K,
             'datapath':datapath,
             'neighmode':'topk',
             'topk'  : 16,
             'mixkey': True
             }

modelparams = {'num_layers_lig':3,
               'num_layers_rec':2,
               'num_channels':16, #within se3
               'l1_in_features':0,
               'l0_out_features':32, #at Xatn
               'n_heads_se3':4,
               'embedding_channels':32,
               'c':32,
               'n_trigonometry_module_stack':5, 
               'div':4,
               'K':K,
               'dropout':0.2,
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate(samples):
    Grec = []
    Glig = []
    
    label = []
    labelxyz = []
    info = []
    for s in samples:
        if s == None: continue
        Grec.append(s[0])
        Glig.append(s[1])
        labelxyz.append(s[2])
        label.append(torch.tensor(s[3]))
        info.append(s[4])

    if len(labelxyz) == 0:
        return None, None, None, None, None, 0
    elif len(labelxyz) == 1:
        labelxyz = torch.cat(labelxyz)[None,...]
    else:
        labelxyz = torch.stack(labelxyz,dim=0).squeeze()
    label = torch.stack(label,dim=0).squeeze()

    a = dgl.batch(Grec)
    b = dgl.batch(Glig)

    return dgl.batch(Grec), dgl.batch(Glig), labelxyz, label, info, len(info)

def structural_loss(prefix, Yrec, Ylig):
    # print(Ylig)

    dY = Yrec-Ylig # BxKx3
    loss1 = torch.sum(dY*dY,dim=1)

    N = Yrec.shape[0] # batch
    loss1_sum = torch.sum(loss1)/N
    meanD = torch.mean(torch.sqrt(loss1))

    #for k in range(K):
    #    print("%-10s %1d"%(prefix, k), loss[k], Yrec[k], Ylig[k])

    dY = torch.sqrt(torch.sum(dY*dY,dim=-1)) #BxK, peratm-dist
    
    return loss1_sum, meanD, dY

def spread_loss(Ylig, A, G, sig=2.0): #label(B x K x 3), attention (B x Nmax x K)
    loss2 = torch.tensor(0.0)
    i = 0
    N = A.shape[0]
    #A: B x max(N) x 4
    #G['x']: N1+N2... 1 x 3
    # Ylig: B x 4 x 3
    
    for b,n in enumerate(G.batch_num_nodes()): #batch
        x = G.ndata['x'][i:i+n,:] #N x 1 x 3
        z = A[b,:n,:] # N x K
        y = Ylig[b][None,:,:].squeeze() # 1 x 4 x 3
        
        #print(b,int(i),int(i+n),x.shape, z.shape)
        dX = x-y
        
        overlap = torch.exp(-torch.sum(dX*dX,axis=-1)/(sig*sig)) # N x 4
        print(overlap.shape, z.shape)
        if z.shape[0] != overlap.shape[0]: continue

        loss2 = loss2 - torch.sum(overlap*z)

        i += n
    return loss2 # max -(batch_size x K)

model = SE3TransformerWrapper(**modelparams)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR) #1.0e-3 < e-4

if os.path.exists("models/%s/model.pkl"%modelname): 
    checkpoint = torch.load(join("models", modelname, "model.pkl"), map_location=device)

    model_dict = model.state_dict()
    trained_dict = {}
    for key in checkpoint["model_state_dict"]:
        if key.startswith("module."):
            newkey = key[7:]
            trained_dict[newkey] = checkpoint["model_state_dict"][key]
        else:
            trained_dict[key] = checkpoint["model_state_dict"][key]
    
    model_dict.update(trained_dict)
    model.load_state_dict(model_dict,strict=False)
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    startepoch = checkpoint["epoch"]+1
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]

    if 'mae' not in train_loss: train_loss['mae'] = []
    if 'reg' not in train_loss: train_loss['reg'] = []
    if 'mae' not in valid_loss: valid_loss['mae'] = []

    print("Loading a checkpoint at epoch %d"%startepoch)

print(count_parameters(model))

keyatms = np.load(f'{datapath}/keyatom.def.npz',allow_pickle=True)#['keyatms'].item()
if 'keyatms' in keyatms: keyatms = keyatms['keyatms'].item()

print("preparing dataset")

# very slow on "a in keyatms"
validmols = [a for a in np.load('data/validlist.npy')] # 1905 valids
#validmols = [a for a in np.load('data/trainlist.npy')] # 1905 valids
validmols.sort()

validset = dataset.DataSet(validmols,**setparams)#K=K,datapath=datapath,neighmode=neighmode)#,pert=True)

print(f"Data loading done for {len(validmols)} validation molecules.")

generator_params = {
    'shuffle': False,
    'num_workers': 8, #1 if '-debug' in sys.argv else 10,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': BATCH,
    'worker_init_fn' : np.random.seed()
}
valid_generator = torch.utils.data.DataLoader(validset, **generator_params)

torch.cuda.empty_cache()

# validate by structure generation
loss_v = {'total':[],'mae':[],'loss1':[],'loss2':[]}
with torch.no_grad():    
    for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(valid_generator):
        if b <= 1: continue # hack!
        
        G1 = G1.to(device)
        G2 = G2.to(device)
            
        M = G2.ndata['x'].shape[0]
        #labelidx = torch.eye(M)[keyidx].to(device) # K x M
        labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
                
        labelxyz = labelxyz.to(device)

        Yrec,z = model( G1, G2, labelidx )

        prefix = "VALID "+info[0]['name']
        loss1,mae,peratm = structural_loss(prefix, Yrec, labelxyz ) #both are Kx3 coordinates
            
        loss2 = w_spread*spread_loss( labelxyz, z, G1 )
        loss = loss1 + loss2
        loss_v['total'].append(loss.cpu().detach().numpy())
        loss_v['loss1'].append(loss1.cpu().detach().numpy())
        loss_v['loss2'].append(loss2.cpu().detach().numpy())
        loss_v['mae'].append(mae.cpu().detach().numpy())

        pnames = [a['name'] for a in info]

        if VERBOSE:
            keyatms = [a['keyatms'] for a in info]
            print(pnames, float(loss), float(mae), G1.ndata['x'].shape, keyatms)
            X = G1.ndata['x'].squeeze() 

            b = 0
            for i,(p,n,a,key,d) in enumerate(zip(pnames,G1.batch_num_nodes(),z,keyatms,peratm)):
                x = X[b:b+n]
                out = open(p+'.d%3.1f.keyatm.pdb'%(torch.mean(d)),'w')
                form = 'ATOM  %5d %4s%4s A%4d    %8.3f%8.3f%8.3f  1.00 %4.2f\n'
                
                for j,x_ in enumerate(x):
                    out.write(form%(j,'H','H',j,x_[0],x_[1],x_[2],0.0))
                    
                aname = ['F','Cl','Br','I']
                for k in range(4):
                    for j,x_ in enumerate(x):
                        if a[j,k] > 0.1:
                            out.write(form%(j,key[k],aname[k],j,x_[0],x_[1],x_[2],a[j,k]))
                out.close()
                b += n
                
