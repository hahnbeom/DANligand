import torch
import sys
import time
import dgl
#import dgl.function as fn
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
from src.model_batch import SE3TransformerWrapper
import src.dataset as dataset
from src.myutils import generate_pose, report_attention

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

MAXEPOCH = 200
BATCH = 23 # ~3time faster w/ B=10, ~4 times w/ B=20
LR = 1.0e-4 #2.0e-3
K = 4 # num key points
verbose = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate(samples):
    Grec = []
    Glig = []
    
    label = []
    labelxyz = []
    info = []
    for s in samples:
        Grec.append(s[0])
        Glig.append(s[1])

        labelxyz.append(s[2])
        label.append(torch.tensor(s[3]))
        info.append(s[4])

    labelxyz = torch.stack(labelxyz,dim=0).squeeze()
    label = torch.stack(label,dim=0).squeeze()
    return dgl.batch(Grec), dgl.batch(Glig), labelxyz, label, info, len(samples)

def custom_loss(prefix, Yrec, Ylig):
    dY = Yrec-Ylig
    loss = torch.sum(dY*dY,dim=1)
    
    if verbose:
        for k in range(K):
            print("%-10s %1d %8.3f"%(prefix, k,float(loss[k])),
                  " %8.3f"*3%tuple(Yrec[k]), "|", " %8.3f"*3%tuple(Ylig[k]))

    N = Yrec.shape[0]
    loss_sum = torch.sum(loss)/N
    meanD = torch.mean(torch.sqrt(loss))
    if verbose:
        print("%-10s MSE: %8.3f / mean(d): %8.3f"%(prefix, float(loss_sum), meanD))
    return loss_sum, meanD

modelparams = {'num_layers_lig':4,
               'num_layers_rec':4,
               'dropout':0.1,
               'num_channels':16,
               'l0_out_features':8,
               'n_heads':4,
               'div':4,
               'K':K
}
model = SE3TransformerWrapper(**modelparams)

# zero initialize weighting FC layers
for i, (name, layer) in enumerate(model.named_modules()):
    if isinstance(layer,torch.nn.Linear) and "phi" in name or "W" in name:
        if isinstance(layer,torch.nn.ModuleList):
            for l in layer:
                if hasattr(l,'weight'): l.weight.data.fill_(0.0)
        else:
            if hasattr(layer,'weight'):
                layer.weight.data.fill_(0.0)

model.to(device)

print(count_parameters(model))

keyatms = np.load('data/keyatom.def.npz',allow_pickle=True)['keyatms'].item()
trainmols = [a for a in np.load('data/trainlist.npy') if a in keyatms] # read only if key idx defined
validmols = [a for a in np.load('data/validlist.npy') if a in keyatms] # read only if key idx defined


trainset = dataset.DataSet(trainmols,K=K)
validset = dataset.DataSet(validmols,K=K,pert=True)

generator_params = {
    'shuffle': False,
    'num_workers': 10, #1 if '-debug' in sys.argv else 10,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': BATCH,
    'worker_init_fn' : np.random.seed()
}
train_generator = torch.utils.data.DataLoader(trainset, **generator_params)
valid_generator = torch.utils.data.DataLoader(validset, **generator_params)
 
optimizer = torch.optim.Adam(model.parameters(), lr=LR) #1.0e-3 < e-4

#G1 = dataset.graph_from_motifnet('data/aces.score.npz')
#G1 = G1.to(device)

for epoch in range(MAXEPOCH):  
    loss_t = []
    mae_t = []
    t0 = time.time()
    for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(train_generator):
        G1 = G1.to(device) # rec
        G2 = G2.to(device) # lig

        M = G2.ndata['x'].shape[0]
        #labelidx = torch.eye(M)[keyidx].to(device) # B x K x M
        labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
        labelxyz = labelxyz.to(device)

        Yrec,_ = model( G1, G2, labelidx)

        prefix = "TRAIN %s"%info[0]['name']

        loss, mae = custom_loss( prefix, Yrec, labelxyz ) #both are Kx3 coordinates

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        loss_t.append(loss.cpu().detach().numpy())
        mae_t.append(mae.cpu().detach().numpy())
    t1 = time.time()
    #print(f"Time: {t1-t0}")

    # validate by structure generation
    loss_v = []
    mae_v = []
    with torch.no_grad():    
        for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(valid_generator):
            G1 = G1.to(device)
            G2 = G2.to(device)
            
            M = G2.ndata['x'].shape[0]
            #labelidx = torch.eye(M)[keyidx].to(device) # K x M
            labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
            labelxyz = labelxyz.to(device)

            Yrec,A = model( G1, G2, labelidx )

            if verbose:
                generate_pose(Yrec, keyidx, G2.ndata['x'].squeeze(), atms, epoch )
                report_attention( G1.ndata['x'].squeeze(), A, epoch )

            prefix = "VALID "+info[0]['name']
            loss,mae = custom_loss(prefix, Yrec, labelxyz ) #both are Kx3 coordinates
            
            loss_v.append(loss.cpu().detach().numpy())
            mae_v.append(mae.cpu().detach().numpy())
         
    print("Train/Valid: %3d %8.4f %8.4f (%8.4f %8.4f)"%(epoch, float(np.mean(loss_t)), float(np.mean(loss_v)),
                                                        float(np.mean(mae_t)), float(np.mean(mae_v)))
          )
