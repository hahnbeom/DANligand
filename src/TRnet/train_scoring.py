import torch
import sys
import time
import dgl
#import dgl.function as fn
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
#from src.model_batch import SE3TransformerWrapper
from src.model import SE3TransformerWrapper
import src.dataset as dataset
from src.myutils import generate_pose, report_attention, rmsd, make_pdb

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

MAXEPOCH = 100
BATCH = 1 # ~3time faster w/ B=10, ~4 times w/ B=20
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

def key_loss(prefix, Yrec, Ylig):
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

def score_loss(Xrec, Prec, Yrec, Xlig, keyidx):
    device = Xrec.get_device()
    
    N = Xlig.shape[0]
    Ylig = Xlig[keyidx]

    com = torch.mean(Yrec,dim=0)
    rms,U = rmsd(Ylig.squeeze(), Yrec) # gradient fails here...

    # Calculate displacement at origin frame
    Xrec = (Xrec - com)
    Xlig = torch.matmul(Xlig-com,U)

    Xlig = Xlig.transpose(0,1)

    dX = Xrec - Xlig
    D = torch.sqrt(torch.sum(dX**2, 2) + 1.0e-6)
    overlap = torch.exp(-D) # M x N

    # assert
    #idx = torch.argmax(overlap,dim=0)
    #for x,y in zip(Xrec[idx],Xlig.squeeze()):
    #    print(x,y)

    # temporary
    mask = torch.zeros((N,14)).to(device) # which cat each lig atm goes to
    mask[:,10] = 1.0 # N x K

    Psum = torch.einsum("mi,mn -> in", Prec, overlap) # K x N
    Psum = torch.mean(torch.matmul(Psum,mask),dim=0) # sum over N
    
    return -torch.sum(Psum)

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
                if hasattr(l,'weight'): l.weight.data.fill_(0.001) # DO NOT USE 0.0! or will get Nan at SVD
        else:
            if hasattr(layer,'weight'):
                layer.weight.data.fill_(0.001)

model.to(device)

print(count_parameters(model))

keyatms = np.load('data/keyatom.def.npz',allow_pickle=True)['keyatms'].item()
trainmols = [a for a in np.load('data/trainlist.npy')[:10] if a in keyatms] # read only if key idx defined
validmols = [a for a in np.load('data/validlist.npy') if a in keyatms] # read only if key idx defined

trainset = dataset.DataSet(trainmols,K=K)
validset = dataset.DataSet(validmols,K=K,pert=True)

generator_params = {
    'shuffle': False,
    'num_workers': 5, #1 if '-debug' in sys.argv else 10,
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
    loss_t = {'score':[],'key':[],'mae':[]}
    
    t0 = time.time()
    for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(train_generator):
        G1 = G1.to(device) # rec
        G2 = G2.to(device) # lig

        M = G2.ndata['x'].shape[0]
        labelidx = torch.eye(M)[keyidx].to(device) # B x K x M
        #labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
        labelxyz = labelxyz.to(device)

        Yrec,_ = model( G1, G2, labelidx)

        Prec = G1.ndata['attr'].to(torch.float32).to(device)
        Xrec = G1.ndata['x'].to(torch.float32).to(device)
        Xlig = G2.ndata['x'].to(torch.float32).to(device)
        
        prefix = "TRAIN %s"%info[0]['name']
        loss1, mae = key_loss( prefix, Yrec, labelxyz ) #both are Kx3 coordinates
        
        loss2 = score_loss( Xrec, Prec, Yrec, Xlig, keyidx )

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_t['score'].append(loss2.cpu().detach().numpy())
        loss_t['key'].append(loss1.cpu().detach().numpy())
        loss_t['mae'].append(mae.cpu().detach().numpy())

        '''
        for name,layer in model.named_modules():
            print(name,layer)
            if isinstance(layer,torch.nn.ModuleList):
                for l in layer: print(name,l.weight.grad)
            else: print(name,layer.weight.grad)
        '''
                
    t1 = time.time()

    #print(f"Time: {t1-t0}")

    # validate by structure generation
    loss_v = {'score':[],'key':[],'mae':[]}
    with torch.no_grad():    
        for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(valid_generator):
            G1 = G1.to(device)
            G2 = G2.to(device)
            
            M = G2.ndata['x'].shape[0]
            labelidx = torch.eye(M)[keyidx].to(device) # K x M
            #labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
            labelxyz = labelxyz.to(device)

            Yrec,A = model( G1, G2, labelidx )
            
            Prec = G1.ndata['attr'].to(torch.float32).to(device)
            Xrec = G1.ndata['x'].to(torch.float32).to(device)
            Xlig = G2.ndata['x'].to(torch.float32).to(device)

            if verbose:
                generate_pose(Yrec, keyidx, G2.ndata['x'].squeeze(), atms, epoch=epoch, report=True )
                report_attention( G1.ndata['x'].squeeze(), A, epoch )

            prefix = "VALID "+info[0]['name']
            loss1,mae = key_loss(prefix, Yrec, labelxyz ) #both are Kx3 coordinates
            loss2 = score_loss( Xrec, Prec, Yrec, Xlig, keyidx )
            loss = loss1 + loss2
            
            loss_v['score'].append(loss2.cpu().detach().numpy())
            loss_v['key'].append(loss1.cpu().detach().numpy())
            loss_v['mae'].append(mae.cpu().detach().numpy())
         
    print("Train/Valid: %3d %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f"%(epoch,
                                                                      float(np.mean(loss_t['score'])), float(np.mean(loss_v['score'])),
                                                                      float(np.mean(loss_t['key'])), float(np.mean(loss_v['key'])),
                                                                      float(np.mean(loss_t['mae'])), float(np.mean(loss_v['mae'])))
          )
