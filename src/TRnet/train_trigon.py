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
BATCH = 3 # ~3time faster w/ B=10, ~4 times w/ B=20
LR = 1.0e-4 #2.0e-3
K = 4 # num key points
datapath = '/ml/motifnet/TRnet.ligand'
verbose = False
neighmode = 'topk' #'distT'
modelname = sys.argv[1]
w_reg = 1.e-4

modelparams = {'num_layers_lig':4,
               'num_layers_rec':4,
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

    labelxyz = torch.stack(labelxyz,dim=0).squeeze()
    label = torch.stack(label,dim=0).squeeze()

    a = dgl.batch(Grec)
    b = dgl.batch(Glig)

    return dgl.batch(Grec), dgl.batch(Glig), labelxyz, label, info, len(info)

def custom_loss(prefix, Yrec, Ylig):
    # print(Ylig)

    dY = Yrec-Ylig
    loss = torch.sum(dY*dY,dim=1)
    
    N = Yrec.shape[0]
    loss_sum = torch.sum(loss)/N
    meanD = torch.mean(torch.sqrt(loss))

    #for k in range(K):
    #    print("%-10s %1d"%(prefix, k), loss[k], Yrec[k], Ylig[k])
        
    return loss_sum, meanD

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

else:    
    if not isdir(join("models", modelname)):
        print("Creating a new dir at", join("models", modelname))
        os.mkdir(join("models", modelname))
        
    # zero initialize weighting FC layers
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer,torch.nn.Linear) and "phi" in name or "W" in name:
            if isinstance(layer,torch.nn.ModuleList):
                for l in layer:
                    if hasattr(l,'weight'): l.weight.data.fill_(0.0)
                else:
                    if hasattr(layer,'weight'):
                        layer.weight.data.fill_(0.0)

    train_loss = {'total':[],'mae':[],'reg':[]}
    valid_loss = {'total':[],'mae':[]}
    startepoch = 0

print(count_parameters(model))


keyatms = np.load(f'{datapath}/keyatom.def.npz',allow_pickle=True)#['keyatms'].item()
if 'keyatms' in keyatms: keyatms = keyatms['keyatms'].item()

print("preparing dataset")

# very slow on "a in keyatms"
trainmols = [a for a in np.load('data/trainlist.npy')] # 15641 trains
validmols = [a for a in np.load('data/validlist.npy')] # 1905 valids

trainset = dataset.DataSet(trainmols,K=K,datapath=datapath,neighmode=neighmode)
validset = dataset.DataSet(validmols,K=K,datapath=datapath,neighmode=neighmode)#,pert=True)

print(f"Data loading done for {len(trainmols)} train and {len(validmols)} validation molecules.")

generator_params = {
    'shuffle': True,
    'num_workers': 8, #1 if '-debug' in sys.argv else 10,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': BATCH,
    'worker_init_fn' : np.random.seed()
}
train_generator = torch.utils.data.DataLoader(trainset, **generator_params)
valid_generator = torch.utils.data.DataLoader(validset, **generator_params)

torch.cuda.empty_cache()
for epoch in range(startepoch,MAXEPOCH):  
    loss_t = {'total':[],'mae':[],'reg':[]}
    t0 = time.time()
    for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(train_generator):
        if b == 1: continue # hack!

        G1 = G1.to(device) # rec
        G2 = G2.to(device) # lig

        M = G2.ndata['x'].shape[0]
        #labelidx = torch.eye(M)[keyidx].to(device) # B x K x M
        labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
        labelxyz = labelxyz.to(device)

        Yrec,z = model( G1, G2, labelidx)

        prefix = "TRAIN %s"%info[0]['name']
        loss, mae = custom_loss( prefix, Yrec, labelxyz ) #both are Kx3 coordinates

        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters(): l2_reg += torch.norm(param)
        loss = loss + w_reg*l2_reg 

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        loss_t['total'].append(loss.cpu().detach().numpy())
        loss_t['reg'].append(l2_reg.cpu().detach().numpy())
        loss_t['mae'].append(mae.cpu().detach().numpy())

    train_loss['total'].append(np.array(loss_t['total']))
    train_loss['reg'].append(np.array(loss_t['reg']))
    train_loss['mae'].append(np.array(loss_t['mae']))
    t1 = time.time()
    #print(f"Time: {t1-t0}")

    # validate by structure generation
    loss_v = {'total':[],'mae':[]}
    with torch.no_grad():    
        for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(valid_generator):
            if b == 1: continue # hack!
            
            G1 = G1.to(device)
            G2 = G2.to(device)
            
            M = G2.ndata['x'].shape[0]
            #labelidx = torch.eye(M)[keyidx].to(device) # K x M
            labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
                
            labelxyz = labelxyz.to(device)

            Yrec,A = model( G1, G2, labelidx )

            prefix = "VALID "+info[0]['name']
            loss,mae = custom_loss(prefix, Yrec, labelxyz ) #both are Kx3 coordinates
            
            loss_v['total'].append(loss.cpu().detach().numpy())
            loss_v['mae'].append(mae.cpu().detach().numpy())
    
    valid_loss['total'].append(np.array(loss_v['total']))
    valid_loss['mae'].append(np.array(loss_v['mae']))

    form = "Train/Valid: %3d %8.4f %8.4f %8.4f (%8.4f %8.4f)"
    print(form%(epoch, float(np.mean(loss_t['total'])), float(np.mean(loss_v['total'])),
                float(np.mean(loss_t['reg'])),
                float(np.mean(loss_t['mae'])), float(np.mean(loss_v['mae']))))
    
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss}, join("models", modelname, "model_%d.pkl"%(epoch)))
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss}, join("models", modelname, "model.pkl"))
   #print("Train/Valid: %3d"%epoch, float(np.mean(loss_t)), float(np.mean(loss_v)),
    #float(np.mean(mae_t)), float(np.mean(mae_v)))

