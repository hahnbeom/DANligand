import torch
import sys,os
import time
import dgl
#import dgl.function as fn
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
from src.model_mix import SE3TransformerWrapper
import src.dataset_mix as dataset
from src.myutils import generate_pose, report_attention
from os import listdir
from os.path import join, isdir, isfile

import warnings
warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

MAXEPOCH = 1000
BATCH = 10 # ~3time faster w/ B=10, ~4 times w/ B=20
LR = 1.0e-4 #2.0e-3
K = 4 # num key points
verbose = False
datapath = '/home/hpark/data'
neighmode = 'topk' #'distT'
modelname = sys.argv[1]
w_reg = 1.e-4

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

        ## ensure all has same datatype
        datatype = s[4]['datatype'] 
        if len(info) > 0 and datatype != info[0]['datatype']:
            continue
        
        Grec.append(s[0])
        Glig.append(s[1])

        labelxyz.append(s[2])
        label.append(torch.tensor(s[3]))
        info.append(s[4])

    labelxyz = torch.stack(labelxyz,dim=0).squeeze()
    label = torch.stack(label,dim=0).squeeze()

    a = dgl.batch(Grec)
    b = dgl.batch(Glig)

    #print([a['datatype'] for a in info])
    return dgl.batch(Grec), dgl.batch(Glig), labelxyz, label, info, len(samples)

def custom_loss(prefix, Yrec, Ylig):
    dY = Yrec-Ylig
    loss = torch.sum(dY*dY,dim=1)
    
    #if verbose:
    #    for k in range(K):
            #print("%-10s %1d"%(prefix, k), loss[k],
            #      " %8.3f"*3%tuple(Yrec[k]), "|", " %8.3f"*3%tuple(Ylig[k]))

    N = Yrec.shape[0]
    loss_sum = torch.sum(loss)/N
    meanD = torch.mean(torch.sqrt(loss))
    if verbose:
        #print("%-10s MSE: %8.3f / mean(d): %8.3f"%(prefix, float(loss_sum), meanD))
        print("%-10s MSE/ mean(d): "%prefix, float(loss_sum), meanD)
    return loss_sum, meanD

modelparams = {'num_layers_lig':4,
               'num_layers_rec':4,
               'num_layers_denoise':2,
               'dropout':0.2,
               'num_channels':16, #within se3
               'l1_in_features':0,
               'l0_out_features':16, #at Xatn
               'n_heads':4,
               'nattn':1, 
               'div':4,
               'K':K
}

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
        
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    startepoch = checkpoint["epoch"]+1
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    
    if 'mae' not in train_loss: train_loss['mae'] = [[0.0] for _ in range(checkpoint["epoch"])]
    if 'maeR' not in train_loss: train_loss['maeR'] = [[0.0] for _ in range(checkpoint["epoch"])]
    if 'reg' not in train_loss: train_loss['reg'] = [[0.0] for _ in range(checkpoint["epoch"])]
    if 'mae' not in valid_loss: valid_loss['mae'] = [[0.0] for _ in range(checkpoint["epoch"])]
    if 'maeR' not in valid_loss: valid_loss['maeR'] = [[0.0] for _ in range(checkpoint["epoch"])]

    print("Loading a checkpoint at epoch %d"%startepoch)

    # freeze all 
    for name, param in model.named_parameters():
        if 'denoise' not in name:
            param.requires_grad = False
            print("freeze", name)

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

    train_loss = {'total':[],'mae':[],'reg':[],'maeR':[]}
    valid_loss = {'total':[],'mae':[],'maeR':[]}
    startepoch = 0

print(count_parameters(model))


keyatms = np.load(f'data/keyatom.def.npz',allow_pickle=True)#['keyatms'].item()
if 'keyatms' in keyatms: keyatms = keyatms['keyatms'].item()

print("preparing dataset")
# very slow on "a in keyatms"
trainmols = [a for a in np.load('data/trainlist.mix.npy')] # if a in keyatms] # read only if key idx defined
validmols = [a for a in np.load('data/validlist.mix.npy')] # if a in keyatms] # read only if key idx defined

set_params = {
    'K':K,
    'batchsize':10,
    'datapath':datapath,
    'neighmode':neighmode,
    'real_data_every_minibatch':1,#only real data at pretraining stage -- ~3 for mixed training
}

# make sure all entries in a mini batch share the same datatype
trainset = dataset.DataSet(trainmols,**set_params)
validset = dataset.DataSet(validmols,**set_params)

print(f"Data loading done for {len(trainmols)} train and {len(validmols)} validation molecules.")

generator_params = {
    'shuffle': False, #should presever in order to sync datatype through minibatch
    'num_workers': 10, #1 if '-debug' in sys.argv else 10,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': BATCH,
    'worker_init_fn' : np.random.seed()
}
train_generator = torch.utils.data.DataLoader(trainset, **generator_params)
valid_generator = torch.utils.data.DataLoader(validset, **generator_params)
 
for epoch in range(startepoch,MAXEPOCH):  
    loss_t = {'total':[],'mae':[],'reg':[],'maeR':[]}
    t0 = time.time()
    for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(train_generator):
        if b == 1: continue # hack!
            
        G1 = G1.to(device) # rec
        G2 = G2.to(device) # lig

        M = G2.ndata['x'].shape[0]
        #labelidx = torch.eye(M)[keyidx].to(device) # B x K x M
        labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
        labelxyz = labelxyz.to(device)
        realdata = (info[0]['datatype'] == 'real')

        Yrec,_ = model( G1, G2, labelidx, denoise=realdata )

        prefix = "TRAIN %s"%info[0]['name']

        loss, mae = custom_loss( prefix, Yrec, labelxyz ) #both are Kx3 coordinates
        print(prefix, realdata, mae)

        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters(): l2_reg += torch.norm(param)
        loss = loss + w_reg*l2_reg 

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        loss_t['total'].append(loss.cpu().detach().numpy())
        loss_t['reg'].append(l2_reg.cpu().detach().numpy())
        
        mae = mae.cpu().detach().numpy()

        if realdata:
            loss_t['maeR'].append(mae)
        else:
            loss_t['mae'].append(mae)

    train_loss['total'].append(np.array(loss_t['total']))
    train_loss['reg'].append(np.array(loss_t['reg']))
    train_loss['mae'].append(np.array(loss_t['mae']))
    train_loss['maeR'].append(np.array(loss_t['maeR']))
    t1 = time.time()
    #print(f"Time: {t1-t0}")

    # validate by structure generation
    loss_v = {'total':[],'mae':[],'maeR':[]}
    with torch.no_grad():    
        for i,(G1,G2,labelxyz,keyidx,info,b) in enumerate(valid_generator):
            if b == 1: continue # hack!
            
            G1 = G1.to(device)
            G2 = G2.to(device)
            
            M = G2.ndata['x'].shape[0]
            #labelidx = torch.eye(M)[keyidx].to(device) # K x M
            labelidx = [torch.eye(n)[idx].to(device) for n,idx in zip(G2.batch_num_nodes(),keyidx)]
            labelxyz = labelxyz.to(device)
            realdata = (info[0]['datatype'] == 'real')

            Yrec,A = model( G1, G2, labelidx, denoise=realdata )

            if verbose:
                generate_pose(Yrec, keyidx, G2.ndata['x'].squeeze(), atms, epoch )
                report_attention( G1.ndata['x'].squeeze(), A, epoch )

            prefix = "VALID "+info[0]['name']
            loss,mae = custom_loss(prefix, Yrec, labelxyz ) #both are Kx3 coordinates
            
            loss_v['total'].append(loss.cpu().detach().numpy())
            mae = mae.cpu().detach().numpy()

            if realdata:
                loss_v['maeR'].append(mae)
            else:
                loss_v['mae'].append(mae)
            
    valid_loss['total'].append(np.array(loss_v['total']))
    valid_loss['mae'].append(np.array(loss_v['mae']))
    train_loss['maeR'].append(np.array(loss_t['maeR']))

    form = "Train/Valid: %3d %8.4f %8.4f %8.4f (%8.4f %8.4f / %8.4f %8.4f)"
    print(form%(epoch, float(np.mean(loss_t['total'])), float(np.mean(loss_v['total'])),
                float(np.mean(loss_t['reg'])),
                float(np.mean(loss_t['mae'])), float(np.mean(loss_v['mae'])),
                float(np.mean(loss_t['maeR'])), float(np.mean(loss_v['maeR'])),
    )
          )
    
    if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss}, join("models", modelname, "best.pkl"))
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss}, join("models", modelname, "model.pkl"))
   #print("Train/Valid: %3d"%epoch, float(np.mean(loss_t)), float(np.mean(loss_v)),
    #float(np.mean(mae_t)), float(np.mean(mae_v)))
