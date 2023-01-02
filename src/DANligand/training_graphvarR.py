#!/usr/bin/env python
import sys
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from deepAccNet_graph.utils import *
from deepAccNet_graph.dataset import *
from deepAccNet_graph.model import SE3Transformer

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

modelname = "flexGR3ctrl"

base_learning_rate = 1e-3
decay = 0.995

DEBUG = ('-debug' in sys.argv)

N_PABLOCKS = 1 #1
RANDOMIZE = 0.2 # Ang
w_reg   = 1.0e-6 # loss ~0.05~0.1
MODELTYPE  = 'simple'
N_L0       = 32+32 #same always ['simple','comm']
#N_L0       = 32+28+1 #full model
#if MODELTYPE == 'simple': N_L0 = 32+32 #from atm&res graph outputs

## let "std" + normalized be the default setup
N_L1       = 0
DISTFEAT   = "std" #[std/1hot]
if DISTFEAT == "1hot": N_EDGE = 10
else: N_EDGE = 2

BNDGRAPH_TYPE = 'bonded'
NUM_LAYERS = (2,0,4)
NNTYPES = ('SE3T','SE3T','SE3T') #
#NNTYPES = ('GCN','SE3T','SE3T') #G3d
variable_gcn = True #false
LOSSTYPE = "MSE"

## Node,Edge definitions
#BALLDIST,BALLMODE = (10.0,'com')
EDGEMODE,EDGEK,EDGEDIST = ('topk',(12,12),(0.0,0.0))
BALLDIST,BALLMODE = (9.0,'all') #slows down by ~20%
#EDGEMODE,EDGEK,EDGEDIST = ('distT',(0,0),(8.0,4.5)) #similar speed to topk=12
#EDGEMODE,EDGEK,EDGEDIST = ('mink',(8,8),(8.0,4.5)) #similar speed to topk=12

# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

# # Instantiating a dataloader
params_loader = {
          'shuffle': True,
          'num_workers': 4,
          'pin_memory': True,
          'collate_fn': collate,
          'batch_size': 1,
}
if DEBUG: params_loader['num_workers'] = 1
batchsize = params_loader['batch_size']

def upsample1(fnat):
    over06 = fnat>0.6
    over07 = fnat>0.7
    over08 = fnat>0.8
    p = over06 + over07 + over08 + 1.0 #weight of 1,2,3,4
    return p/np.sum(p)

train_set = Dataset(np.load("data/train_proteins5.npy"), 
                    root_dir="/projects/ml/ligands/v4/",
                    ball_radius=BALLDIST,
                    randomize=RANDOMIZE, tag_substr=['rigid','flex'],
                    upsample=upsample1,
                    sasa_method='sasa', bndgraph_type=BNDGRAPH_TYPE,
                    edgemode=EDGEMODE, edgek=EDGEK, edgedist=EDGEDIST,
                    ballmode=BALLMODE,
                    distance_feat=DISTFEAT,
                    debug=DEBUG)

train_generator = data.DataLoader(train_set,
                                  worker_init_fn=lambda _: np.random.seed(),
                                  **params_loader)

val_set = Dataset(np.load("data/valid_proteins5.npy"), 
                  root_dir="/projects/ml/ligands/v4/",
                  ball_radius=BALLDIST,
                  tag_substr=['rigid','flex'],
                  upsample=upsample1,
                  sasa_method='sasa', bndgraph_type=BNDGRAPH_TYPE,
                  edgemode=EDGEMODE, edgek=EDGEK, edgedist=EDGEDIST,
                  ballmode=BALLMODE,
                  distance_feat=DISTFEAT,
                  debug=DEBUG)

valid_generator = data.DataLoader(val_set,
                                  worker_init_fn=lambda _: np.random.seed(),
                                  **params_loader)

model = SE3Transformer(
    num_layers     = NUM_LAYERS, 
    l0_in_features = (65+28+2,28+1,N_L0),
    l1_in_features = (0,0,N_L1),  
    num_degrees    = 2,
    num_channels   = (32,32,32),
    edge_features  = (N_EDGE,N_EDGE,N_EDGE), #dispacement & (bnd, optional)
    div            = (2,2,2),
    n_heads        = (2,2,2),
    pooling        = "avg",
    chkpoint       = True,
    modeltype      = MODELTYPE,
    nntypes        = NNTYPES, #('TFN','SE3T','SE3T'),
    variable_gcn   = variable_gcn
)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
print("nparams: ", count_parameters(model))

c = 0
max_epochs = 300
silent = False
w_loss1 = 0.5 #global
w_loss2 = 0.5 #per-atm
gradient_accum_step = 10

epoch = 0
train_loss = {"total":[], "global":[], "local":[], "reg":[]}
valid_loss = {"total":[], "global":[], "local":[]}

if os.path.exists('models/%s/start.pkl'%(modelname)):
    if not silent: print("Restart from a checkpoint")
    checkpoint = torch.load(join("models", modelname, "start.pkl"))
    model.load_state_dict(checkpoint["model_state_dict"])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

elif os.path.exists('models/%s/best.pkl'%(modelname)):
    if not silent: print("Loading a checkpoint")
    checkpoint = torch.load(join("models", modelname, "best.pkl"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]+1
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    if not silent: print("Restarting at epoch", epoch)
    assert(len(train_loss["total"]) == epoch)
    assert(len(valid_loss["total"]) == epoch)
else:
    if not silent: print("Training a new model")
    if not isdir(join("models", modelname)):
        if not silent: print("Creating a new dir at", join("models", modelname))
        os.mkdir(join("models", modelname))
    
start_epoch = epoch
if LOSSTYPE == "MAE":
    Loss = torch.nn.L1Loss()
else:
    Loss = torch.nn.MSELoss()
    
for epoch in range(start_epoch, max_epochs):  
    lr = base_learning_rate*np.power(decay, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Go through samples (batch size 1)
    b_count=0
    temp_loss = {"total":[], "global":[], "local":[], "reg":[]}
    optimizer.zero_grad()
    for i, (G_bnd, G_atm, G_res, info) in enumerate(train_generator): # G: InputFeatures y: labels
        if not G_bnd:
            print("skip %s %s"%(info[0]['pname'],info[0]['sname']))
            continue
        
        idx = {}
        idx['ligidx'] = info[0]['ligidx'].to(device)
        idx['r2a'] = info[0]['r2amap'].to(device)
        idx['repsatm_idx'] = info[0]['repsatm_idx'].to(device)
        fnat = info[0]['fnat'].to(device)
        lddt = info[0]['lddt'].to(device)[None,:]
        
        pred_fnat,pred_lddt = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), idx)

        if lddt.size() != pred_lddt.size(): continue

        loss1 = Loss(pred_fnat, fnat)
        loss2 = Loss(pred_lddt, lddt)
        
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)

        loss = w_loss1*loss1 + w_loss2*loss2 + w_reg*l2_reg
       
        if not np.isnan(loss.cpu().detach().numpy()):
            loss.backward(retain_graph=True)
        else:
            print("nan loss encountered", prediction.float(), fnat) 

        temp_loss["global"].append(loss1.cpu().detach().numpy())
        temp_loss["local"].append(loss2.cpu().detach().numpy())
        temp_loss["reg"].append(l2_reg.cpu().detach().numpy())
        temp_loss["total"].append(loss.cpu().detach().numpy()) # append only
        
        c+=1
        b_count+=1
        if i%gradient_accum_step == gradient_accum_step-1:
            optimizer.step()
            optimizer.zero_grad()
            
        sys.stdout.write("\rEpoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %.2f"
                         %(modelname, epoch, max_epochs, b_count, len(train_generator), temp_loss["total"][-1]))
        
    train_loss["global"].append(np.array(temp_loss["global"]))
    train_loss["local"].append(np.array(temp_loss["local"]))
    train_loss["reg"].append(np.array(temp_loss["reg"]))
    train_loss["total"].append(np.array(temp_loss["total"]))
    
    b_count=0
    with torch.no_grad(): # without tracking gradients
         # Loop over validation 10 times to get stable evaluation
        temp_loss = {"total":[], "global":[], "local":[]}
        for i in range(5):
            for G_bnd, G_atm, G_res, info in valid_generator:
                if not G_bnd:
                    print("skip %s %s"%(info[0]['pname'],info[0]['sname']))
                    continue

                idx = {}
                idx['ligidx'] = info[0]['ligidx'].to(device)
                idx['r2a'] = info[0]['r2amap'].to(device)
                idx['repsatm_idx'] = info[0]['repsatm_idx'].to(device)
                fnat = info[0]['fnat'].to(device)
                lddt = info[0]['lddt'].to(device)[None,:]
                
                pred_fnat,pred_lddt = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), idx)
                
                if lddt.size() != pred_lddt.size(): continue

                loss1 = Loss(pred_fnat, fnat)
                loss2 = Loss(pred_lddt, lddt)
                loss =  w_loss1*loss1 + w_loss2*loss2
                
                temp_loss["global"].append(loss1.cpu().detach().numpy())
                temp_loss["local"].append(loss2.cpu().detach().numpy())
                temp_loss["total"].append(loss.cpu().detach().numpy())

                b_count+=1
            
        valid_loss["global"].append(np.array(temp_loss["global"]))
        valid_loss["local"].append(np.array(temp_loss["local"]))
        valid_loss["total"].append(np.array(temp_loss["total"]))
        
    # Update the best model if necessary:
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

