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

modelname = "flexGR3comm4"
#modelname = "cmGR3comm"

base_learning_rate = 1e-3
decay = 0.995

DEBUG = ('-debug' in sys.argv)

N_PABLOCKS = 1 #1
RANDOMIZE = 0.0 # Ang
w_reg   = 1.0e-6 # loss ~0.05~0.1
MODELTYPE  = 'comm'
N_L0       = 32+32 #same always ['simple','comm']
#N_L0       = 32+28+1 #full model
#if MODELTYPE == 'simple': N_L0 = 32+32 #from atm&res graph outputs

## let "std" + normalized be the default setup
N_L1       = 0
DISTFEAT   = "std" #[std/1hot]
if DISTFEAT == "1hot": N_EDGE = 10
else: N_EDGE = 2

BNDGRAPH_TYPE = 'bonded'
NUM_LAYERS = (2,4,4)
NNTYPES = ('SE3T','SE3T','SE3T') #
#NNTYPES = ('GCN','SE3T','SE3T') #G3d
variable_gcn = True #false
LOSSTYPE = "MSE"
NORMQ = False

## Node,Edge definitions
#BALLDIST,BALLMODE = (10.0,'com')
#EDGEMODE,EDGEK = ('topk',(12,12),(0,0))
BALLDIST,BALLMODE = (9.0,'all') #slows down by ~20%
EDGEMODE,EDGEK,EDGEDIST = ('dist',(0,0),(8.0,4.5)) #similar speed to topk=12
#EDGEMODE,EDGEK,EDGEDIST = ('mink',(8,8),(8.0,4.5)) #similar speed to topk=12

# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

# # Instantiating a dataloader
params_loader = {
          'shuffle': False,
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
    variable_gcn   = variable_gcn,
    drop_out        = 0.0
)

model.to(device)

checkpoint = torch.load(join("models", modelname, "best.pkl"))
model.load_state_dict(checkpoint["model_state_dict"])

#trgs = [l[:-1] for l in open(sys.argv[1])]
trgs = np.load('data/valid_proteins5.npy')

with torch.no_grad(): # without tracking gradients
    # Loop over validation 10 times to get stable evaluation
    temp_loss = {"total":[], "global":[], "local":[]}

    val_set = Dataset(trgs,
                      #root_dir="/net/scratch/hpark/CMdock/features/",
                      root_dir="/projects/ml/ligands/v4.reps/",
                      ball_radius=BALLDIST,
                      tag_substr=['rigid','flex'], #['CM'],
                      sasa_method='sasa', bndgraph_type=BNDGRAPH_TYPE,
                      edgemode=EDGEMODE, edgek=EDGEK, edgedist=EDGEDIST,
                      ballmode=BALLMODE,
                      distance_feat=DISTFEAT,
                      nsamples_per_p=100,
                      sample_mode='serial')

    valid_generator = data.DataLoader(val_set,
                                      worker_init_fn=lambda _: np.random.seed(),
                                      **params_loader)
    pnames = []
    snames = {}
    for i, (G_bnd, G_atm, G_res, info) in enumerate(valid_generator):
        if not G_bnd:
            print("skip %s %s"%(info[0]['pname'],info[0]['sname']))
            continue
        pname = info[0]["pname"]
        sname = info[0]["sname"]
        pindex = info[0]["pindex"]
        if pname not in pnames:
            pnames.append(pname)
            snames[pname] = []

        if sname in snames[pname]: continue
        snames[pname].append(sname)
            
        idx = {}
        idx['ligidx'] = info[0]['ligidx'].to(device)
        idx['r2a'] = info[0]['r2amap'].to(device)
        idx['repsatm_idx'] = info[0]['repsatm_idx'].to(device)
        fnat = info[0]['fnat'].to(device)
        lddt = info[0]['lddt'].to(device)[None,:]
                
        pred_fnat,pred_lddt = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), idx)
                
        if pname not in pnames: pnames.append(pname)
        print("%4s %4d %15s %6.3f %6.3f %6.3f %3d"%(pname,pnames.index(pname),sname,
                                                    float(fnat),float(pred_fnat),
                                                    abs(float(fnat)-float(pred_fnat)),pindex))
 
