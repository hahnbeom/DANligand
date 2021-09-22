#!/usr/bin/env python
import sys
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from deepAccNet_XG.utilsXG import *
from deepAccNet_XG.dataset_combo import *
from deepAccNet_XG.model_aff import SE3Transformer

# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

HYPERPARAMS = {
    "modelname" : 'trsf_combo1', #"XGrepro2",
    "transfer"   : True, #transfer learning starting from "start.pkl"
    "base_learning_rate" : 1.0e-3,
    "gradient_accum_step" : 10,
    "max_epochs": 300,
    "w_lossG"   : 0.5, #global
    "w_lossL"   : 0.5, #per-atm
    "w_lossDG1" : 0.01,
    "w_lossDG2" : 0.01*0.2,
    "w_lossVS"  : 0.0, #BCE classification 
    "w_reg"     : 1.0e-5, # loss ~0.05~0.1
    "randomize" : 0.0, # Ang
    "modeltype" : 'comm',
    'num_layers': (2,4,4),
    'use_l1'    : 1,
    'setsuffix': 'v6c', 
    'hfinal_from': (1,1), #skip-connection, ligres
    'se3_on_energy': 1,
    'dGlayer'   : 'full', #['old','ligandonly','full']
    'clip_grad' : 1.0, #set < 0 if don't want
    #'hfinal_from': (int(sys.argv[2]),int(sys.argv[3])), #skip-connection, ligres
    # only for VS
}

# default setup
set_params = {
    'root_dir'     : "/projects/ml/ligands/v6c/", #contains combo
    'ball_radius'  : 9.0,
    'ballmode'     : 'all',
    'upsample'     : sampleDGonly,
    'sasa_method'  : 'sasa',
    'bndgraph_type': 'bonded',
    'edgemode'     : 'distT',
    'edgek'        : (0,0),
    'edgedist'     : (8.0,4.5),
    'distance_feat': 'std',
    'aa_as_het'    : True,
    'more_resfeatures' : True, # 3 -> 9
    'affinity_digits':np.arange(0,12.1,2.0),
    'nsamples_per_p': 1.0,
    'sample_mode'  : 'serial',
    'affinity_info': None, # turn off for prediction
    'debug'        : ('-debug' in sys.argv),
    }

# # Instantiating a dataloader
generator_params = {
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 1,
}
if set_params['debug']: generator_params['num_workers'] = 1
NRESFEATURE = N_AATYPE+2
if set_params['more_resfeatures']: NRESFEATURE += 6

def load_model(silent=False):
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    transfer = HYPERPARAMS['transfer']

    model = SE3Transformer(
        num_layers     = HYPERPARAMS['num_layers'], 
        l0_in_features = (65+N_AATYPE+3,NRESFEATURE, #[islig,aa1hot,sasa]+[netq,nchi,natm,kappa1,kappa2,FlexID]
                          32+32),
        l1_in_features = (0,0,HYPERPARAMS['use_l1']),
        hfinal_from    = HYPERPARAMS['hfinal_from'], 
        modeltype      = HYPERPARAMS['modeltype'],
        nntypes        = ('SE3T','SE3T','SE3T'),
        se3_on_energy  = HYPERPARAMS['se3_on_energy'],
        dGlayer        = HYPERPARAMS['dGlayer'],
    )
    
    model.to(device)

    modelpath = '/'.join(os.path.abspath(__file__).split('/')[:-1])+"/models/"+modelname
    if os.path.exists('%s/best.pkl'%(modelpath)):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(modelpath+'/best.pkl')
        model.load_state_dict(checkpoint["model_state_dict"])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        if not silent: print("Restarting at epoch", epoch)
        
    else:
        sys.exit("no existing model file!")
        
    return model

def enumerate_an_epoch(model, optimizer, generator):
    
    # affinity related
    MAX_DG = max(set_params['affinity_digits'])
    N_DG_BINS = len(set_params['affinity_digits'])
    DG_BINSIZE = 2.0 #?
    
    # > 4 as binder else non-binder
    binarykernel = torch.zeros([N_DG_BINS,2]).to(device)
    binarykernel[:2,0] = binarykernel[2:,1] = 1.0

    for i, (G_bnd, G_atm, G_res, info) in enumerate(generator): 
        # Get prediction and target value
        if isinstance(info,dict): info = [info]
        
        if not G_bnd:
            print("skip %s %s"%(info[0]['pname'],info[0]['sname']))
            continue

        idx = {}
        idx['ligidx'] = info[0]['ligidx'].to(device)
        idx['r2a'] = info[0]['r2amap'].to(device)
        idx['repsatm_idx'] = info[0]['repsatm_idx'].to(device)

        if 'dG' in info[0] and 'fnat' in info[0]:
            fnat = info[0]['fnat'].to(device)
            lddt = info[0]['lddt'].to(device)[None,:]
            fnatlogistic = 1.0/(1.0+torch.exp(-20*(fnat-0.5)))
            dG   = max(1.5,min(info[0]['dG'],MAX_DG)*fnatlogistic)
            dG = torch.tensor(dG).float().to(device)
            idG = torch.tensor([int(dG/DG_BINSIZE)]).to(device)
        else:
            fnat,dG,idG = 0.0,0.0,-1
        
        pred_fnat,pred_lddt,dg_logits = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), idx)

        if lddt.size() != pred_lddt.size(): continue

        pred_fnat = pred_fnat.to(device)
        pred_lddt = pred_lddt.to(device)
        dg_logits = dg_logits.unsqueeze(0).to(device) #batch dimension
        
        #dg_logits: NBIN
        # map probability to idx0 for -logP < 4 & rest P to idx1
        catlogits = torch.nn.functional.softmax(torch.matmul(dg_logits,binarykernel),dim=-1) #binary class

        Ps = torch.nn.functional.softmax(dg_logits,dim=-1).squeeze(0)
        Ebins = torch.arange(DG_BINSIZE*0.5,MAX_DG+DG_BINSIZE*0.5+0.1,DG_BINSIZE).to(device)
        dGcalc = torch.sum(Ps*Ebins)
        Ps = np.array(torch.nn.functional.softmax(dg_logits,dim=-1).cpu().detach().numpy())

        l = "%-30s : "%(info[0]['pname'])+\
            "%6.2f | %6.2f %6.2f %4d"%(pred_fnat, dG, dGcalc, idG)+" %5.3f"*len(Ps[0])%tuple(Ps[0])+'\n'

        sys.stdout.write(l)
        if i == len(generator)-1: break

def main():
    modelname = sys.argv[1]
    HYPERPARAMS['modelname'] = modelname
    model = load_model()
    
    setsuffix = HYPERPARAMS['setsuffix']

    # copy SE3 cache...
    if not os.path.exists('cache'):
        path = '/'.join(os.path.abspath(__file__).split('/')[:-1])+"/cache"
        os.system('cp -r %s ./'%path)

    if '-test' in sys.argv:
        test_set = Dataset(np.load("data/test_combo.npy"), **set_params)
        generator = data.DataLoader(test_set,
                                    **generator_params)
    else: #input arg
        set_params['root_dir'] = os.getcwd()+'/'
        targets = [l[:-5].replace('.prop','').replace('.lig','') for l in open(sys.argv[2])]
        generator = Dataset(targets, **set_params)
        
    # validation
    with torch.no_grad(): # without tracking gradients
        temp_loss = enumerate_an_epoch(model, None, generator)

if __name__ == "__main__":
    main()
