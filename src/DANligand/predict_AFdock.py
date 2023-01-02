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
    "modelname" : 'AFdock2', #"XGrepro2",
    "transfer"   : True, #transfer learning starting from "start.pkl"
    "base_learning_rate" : 1.0e-4,
    "gradient_accum_step" : 10,
    "max_epochs": 100,
    "w_lossG"   : 0.5, #global
    "w_lossL"   : 0.5, #per-atm
    "w_lossDG1" : 0.0,
    "w_lossDG2" : 0.0,
    "w_lossVS"  : 0.0, #BCE classification 
    "w_reg"     : 1.0e-5, # loss ~0.05~0.1
    "f_cotrain" : (1.0, 0.0, 0.0, 0.0), #QA-,VS-,VScross-,DG-train #--train only 5% at every epoch w/ dG
    "randomize" : 0.0, # Ang
    "modeltype" : 'comm',
    'num_layers': (2,4,4),
    'use_l1'    : 1,
    'nalt_train': 1,
    'setsuffix': 'v6', 
    'hfinal_from': (1,1), #skip-connection, ligres
    'se3_on_energy': 0,
    'dGlayer'   : 'old',
    'clip_grad' : 1.0, #set < 0 if don't want
    #'hfinal_from': (int(sys.argv[2]),int(sys.argv[3])), #skip-connection, ligres
    # only for VS
}

# default setup
set_params = {
    'root_dir'     : "/projects/ml/ligands/AFdock/",
    'ball_radius'  : 9.0,
    'ballmode'     : 'all',
    'upsample'     : None,
    'sample_mode'  : 'serial',
    'sasa_method'  : 'sasa',
    'bndgraph_type': 'bonded',
    'edgemode'     : 'distT',
    'edgek'        : (0,0),
    'edgedist'     : (8.0,4.5),
    'distance_feat': 'std',
    'aa_as_het'    : True,
    'more_resfeatures' : False, # 3 -> 9
    'affinity_digits':np.arange(0,12.1,2.0),
    'debug'        : ('-debug' in sys.argv),
    }

# # Instantiating a dataloader
generator_params = {
    'shuffle': False,
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
        drop_out       = 0.0,
        se3_on_energy  = HYPERPARAMS['se3_on_energy'],
        dGlayer        = HYPERPARAMS['dGlayer'],
    )

    ## Release this part
    # freeze all energy-nonrelated layers
    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith('Gblock_enr') or name.startswith('Ublock') or name.startswith('Eblock'):
                pass
            else:
                param.requires_grad = False
            
    for name, param in model.named_parameters():
        if param.requires_grad: print( name )
    '''
            
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
    print("nparams: ", count_parameters(model))

    if os.path.exists('models/%s/best.pkl'%(modelname)):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(join("models", modelname, "best.pkl"))#, location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        if not silent: print("Restarting at epoch", epoch)
        #print(train_loss["total"], len(train_loss["total"]))
        #assert(len(train_loss["total"]) == epoch)
        #assert(len(valid_loss["total"]) == epoch)
        
    elif transfer and (os.path.exists('models/%s/start.pkl'%(modelname))):
        checkpoint = torch.load(join("models", modelname, "start.pkl"))#, location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = 0
        train_loss = {"total":[], "global":[], "local":[], "VS":[], "DG":[], "reg":[]}
        valid_loss = {"total":[], "global":[], "local":[], "VS":[], "DG":[]}
        
    else:
        if not silent: print("Training a new model")
        epoch = 0
        train_loss = {"total":[], "global":[], "local":[], "VS":[], "DG":[], "reg":[]}
        valid_loss = {"total":[], "global":[], "local":[], "VS":[], "DG":[]}
        if not isdir(join("models", modelname)):
            if not silent: print("Creating a new dir at", join("models", modelname))
            os.mkdir(join("models", modelname))
    
    return epoch, model, optimizer, train_loss, valid_loss

def enumerate_an_epoch(model, optimizer, generator,
                       w_loss, temp_loss, mode='QA',
                       is_training=True, header=""):

    b_count=0
    w_reg = HYPERPARAMS['w_reg']
    gradient_accum_step = HYPERPARAMS['gradient_accum_step']

    # affinity related
    MAX_DG = max(set_params['affinity_digits'])
    N_DG_BINS = len(set_params['affinity_digits'])
    DG_BINSIZE = 2.0 #?
    # > 4 as binder else non-binder
    binarykernel = torch.zeros([N_DG_BINS,2]).to(device)
    binarykernel[:2,0] = binarykernel[2:,1] = 1.0

    for i, (G_bnd, G_atm, G_res, info) in enumerate(generator):
        # Get prediction and target value
        if not G_bnd:
            print("skip %s %s"%(info['pname'],info['sname']))
            continue
            
        idx = {}
        idx['ligidx'] = info['ligidx'].to(device)
        idx['r2a'] = info['r2amap'].to(device)
        idx['repsatm_idx'] = info['repsatm_idx'].to(device)
        fnat = info['fnat'].to(device)
        lddt = info['lddt'].to(device)[None,:]
        
        fnatlogistic = 1.0/(1.0+torch.exp(-20*(fnat-0.5)))
        
        pred_fnat,pred_lddt,dg_logits = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), idx)

        if lddt.size() != pred_lddt.size(): continue

        delta = float(pred_fnat-fnat)
        print("%s %s : fnat/pred: %8.3f %8.3f %8.3f"%(info['pname'],info['sname'],float(fnat.float()), float(pred_fnat.float()),delta))

    return temp_loss
            
def main():
    decay = 0.99
    max_epochs = HYPERPARAMS['max_epochs']
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    
    start_epoch,model,optimizer,train_loss,valid_loss = load_model()

    #set_params['root_dir'] = '/proj'
    set_params['nsamples_per_p'] = 100
    targets = np.load('data/test.rd.npy')
    generator = Dataset(targets, **set_params)
    print(len(generator))
    
    with torch.no_grad(): # without tracking gradients
        #book-keeping
        temp_lossQA = enumerate_an_epoch(model, optimizer, generator, 
                                         0.0, {}, 
                                         is_training=False)

                
if __name__ == "__main__":
    main()

