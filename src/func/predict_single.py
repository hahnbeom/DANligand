#!/usr/bin/env python
import sys
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from src.myutils import *
from src.dataset import *
from src.model import SE3Transformer
# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

HYPERPARAMS = {
    "modelname" : sys.argv[1], #"XGrepro2",
    "transfer"   : False, #transfer learning starting from "start.pkl"
    "base_learning_rate" : 1e-5, #still too big?
    "gradient_accum_step" : 10,
    "max_epochs": 100,
    "w_lossBin"   : 1.0, #motif or not
    "w_lossCat"   : 1.0, #which category 
    "w_lossxyz"   : 0.0, #MSE
    "w_lossrot"   : 0.0, #MSE
    "w_reg"     : 1.0e-6, # loss ~0.05~0.1
    "modeltype" : 'comm',
    'num_layers': (1,2,2),
    'nchannels' : 32, #default 32
    'use_l1'    : 1,
    'nalt_train': 1,
    'setsuffix' : 'v5',
    'ansidx'   : int(sys.argv[2]), #-1 for all-type category prediction
    #'hfinal_from': (1,1), #skip-connection, ligres
    'clip_grad' : -1.0, #set < 0 if don't want
    #'hfinal_from': (int(sys.argv[2]),int(sys.argv[3])), #skip-connection, ligres
    # only for VS
}

# default setup
set_params = {
    'root_dir'     : "/projects/ml/ligands/motif/", #let each set get their own...
    'ball_radius'  : 12.0,
    'ballmode'     : 'all',
    'sasa_method'  : 'sasa',
    'edgemode'     : 'distT',
    'edgek'        : (0,0),
    'edgedist'     : (10.0,6.0), 
    'distance_feat': 'std',
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 1.0, # Ang, pert the motif coord!
    "CBonly"       : ('-CB' in sys.argv),
    #'aa_as_het'   : True,
    'debug'        : ('-debug' in sys.argv),
    }

# # Instantiating a dataloader
generator_params = {
    'shuffle': False, #True,
    'num_workers': 4,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 1,
}
if set_params['debug']: generator_params['num_workers'] = 1

def load_model(silent=False):
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    nchannels = HYPERPARAMS['nchannels']

    outtype = 'category'
    if HYPERPARAMS['ansidx'] > 0:
        outtype = 'binary'

    # l0 features dropped -- "is_lig"
    model = SE3Transformer(
        num_layers     = HYPERPARAMS['num_layers'], 
        l0_in_features = (65+N_AATYPE+2, N_AATYPE+1, nchannels+nchannels), #no aa-type in atm graph
        l1_in_features = (0,0,HYPERPARAMS['use_l1']),
        num_channels   = (nchannels,nchannels,nchannels),
        modeltype      = HYPERPARAMS['modeltype'],
        nntypes        = ('SE3T','SE3T','SE3T'),
        outtype        = outtype,
    )
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
    print("nparams: ", count_parameters(model))

    if os.path.exists('models/%s/best.pkl'%(modelname)):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(join("models", modelname, "best.pkl"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        if not silent: print("Restarting at epoch", epoch)
        
    else:
        if not silent: print("Training a new model")
        epoch = 0
        train_loss = {"total":[], "Cat":[], "Bin":[], "xyz":[], "rot":[], "reg":[]}
        valid_loss = {"total":[], "Cat":[], "Bin":[], "xyz":[], "rot":[]}
        if not isdir(join("models", modelname)):
            if not silent: print("Creating a new dir at", join("models", modelname))
            os.mkdir(join("models", modelname))
    
    return epoch, model, optimizer, train_loss, valid_loss

def is_same_simpleidx(labeled,topred):
    return (labeled==topred)

def enumerate_an_epoch(model, optimizer, generator,
                       w_loss, temp_loss, mode='QA',
                       is_training=True, header="",
                       test=True):

    b_count=0
    w_reg = HYPERPARAMS['w_reg']
    gradient_accum_step = HYPERPARAMS['gradient_accum_step']
    ansidx = HYPERPARAMS['ansidx']
        
    # > 0 as motif  
    if ansidx > 0: #take simpler
        expected_nout = 2
    else:
        expected_nout = len(myutils.MOTIFS)
    
    binarykernel = torch.zeros([expected_nout,2]).to(device)
    binarykernel[:1,0] = binarykernel[1:,1] = 1.0

    for i, (G_bnd, G_atm, G_res, info) in enumerate(generator):
        # Get prediction and target value
        if not G_bnd:
            print("skip ", info['pname'],info['sname'])
            continue

        r2a = info['r2a'].to(device)
        motifidx   = info['motifidx']
        hasmotif   = torch.tensor(float(motifidx==ansidx)).to(device).repeat(1,2)
        hasmotif[:,0] = 1-hasmotif[:,1]
        
        if ansidx > 0: #take simpler
            expected_nout = 2
            identical_simpleidx = is_same_simpleidx(myutils.SIMPLEMOTIFIDX[int(motifidx)],myutils.SIMPLEMOTIFIDX[ansidx])
            motifidx = torch.tensor([identical_simpleidx]).long().to(device)
        else:
            expected_nout = len(myutils.MOTIFS)
            motifidx = torch.tensor([motifidx]).long().to(device) #integer
                    
        # simple output on motif type 
        logits,dxyz_pred,rot_pred = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), r2a)
        if logits.shape[-1] != expected_nout: continue
        
        logits = logits.to(device) #batch dimension
        P = torch.nn.functional.softmax(logits)

        if test:
            form = "%-15s %2d %2d %2d %2d |  %6.3f %6.3f"
            print(form%(info['sname'][0], 
                        int(info['motifidx']), ansidx, #label, topredict
                        int(hasmotif[0,1]), identical_simpleidx, #iscorrectbinaryForBin, isroughtypeForCat
                        float(P[0,0]), float(P[0,1])))
        
def main():
    setsuffix = HYPERPARAMS['setsuffix']
    test_set = Dataset(np.load("data/test_proteins%s.npy"%setsuffix),
                      **set_params)

    test_generator = data.DataLoader(test_set,
                                     **generator_params)
    
    _,model,optimizer,_,_ = load_model()

    with torch.no_grad(): # without tracking gradients
        enumerate_an_epoch(model, None, test_generator, 
                           [0.0,0.0,0.0,0.0], {},
                           is_training=False, test=True)

if __name__ == "__main__":
    main()

