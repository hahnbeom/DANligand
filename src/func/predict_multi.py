#!/usr/bin/env python
import sys
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from src.myutils import *
from src.dataset import *
from src.model_multi import SE3Transformer
# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

HYPERPARAMS = {
    "modelname" : None, #sys.argv[1], #"XGrepro2",
    "transfer"   : False, #transfer learning starting from "start.pkl"
    "base_learning_rate" : 1e-3, #still too big?
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
    'setsuffix' : 'v5nofake',
    'ansidx'   : list(range(1,14)), #[int(word) for word in sys.argv[2].split(',')], #-1 for all-type category prediction
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
    "randomize"    : 0.0, # Ang, pert the rest
    "randomize_lig": 0.0, # Ang, pert the motif coord!
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
    if isinstance(HYPERPARAMS['ansidx'],list):
        outtype = HYPERPARAMS['ansidx'] #extension of binary

    # l0 features dropped -- "is_lig"
    model = SE3Transformer(
        num_layers     = HYPERPARAMS['num_layers'], 
        l0_in_features = (65+N_AATYPE+2, N_AATYPE+1, nchannels+nchannels), #no aa-type in atm graph
        l1_in_features = (0,0,HYPERPARAMS['use_l1']),
        num_channels   = (nchannels,nchannels,nchannels),
        modeltype      = HYPERPARAMS['modeltype'],
        nntypes        = ('SE3T','SE3T','SE3T'),
        drop_out       = 0.0,
        outtype        = outtype,
    )
    
    model.to(device)

    modelpath = '/'.join(os.path.abspath(__file__).split('/')[:-1])+"/models"
    print(modelpath)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
    print("nparams: ", count_parameters(model))

    if os.path.exists('%s/%s/best.pkl'%(modelpath,modelname)):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(join(modelpath, modelname, "best.pkl"), map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        if not silent: print("Restarting at epoch", epoch)
        
    else:
        sys.exit("Model path not found!")
    
    return epoch, model, optimizer, train_loss, valid_loss

def is_same_simpleidx(label,idxs):
    #print(label,idx)
    return np.array([[float(SIMPLEMOTIFIDX[i]==SIMPLEMOTIFIDX[j]) for i in idxs] for j in label])

def enumerate_an_epoch(model, optimizer, generator,
                       w_loss, temp_loss, mode='QA',
                       is_training=True, header=""):

    if temp_loss == {}: temp_loss = {"total":[],"Cat":[],"Bin":[],'xyz':[],'rot':[],'reg':[]}

    b_count=0
    w_reg = HYPERPARAMS['w_reg']
    gradient_accum_step = HYPERPARAMS['gradient_accum_step']
    ansidx = HYPERPARAMS['ansidx']
        
    # > 0 as motif  
    if isinstance(ansidx,list): #take simpler
        expected_nout = len(ansidx)
    else:
        expected_nout = len(myutils.MOTIFS)
    
    for i, (G_bnd, G_atm, G_res, info) in enumerate(generator):
        # Get prediction and target value
        if not G_bnd:
            print("skip ", info['pname'],info['sname'])
            continue

        r2a = info['r2a'].to(device)
        # simple output on motif type 
        Ps,dxyz_pred,rot_pred = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), r2a)
        
        # go individually for a hack...
        if Ps.shape[1] != expected_nout: continue
        Ps = np.array(Ps[0,:,1].cpu())
        
        l = "%-10s"%info['sname']+" %8.3f %8.3f %8.3f"%tuple(info['xyz'])
        
        if 'motifidx' in info and info['motifidx'] != -1:
            motifidx   = info['motifidx'].long() #label

            Ps_cat = is_same_simpleidx(motifidx,ansidx)
            #Ps_cat = torch.transpose(torch.tensor(Ps_cat).repeat(2,1),0,1)
            #Ps_cat[:,0] = 1-Ps_cat[:,1]
        
            Ps_bin = np.array([[float(idx==key) for key in ansidx] for idx in motifidx])
            #Ps_bin = torch.transpose(torch.tensor(Ps_bin).repeat(2,1),0,1)
            #Ps_bin[:,0] = 1-Ps_bin[:,1]
            
            #Ps_cat = np.array(Ps_cat[:,1])
            #Ps_bin = np.array(Ps_bin[:,1])

            Pcorrect_cat = max(Ps*Ps_cat)
            Pcorrect_bin = np.dot(Ps,Ps_bin)/np.sum(Ps_bin)
            l += " %2d | %6.3f %6.3f"%(int(info['motifidx']),Pcorrect_bin,Pcorrect_cat)
            
        else:
            l += " %2d | %6.3f %6.3f"%(-1,-1.0,-1.0)
            
        l += ' %2d %6.3f |'%(np.argmax(Ps)+1,max(Ps))
        l += " %5.3f"*len(Ps)%tuple(Ps)
        print(l)
            
def main_test():
    setsuffix = HYPERPARAMS['setsuffix']
    test_set = Dataset(np.load("data/test_proteins%s.npy"%setsuffix),
                      **set_params)
    test_generator = data.DataLoader(test_set,
                                     **generator_params)
    
    _,model,optimizer,_,_ = load_model()
    
    # validation
    with torch.no_grad(): # without tracking gradients
        enumerate_an_epoch(model, None, test_generator, 
                           [0.0,0.0,0.0,0.0], {})
        
def main_input(targets,inputpath='./'):
    # targets should be in "{trg}.{i}"
    
    set_params['root_dir'] = inputpath

    # check whether files exist
    for trggrid in targets:
        trg,grid = trggrid.split('.')
        if not os.path.exists('%s/%s.lig.npz'%(inputpath,trg)) or \
           not os.path.exists('%s/%s.prop.npz'%(inputpath,trg)):
            sys.exit("File does not exist! %s/%s.[lig,prop].npz"%(inputpath,trg))
    
    inputs = Dataset(targets, **set_params) 
    _,model,optimizer,_,_ = load_model()
    
    # validation
    with torch.no_grad(): # without tracking gradients
        enumerate_an_epoch(model, None, inputs, 
                           [0.0,0.0,0.0,0.0], {})


if __name__ == "__main__":
    modelname = sys.argv[1]
    HYPERPARAMS['modelname'] = modelname
    
    if '-input' in sys.argv:
        gridnpz = sys.argv[2]
        trg = gridnpz.split('/')[-1].split('.')[0]
        if '/' in gridnpz:
            inputpath = gridnpz.split('/')[0]+'/'
        else:
            inputpath = './'
        targets = ['%s.%d'%(trg,i) for i,_ in enumerate(np.load(gridnpz)['xyz'])]

        main_input(targets,inputpath)

    else:
        main_test() #default test proteins, etc
        
