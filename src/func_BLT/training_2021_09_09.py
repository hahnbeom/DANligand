#!/usr/bin/env python
import sys
import os

from scipy.spatial.transform import Rotation

import numpy as np
import torch

import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from src.myutils import *
from src.dataset import *
from src.model_multi import SE3Transformer
import src.motif as motif
# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

HYPERPARAMS = {
    "modelname" : sys.argv[1], #"XGrepro2",
    "base_learning_rate" : 1.0e-3, #dflt 1e-3
    'num_layers': (1,2,2),
    "w_reg"     : 1.0e-6, # loss ~0.05~0.1
    "max_epochs": 100,
    "w_lossBin"   : 1.0, #motif or not
    "w_lossCat"   : 1.0, #which category
    "w_lossxyz"   : 1.0, #MSE
    "w_lossrot"   : 0.0, #MSE

    # misc options below
    "modeltype" : 'comm',
    "gradient_accum_step" : 10,
    'nchannels' : 32, #default 32
    'use_l1'    : 0,
    'setsuffix' : 'v5or',
    'ansidx'   : list(range(0,14)), #[int(word) for word in sys.argv[2].split(',')], #-1 for all-type category prediction
    'learn_OR'  : True,
    "transfer"  : False, #transfer learning starting from "start.pkl"
    'clip_grad' : -1.0, #set < 0 if don't want
}


# default setup
set_params = {
    # IMPORTANT OPTIONS:
    "randomize_lig": 1.0, # Ang, pert the motif coord!
    'root_dir'     : "/projects/ml/ligands/motif/backbone/", #let each set get their own...
    'xyz_as_bb'    : True, #request to predict backbone not the motif

    # default below
    'ball_radius'  : 12.0,
    'ballmode'     : 'all',
    'sasa_method'  : 'sasa',
    'edgemode'     : 'distT',
    'edgek'        : (0,0),
    'edgedist'     : (10.0,6.0),
    'distance_feat': 'std',
    "randomize"    : 0.2, # Ang, pert the rest
    "CBonly"       : False,
    'debug'        : ('-debug' in sys.argv),
    }

# # Instantiating a dataloader
generator_params = {
    'shuffle': True, #True,
    'num_workers': 2,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 1,
}
if set_params['debug']: generator_params['num_workers'] = 0

def load_model(silent=False):
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    nchannels = HYPERPARAMS['nchannels']

    outtype = 'category'
    if isinstance(HYPERPARAMS['ansidx'],list):
        outtype = HYPERPARAMS['ansidx'] #extension of binary

    # l0 features dropped -- "is_lig"
    print("loading model:")
    model = SE3Transformer(
        num_layers     = HYPERPARAMS['num_layers'],
        l0_in_features = (65+N_AATYPE+2, N_AATYPE+1, nchannels+nchannels), #no aa-type in atm graph
        l1_in_features = (0,0,HYPERPARAMS['use_l1']),
        num_channels   = (nchannels,nchannels,nchannels),
        modeltype      = HYPERPARAMS['modeltype'],
        #nntypes        = ('SE3T','SE3T','SE3T'),
        nntypes        = ('TFN','TFN','TFN'), # TODO: switch back
        outtype        = outtype,
        drop_out       = 0.,
        learn_orientation = HYPERPARAMS['learn_OR']
    )
    print("finished loading model")

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

def is_same_simpleidx(label,idxs):
    #print(label,idx)
    return np.array([[float(motif.SIMPLEMOTIFIDX[i]==motif.SIMPLEMOTIFIDX[j]) for i in idxs] for j in label])


def rotate_example(G_bnd, G_atm, G_res, R, offset):
    """rotate_example updates the graphs corresponding to a rotation of the
    global reference frame.

    The underlying graphs are modified.

    Args:
        G_bnd, G_atm, G_res: bond, atom and residue graphs
        R, offset: rotation and translation of the global frame of shapes [3,3]
            and [3]

    Returns:
        updated graphs
    """
    G_bnd.edata['d'] = torch.einsum('kj,ij->ki', G_bnd.edata['d'], R)
    G_atm.edata['d'] = torch.einsum('kj,ij->ki', G_atm.edata['d'], R)
    G_res.edata['d'] = torch.einsum('kj,ij->ki', G_res.edata['d'], R)

    G_atm.ndata['x'] += offset[None, None]
    G_res.ndata['x'] += offset[None, None]

    # TODO MAKE THIS NOT DUBM
    G_atm.ndata['x'] *= 0.
    G_res.ndata['x'] *= 0.
    return G_bnd, G_atm, G_res

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
        expected_nout = len(motif.MOTIFS)


    # pull out first datapoint from generator.
    #for i, (G_bnd, G_atm, G_res, info) in enumerate(generator):
    for G_bnd, G_atm, G_res, info in generator:
        #import ipdb; ipdb.set_trace()
        if not G_bnd:
            print("skip ", info['pname'],info['sname'])
            continue

        # Get prediction and target value
        r2a = info['r2a'].to(device)
        motifidx   = info['motifidx'].long() # label of the true functional group

        Ps_cat = is_same_simpleidx(motifidx,ansidx)
        Ps_cat = torch.transpose(torch.tensor(Ps_cat).repeat(2,1),0,1)
        Ps_cat[:,0] = 1-Ps_cat[:,1]

        Ps_bin = [[float(idx==key) for key in ansidx] for idx in motifidx]
        Ps_bin = torch.transpose(torch.tensor(Ps_bin).repeat(2,1),0,1)
        Ps_bin[:,0] = 1-Ps_bin[:,1]

        Ps_cat = Ps_cat.float().to(device)
        Ps_bin = Ps_bin.float().to(device)

        # simple output on motif type
        Ps, dxyz_pred, rot_pred = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), r2a)

        def MyLLloss(P,Q): #Q prediction, P answer
            loss = torch.mean(-P*torch.log(Q)) #mean would better have normalization across setups?
            return loss

        loss1,loss2,loss3,loss4 = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)

        Ps = Ps.to(device)
        loss1 = MyLLloss(Ps_cat,Ps)
        loss2 = MyLLloss(Ps_bin,Ps)

        form1 = " | %-15s %2d"+' |'
        l = form1%tuple([info['sname'][0],int(info['motifidx'])])
        Ps = [int(P[1]) for P in Ps_cat]
        l += ' %1d'*len(Ps)%tuple(Ps)
        Ps = [int(P[1]) for P in Ps_bin]
        l += ' %1d'*len(Ps)%tuple(Ps)

        #try:
        if w_loss[2] > 0 or w_loss[3] > 0:
            LossMSE = torch.nn.MSELoss()

            dxyz = info['dxyz'].to(device)
            rot  = info['rot'].to(device) #should be symmetry aware; shape=[n,4] where n={1,2,3}
            dxyz_pred = dxyz_pred.to(device)

            loss3 = torch.tensor(0.0)
            loss4 = torch.tensor(0.0)
            if int(motifidx) > 0:
                loss3 = LossMSE(dxyz_pred[motifidx][None],dxyz)
            rot_pred = Rotation.from_matrix(rot_pred[motifidx].cpu().detach().numpy()).as_quat()
            rot_pred = torch.tensor(rot_pred, dtype=torch.float).to(device)
            loss4 = motif.LossRot(rot,rot_pred) # custom loss

    return temp_loss

def main():
    decay = 0.98
    max_epochs = HYPERPARAMS['max_epochs']
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']

    start_epoch,model,optimizer,train_loss,valid_loss = load_model()

    generators = load_dataset(set_params, generator_params, setsuffix=HYPERPARAMS['setsuffix'])
    train_generator,valid_generator = generators[:2]

    w = (HYPERPARAMS['w_lossCat'],HYPERPARAMS['w_lossBin'],
         HYPERPARAMS['w_lossxyz'],HYPERPARAMS['w_lossrot'])

    for epoch in range(start_epoch, max_epochs):
        print("Starting Epoch ", epoch)
        lr = base_learning_rate*np.power(decay, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        # Go through samples (batch size 1)
        # training
        header = "Epoch(%s): [%2d/%2d] QA"%(modelname, epoch, max_epochs)

        temp_loss = {}
        temp_loss = enumerate_an_epoch(model, optimizer, train_generator,
                                       w, temp_loss,
                                       is_training=True, header=header)

        for key in ['Bin','Cat','xyz','rot','reg']: train_loss[key].append(temp_loss[key])
        train_loss['total'].append(temp_loss['total'])

        # validation
        with torch.no_grad(): # without tracking gradients
            temp_loss = {}
            for i in range(3): # repeat multiple times for stable numbers
                temp_loss = enumerate_an_epoch(model, optimizer, valid_generator,
                                               w, temp_loss, is_training=False)

            for key in ['Bin','Cat','xyz','rot']: valid_loss[key].append(temp_loss[key])
            valid_loss['total'].append(temp_loss['total'])

        # Update the best model if necessary:
        if epoch == 0 or (np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1])):
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

if __name__ == "__main__":
    main()
