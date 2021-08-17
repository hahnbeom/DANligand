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
    "base_learning_rate" : 1e-4, #still too big?
    "gradient_accum_step" : 10,
    "max_epochs": 100,
    "w_lossBin"   : 0.5, #metal or not
    "w_lossCat"   : 0.5, #which category
    "w_lossxyz"   : 0.0, #MSE
    "w_reg"     : 1.0e-6, # loss ~0.05~0.1
    "modeltype" : 'simple',
    'num_layers': (1,0,4),
    'nchannels' : 16, #default 32
    'use_l1'    : 1,
    'nalt_train': 1,
    'setsuffix': 'v5', 
    #'hfinal_from': (1,1), #skip-connection, ligres
    'clip_grad' : 1.0, #set < 0 if don't want
    #'hfinal_from': (int(sys.argv[2]),int(sys.argv[3])), #skip-connection, ligres
    # only for VS
}

# default setup
set_params = {
    'root_dir'     : "/projects/ml/ligands/metalfeatures/", #let each set get their own...
    'ball_radius'  : 12.0,
    'ballmode'     : 'all',
    #'upsample'     : upsample1,
    'sasa_method'  : 'sasa',
    #'bndgraph_type': 'bonded',
    'edgemode'     : 'distT',
    'edgek'        : (0,0),
    'edgedist'     : (8.0,float(sys.argv[2])),
    'distance_feat': 'std',
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 1.0, # Ang, pert the metal coord!
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

    # l0 features dropped -- "is_lig"
    model = SE3Transformer(
        num_layers     = HYPERPARAMS['num_layers'], 
        l0_in_features = (65+N_AATYPE+2,N_AATYPE+1,nchannels),
        l1_in_features = (0,0,HYPERPARAMS['use_l1']),
        num_channels   = (nchannels,nchannels,nchannels),
        modeltype      = HYPERPARAMS['modeltype'],
        nntypes        = ('SE3T','SE3T','SE3T'),
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
        train_loss = {"total":[], "Cat":[], "Bin":[], "xyz":[], "reg":[]}
        valid_loss = {"total":[], "Cat":[], "Bin":[], "xyz":[]}
        if not isdir(join("models", modelname)):
            if not silent: print("Creating a new dir at", join("models", modelname))
            os.mkdir(join("models", modelname))
    
    return epoch, model, optimizer, train_loss, valid_loss

def enumerate_an_epoch(model, optimizer, generator,
                       w_loss, temp_loss, mode='QA',
                       is_training=True, header=""):

    if temp_loss == {}: temp_loss = {"total":[],"Cat":[],"Bin":[],'xyz':[],'reg':[]}

    b_count=0
    w_reg = HYPERPARAMS['w_reg']
    gradient_accum_step = HYPERPARAMS['gradient_accum_step']

    # > 0 as metal  
    binarykernel = torch.zeros([NMETALS+1,2]).to(device)
    #binarykernel[:1,0] = 1.0 #P=1 for [0,0] 
    #binarykernel[1:,1] = 1.0 #P=1 for [1:,1]
    binarykernel[:1,1] = 1.0 #P=1 for [0, 1] 
    binarykernel[1:,0] = 1.0 #P=1 for [1:,0]

    for i, (G_bnd, G_atm, G_res, info) in enumerate(generator): 
        # Get prediction and target value
        if not G_bnd:
            #print("skip ", info['pname'],info['sname'])
            continue

        r2a = None #info['r2a'].to(device)
        metalidx   = info['metalidx']
        hasmetal   = torch.tensor(float(metalidx>0)).to(device).repeat(1,2)
        hasmetal[:,1] = 1-hasmetal[:,0]
        
        #hasmetal = torch.zeros((2,NMETALS+1))
        #if metalidx>0: hasmetal[:1,0] = hasmetal[1:,1] = 1.0
        #else: hasmetal[1:,0] = hasmetal[:1,1] = 1.0
        metalidx = torch.tensor([metalidx]).long().to(device) #integer
        
        # simple output on metal type 
        logits,dxyz_pred = model(G_bnd.to(device), G_atm.to(device), None, r2a)
        if len(logits.shape) < 2 or logits.shape[1] != NMETALS+1: continue

        logits = logits.to(device)
        #logits = logits.unsqueeze(0).to(device) #batch dimension

        LossCat = torch.nn.CrossEntropyLoss()
        loss1 = LossCat(logits, metalidx)
        
        LossBin = torch.nn.BCELoss()
        Ps = torch.nn.functional.softmax(logits,dim=-1).squeeze()
        binnedlogits = torch.nn.functional.softmax(torch.matmul(Ps,binarykernel),dim=-1)[None,:]

        loss2 = LossBin(binnedlogits, hasmetal)

        LossXYZ = torch.nn.MSELoss()
        # why this fail sometimes
        try:
        #if True:
            dxyz = info['dxyz'].to(device)
            dxyz_pred = dxyz_pred.to(device)
            if int(metalidx) > 0:
                loss3 = LossXYZ(dxyz_pred,dxyz)
            else:
                loss3 = torch.tensor(0.0)
            Ps = Ps.cpu().detach().numpy()

            #print(dxyz, dxyz_pred)
            form = " | %-8s %1d %1d %6.2f (%6.2f %6.2f %6.2f) | "

            l = form%(info['sname'][0], int(hasmetal[0][0]), int(metalidx),
                      float(loss1+loss2+loss3),float(loss1),float(loss2),float(loss3))
            l += " %5.3f"*len(Ps)%tuple(Ps)
            print(l)
            
            loss = w_loss[0]*loss1 + w_loss[1]*loss2 + w_loss[2]*loss3

        #else:
        except:
            print("skip due to device issue")
            continue

        if is_training:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters(): l2_reg += torch.norm(param)
            loss = loss + w_reg*l2_reg

            if not np.isnan(loss.cpu().detach().numpy()):
                loss.backward(retain_graph=True)
            else:
                print("nan loss encountered", prediction.float(), fnat)
            temp_loss["reg"].append(l2_reg.cpu().detach().numpy())

            if i%gradient_accum_step == gradient_accum_step-1:
                if HYPERPARAMS['clip_grad'] > 0.0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), HYPERPARAMS['clip_grad'])

                optimizer.step()
                optimizer.zero_grad()
        
        b_count+=1
        temp_loss["Cat"].append(loss1.cpu().detach().numpy())
        temp_loss["Bin"].append(loss2.cpu().detach().numpy())
        temp_loss["xyz"].append(loss3.cpu().detach().numpy())
        temp_loss["total"].append(loss.cpu().detach().numpy()) # append only

        if header != "":
            sys.stdout.write("\r%s, Batch: [%2d/%2d], loss: %8.4f"%(header,b_count,len(generator),temp_loss["total"][-1]))
    return temp_loss
            
def main():
    decay = 0.98
    max_epochs = HYPERPARAMS['max_epochs']
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    
    start_epoch,model,optimizer,train_loss,valid_loss = load_model()

    generators = load_dataset(set_params, generator_params, setsuffix=HYPERPARAMS['setsuffix'])
    train_generator,valid_generator = generators[:2]
    
    w = (HYPERPARAMS['w_lossBin'],HYPERPARAMS['w_lossCat'],HYPERPARAMS['w_lossxyz'])
    
    for epoch in range(start_epoch, max_epochs):  
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
            
        for key in ['Bin','Cat','xyz','reg']: train_loss[key].append(temp_loss[key])
        train_loss['total'].append(temp_loss['total'])

        # validation
        with torch.no_grad(): # without tracking gradients
            temp_loss = {}
            for i in range(3): # repeat multiple times for stable numbers
                temp_loss = enumerate_an_epoch(model, optimizer, valid_generator, 
                                               w, temp_loss, is_training=False)

            for key in ['Bin','Cat','xyz']: valid_loss[key].append(temp_loss[key])
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

