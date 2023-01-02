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
    "modelname" : sys.argv[1], #"XGrepro2",
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
    "randomize" : 0.2, # Ang
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
    'upsample'     : upsample1,
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

    if temp_loss == {}:
        temp_loss = {"total":[], "global":[], "local":[], "reg":[]}

    if w_loss[0] < 1e-6 and w_loss[1] < 1e-6:
        temp_loss = {'global':[0.0],'local':[0.0],'reg':[0.0],'total':[0.0]}
        return temp_loss
    
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
            print("skip %s %s"%(info[0]['pname'],info[0]['sname']))
            continue
            
        idx = {}
        idx['ligidx'] = info[0]['ligidx'].to(device)
        idx['r2a'] = info[0]['r2amap'].to(device)
        idx['repsatm_idx'] = info[0]['repsatm_idx'].to(device)
        fnat = info[0]['fnat'].to(device)
        lddt = info[0]['lddt'].to(device)[None,:]
        
        fnatlogistic = 1.0/(1.0+torch.exp(-20*(fnat-0.5)))
        
        pred_fnat,pred_lddt,dg_logits = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), idx)

        if lddt.size() != pred_lddt.size(): continue

        pred_fnat = pred_fnat.to(device)
        pred_lddt = pred_lddt.to(device)
        
        LossG = torch.nn.MSELoss()
        LossL = torch.nn.MSELoss()
        loss1 = LossG(pred_fnat, fnat.float())
        loss2 = LossL(pred_lddt, lddt.float())

        #print(" : fnat/pred: %8.3f %8.3f"%(float(fnat.float()), float(pred_fnat.float())))
        loss = w_loss[0]*loss1 + w_loss[1]*loss2

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
        temp_loss["global"].append(loss1.cpu().detach().numpy())
        temp_loss["local"].append(loss2.cpu().detach().numpy())
        temp_loss["total"].append(loss.cpu().detach().numpy()) # append only

        if header != "":
            sys.stdout.write("\r%s, Batch: [%2d/%2d], loss: %8.4f"%(header,b_count,len(generator),temp_loss["total"][-1]))
    return temp_loss
            
def main():
    decay = 0.99
    max_epochs = HYPERPARAMS['max_epochs']
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    
    start_epoch,model,optimizer,train_loss,valid_loss = load_model()

    generators = load_dataset(set_params, generator_params,
                              setsuffix=".rd",
                              randomize=HYPERPARAMS['randomize'])
    train_generator,valid_generator = generators
    
    w_QA = (HYPERPARAMS['w_lossG'],HYPERPARAMS['w_lossL'])
    w_VS = (HYPERPARAMS['w_lossVS'],0.0)
    w_DG = (HYPERPARAMS['w_lossDG1'],HYPERPARAMS['w_lossDG2'])
    nalt_train = HYPERPARAMS['nalt_train']
    
    for epoch in range(start_epoch, max_epochs):  
        lr = base_learning_rate*np.power(decay, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        # Go through samples (batch size 1)
        # training
        temp_lossQA, temp_lossDG, temp_lossVS1, temp_lossVS2 = {},{},{},{}

        for alt in range(nalt_train):
            # 1) let VS/DG go first in trsfer learning
            if w_VS[0] > 0.0:
                # DUD-E set 
                header = "Epoch(%s): [%2d/%2d] VS"%(modelname, epoch, max_epochs)
                temp_lossVS1 = enumerate_an_epoch(model, optimizer, trainVS_generator1, 
                                                  w_VS, temp_lossVS1, mode='VS',
                                                  is_training=True, header=header)

                # "cross-dock" set -- VS
                header = "Epoch(%s): [%2d/%2d] VScross "%(modelname, epoch, max_epochs)
                temp_lossVS2 = enumerate_an_epoch(model, optimizer, trainVS_generator2, 
                                                  w_VS, temp_lossVS2, mode='VS',
                                                  is_training=True, header=header)
            
                train_loss['VS'].append(temp_lossVS1['global']+temp_lossVS2['global'])

             # 2) deltaG
            if w_DG[0] > 0.0 or w_DG[1] > 0.0:
                header = "Epoch(%s): [%2d/%2d] dG "%(modelname, epoch, max_epochs)
                temp_lossDG = enumerate_an_epoch(model, optimizer, trainDG_generator, 
                                                 w_DG, temp_lossDG, mode='DG',
                                                 is_training=True, header=header)
                train_loss['DG'].append(temp_lossDG['global'])

            # 0) QA train
            header = "Epoch(%s): [%2d/%2d] QA"%(modelname, epoch, max_epochs)
            temp_lossQA = enumerate_an_epoch(model, optimizer, train_generator, 
                                             w_QA, temp_lossQA, 
                                             is_training=True, header=header)
            
        # end nalt
        for key in ['global','local','reg']:
            train_loss[key].append(temp_lossQA[key])

        # summ up total loss
        totalloss = temp_lossQA['total']#
        if 'total' in temp_lossVS1: totalloss +=  temp_lossVS1['total'] + temp_lossVS2['total']
        if 'total' in temp_lossDG: totalloss += temp_lossDG['total']
            
        train_loss['total'].append(totalloss) 
        # finish alt
        #print("Train loss: ", epoch, np.mean(train_loss['total'][-1]))

        # validation
        w_QAv = w_QA #(w_QA[0]+1e-4,w_QA[1]+1e-4)
        with torch.no_grad(): # without tracking gradients
            temp_lossQA, temp_lossVS, temp_lossDG = {},{},{}
            for i in range(3): # repeat multiple times for stable numbers
                #book-keeping
                temp_lossQA = enumerate_an_epoch(model, optimizer, valid_generator, 
                                                 w_QAv, temp_lossQA, 
                                                 is_training=False)

                if HYPERPARAMS['w_lossVS'] > 0:
                    temp_lossVS = enumerate_an_epoch(model, optimizer, validVS_generator, 
                                                     w_VS, temp_lossVS, mode='VS',
                                                     is_training=False)
                if HYPERPARAMS['w_lossDG1'] > 0 or HYPERPARAMS['w_lossDG2'] > 0:
                    temp_lossDG = enumerate_an_epoch(model, optimizer, validDG_generator, 
                                                     w_DG, temp_lossDG, mode='DG',
                                                     is_training=False)
                
            for key in ['global','local']:
                valid_loss[key].append(temp_lossQA[key])
                
            if HYPERPARAMS['w_lossVS'] > 0: valid_loss['VS'].append(temp_lossVS['global'])
            if HYPERPARAMS['w_lossDG1'] > 0 or HYPERPARAMS['w_lossDG2'] > 0: valid_loss['DG'].append(temp_lossDG['global'])
                
            totalloss = temp_lossQA['total'] 
            if 'total' in temp_lossVS: totalloss += temp_lossVS['total']
            if 'total' in temp_lossDG: totalloss += temp_lossDG['total']
                
            valid_loss['total'].append(totalloss)
            
            #print("ValidLoss:", np.mean(valid_loss['global']))

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

