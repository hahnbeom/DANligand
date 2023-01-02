#!/usr/bin/env python
import sys
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from deepAccNet_XG.utilsXG import *
from deepAccNet_XG.datasetXG import *
from deepAccNet_XG.model import SE3Transformer
# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

HYPERPARAMS = {
    "modelname" : "XGreproES",
    "transfer"   : False, #transfer learning starting from "start.pkl"
    "base_learning_rate" : 1e-3, #still too big?
    "gradient_accum_step" : 10,
    "w_lossG"   : 0.5, #global
    "w_lossL"   : 0.5, #per-atm
    "w_reg"     : 0.00001, # loss ~0.05~0.1
    "f_cotrain" : (1.0,0.0,1.0), #QAtrain,VScross-train,QAvalid
    "randomize" : 0.2, # Ang
    "modeltype" : 'comm',
    'num_layers': (2,2,2),
    # only for VS
    "w_lossVS"  : 0.0, #0.1? #classification
    "VSlossfunc": "BCE", #[BCE,MSE]
}

# default setup
set_params = {
    #'root_dir'     : "/net/scratch/hpark/crossdock/features/",
    'root_dir'     : "/projects/ml/ligands/v5.ext/",
    'ball_radius'  : 9.0,
    'ballmode'     : 'all',
    'upsample'     : upsample1,
    'sasa_method'  : 'sasa',
    'bndgraph_type': 'bonded',
    'edgemode'     : 'distT',
    'edgek'        : (0,0),
    'edgedist'     : (8.0,4.5),
    'distance_feat': 'std',
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

print("Naatypes:", N_AATYPE)

def load_model(silent=False):
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    transfer = HYPERPARAMS['transfer']

    model = SE3Transformer(
        num_layers     = HYPERPARAMS['num_layers'], 
        l0_in_features = (65+N_AATYPE+3,N_AATYPE+2,32+32),
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
        #print(train_loss["total"], len(train_loss["total"]))
        #assert(len(train_loss["total"]) == epoch)
        #assert(len(valid_loss["total"]) == epoch)
        
    elif transfer and (os.path.exists('models/%s/start.pkl'%(modelname))):
        checkpoint = torch.load(join("models", modelname, "start.pkl"))
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = 0
        train_loss = {"total":[], "global":[], "local":[], "VS":[], "reg":[]}
        valid_loss = {"total":[], "global":[], "local":[], "VS":[]}
        
    else:
        if not silent: print("Training a new model")
        epoch = 0
        train_loss = {"total":[], "global":[], "local":[], "VS":[], "reg":[]}
        valid_loss = {"total":[], "global":[], "local":[], "VS":[]}
        if not isdir(join("models", modelname)):
            if not silent: print("Creating a new dir at", join("models", modelname))
            os.mkdir(join("models", modelname))
    
    return epoch, model, optimizer, train_loss, valid_loss

def enumerate_an_epoch(model, optimizer, generator,
                       kernels, lossfunc, w_loss, temp_loss,
                       is_training, header=""):

    if temp_loss == {}:
        temp_loss = {"total":[], "global":[], "local":[], "reg":[]}

    if w_loss[0] < 1e-6 and w_loss[1] < 1e-6:
        temp_loss = {'global':[0.0],'local':[0.0],'reg':[0.0],'total':[0.0]}
        return temp_loss
    
    b_count=0
    w_reg = HYPERPARAMS['w_reg']
    gradient_accum_step = HYPERPARAMS['gradient_accum_step']

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
        
        pred_fnat,pred_lddt = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), idx)

        if lddt.size() != pred_lddt.size(): continue

        pred_fnat = kernels[0](pred_fnat).to(device)
        pred_lddt = kernels[1](pred_lddt).to(device)

        if lossfunc == "MSE":
            LossG = torch.nn.MSELoss()
            LossL = torch.nn.MSELoss()
        elif lossfunc == "BCE":
            LossG = torch.nn.BCELoss()
            LossL = torch.nn.MSELoss()
            pred_fnat = pred_fnat.repeat(1,2)
            pred_fnat[:,1] = 1-pred_fnat[:,0]
            fnat = fnat.repeat(1,2)
            fnat[:,1] = 1-fnat[:,0]
            
        loss1 = LossG(pred_fnat, fnat.float())
        loss2 = LossL(pred_lddt, lddt.float())
        loss = w_loss[0]*loss1 + w_loss[1]*loss2

        #print("%-10s %6.3f %6.3f %6.3f %6.3f"%(info[0]['pname'],
        #                                       float(loss),float(loss1),float(fnat[0]),float(pred_fnat)))

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
                optimizer.step()
                optimizer.zero_grad()
        
        b_count+=1
        temp_loss["global"].append(loss1.cpu().detach().numpy())
        temp_loss["local"].append(loss2.cpu().detach().numpy())
        temp_loss["total"].append(loss.cpu().detach().numpy()) # append only

        if header != "":
            sys.stdout.write("\r%s, Batch: [%2d/%2d], loss: %.2f"%(header,b_count,len(generator),temp_loss["total"][-1]))
    return temp_loss
            
def main():
    decay = 0.995
    max_epochs = 300
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    
    start_epoch,model,optimizer,train_loss,valid_loss = load_model()

    generators = load_dataset(set_params, generator_params, setsuffix='v5ext',
                              f_cotrain=HYPERPARAMS['f_cotrain'],
                              randomize=HYPERPARAMS['randomize'])
    train_generator,valid_generator = generators[:2]
    trainVS_generator1,trainVS_generator2,validVS_generator = generators[2:]
    
    def identity(x):
        return x
    def logistic(x):
        return 1.0/(1.0+torch.exp(-20*(x-0.5)))
    
    kernels_QA = [identity,identity] #global/local
    kernels_VS = [logistic,identity] #global/local
    w_QA = (HYPERPARAMS['w_lossG'],HYPERPARAMS['w_lossL'])
    w_VS = (HYPERPARAMS['w_lossVS'],0.0)
    VSloss = HYPERPARAMS['VSlossfunc']
    
    for epoch in range(start_epoch, max_epochs):  
        lr = base_learning_rate*np.power(decay, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        # Go through samples (batch size 1)
        # training
        temp_lossQA, temp_lossVS1, temp_lossVS2 = {},{},{}

        # let VS go first in trsfer learning
        if HYPERPARAMS['w_lossVS'] > 0:
            header = "Epoch(%s): [%2d/%2d] VS"%(modelname, epoch, max_epochs)
            temp_lossVS1 = enumerate_an_epoch(model, optimizer, trainVS_generator1, kernels_VS,
                                              VSloss, w_VS, temp_lossVS1,
                                              is_training=True, header=header)
            
            header = "Epoch(%s): [%2d/%2d] VScross "%(modelname, epoch, max_epochs)
            temp_lossVS2 = enumerate_an_epoch(model, optimizer, trainVS_generator2, kernels_VS,
                                              VSloss, w_VS, temp_lossVS2,
                                              is_training=True, header=header)
            
            train_loss['VS'].append(temp_lossVS1['global']+temp_lossVS2['global'])
        
        header = "Epoch(%s): [%2d/%2d] QA"%(modelname, epoch, max_epochs)
        temp_lossQA = enumerate_an_epoch(model, optimizer, train_generator, kernels_QA,
                                         "MSE", w_QA, temp_lossQA,
                                         is_training=True, header=header)
            
        for key in ['global','local','reg']:
            train_loss[key].append(temp_lossQA[key])
        totalloss = temp_lossQA['total']
        if 'total' in temp_lossVS1: totalloss += temp_lossVS1['total']
        if 'total' in temp_lossVS2: totalloss += temp_lossVS2['total']
        train_loss['total'].append(totalloss) 

        # validation
        w_QAv = (w_QA[0]+1e-4,w_QA[1]+1e-4)
        with torch.no_grad(): # without tracking gradients
            temp_lossQA, temp_lossVS = {},{}
            for i in range(5): # repeat multiple times for stable numbers
                #book-keeping
                temp_lossQA = enumerate_an_epoch(model, optimizer, valid_generator, kernels_QA,
                                                 "MSE", w_QAv, temp_lossQA,
                                                 is_training=False)

                if HYPERPARAMS['w_lossVS'] > 0:
                    temp_lossVS = enumerate_an_epoch(model, optimizer, validVS_generator, kernels_VS,
                                                     VSloss, w_VS, temp_lossVS,
                                                     is_training=False)
                
            for key in ['global','local']:
                valid_loss[key].append(temp_lossQA[key])
            totalloss = temp_lossQA['total']
            if 'total' in temp_lossVS: totalloss += temp_lossVS['total']
                
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

