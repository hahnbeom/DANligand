#!/usr/bin/env python
import os, sys
import numpy as np
from os.path import join, isdir, isfile
import torch
import time

## DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model_former import EndtoEndModel
from src.myutils import count_parameters, to_cuda
from src.dataset_screen_work import collate, DataSet
import src.Loss as Loss

from args2 import args_formerNR_scratch as args

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")

NTYPES = 6
ddp = True
silent = False

# default setup
set_params={
    'datapath' : "/ml/motifnet/",
    'ball_radius'  : 8.0,
    'edgedist'     : (2.2,4.5), # grid: 18 neighs -- exclude cube-edges
    'edgemode'     : 'topk',
    'edgek'        : (8,16),
    "randomize"    : 0.2, # Ang, pert the rest
    #"randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    "ntype"        : NTYPES,
    'debug'        : ('-debug' in sys.argv),
    'maxedge'      : 30000,
    'maxnode'      : 1800,
    'drop_H'       : True
}

params_loader={
    'shuffle': (not ddp), 
    'num_workers':5 if not args.debug else 1,
    'pin_memory':True,
    'collate_fn':collate,
    'batch_size':1 if not args.debug else 1}

if not ddp:
    rank = 0

### load_params / making model,optimizer,loss,etc.
def load_params(rank):
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    ## model
    model = EndtoEndModel(args)
    model.to(device)

    ## loss
    train_loss_empty={"total":[],"BCEc":[],"BCEg":[],"BCEr":[],"contrast":[],"reg":[], "struct":[], "mae":[], "spread":[], "Screen":[]}
    valid_loss_empty={"total":[],"BCEc":[],"BCEg":[],"BCEr":[],"contrast":[],"struct":[], "mae":[], "spread":[], "Screen":[]}
    
    epoch=0
    ## optimizer
    optimizer=torch.optim.Adam(model.parameters(),lr=args.LR)

    if os.path.exists("models/%s/model.pkl"%args.modelname):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(join("models", args.modelname, "model.pkl"),map_location=device)

        trained_dict = {}
        model_dict = model.state_dict()
        model_keys = list(model_dict.keys())
        
        for key in checkpoint["model_state_dict"]:
            if key.startswith("module."):
                newkey = key.replace('module.','se3_Grid.')
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
            else:
                if key in model_keys:
                    wts = checkpoint["model_state_dict"][key]
                    if wts.shape == model_dict[key].shape: # load only if has the same shape
                        trained_dict[key] = wts
                    else:
                        print("skip", key)

        nnew, nexist = 0,0
        for key in model_keys:
            if key not in trained_dict:
                nnew += 1
            else:
                nexist += 1
        if rank == 0: print("params", nnew, nexist)
        
        model.load_state_dict(trained_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint["epoch"]+1 
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        for key in train_loss_empty:
            if key not in train_loss: train_loss[key] = []
        for key in valid_loss_empty:
            if key not in valid_loss: valid_loss[key] = []
            
        if not silent: print("Restarting at epoch", epoch)
        
    else:
        if not silent: print("Training a new model")
        train_loss = train_loss_empty
        valid_loss = valid_loss_empty
    
        best_models = []
        if not isdir(join("models", args.modelname)):
            if not silent: print("Creating a new dir at", join("models", args.modelname))
            os.mkdir(join("models", args.modelname))

    # temporary
    # re-initialize class module
    if epoch == 0:
        for i, (name, layer) in enumerate(model.named_modules()):
            if isinstance(layer,torch.nn.Linear) and \
               ("class" in name or 'Xform' in name): #".wr" in name or '.wl' in name:
                if rank == 0:
                    print("reweight", name)
                #layer.weight.data.fill_(0.0)
                layer.weight.data *= 0.1

    if rank == 0:
        print("Nparams:",count_parameters(model))
        print("Loaded")

    return model,optimizer,epoch,train_loss,valid_loss

def load_data(txt, world_size, rank):
    from torch.utils import data

    print("loading datafile", txt)
    target_s = []
    ligands_s = []
    is_ligand_s = []
    for ln in open(txt,'r'):
        x = ln.strip().split()
        is_ligand = bool(x[0]) #1: PL, 0: PP
        target = x[1]
        liglist = x[2:]
        
        target_s.append(target)
        ligands_s.append(liglist)
        is_ligand_s.append(is_ligand)
        
    TN = int(len(target_s)*(5/6)) # 5:1 train/validation
    
    train_set = DataSet(target_s[:TN], is_ligand_s[:TN], ligands_s[:TN], **set_params)
    valid_set = DataSet(target_s[TN:], is_ligand_s[TN:], ligands_s[TN:], **set_params)

    if ddp:
        train_sampler = data.distributed.DistributedSampler(train_set,num_replicas=world_size,rank=rank)
        valid_sampler = data.distributed.DistributedSampler(valid_set,num_replicas=world_size,rank=rank)
        train_loader = data.DataLoader(train_set,sampler=train_sampler,**params_loader)
        valid_loader = data.DataLoader(valid_set,sampler=valid_sampler,**params_loader)
    else:
        train_loader = data.DataLoader(train_set, **params_loader)
        valid_loader = data.DataLoader(valid_set, **params_loader)
    return train_loader, valid_loader

### train_model
def train_model(rank,world_size,dumm):
    count=np.zeros(NTYPES)
    gpu=rank%world_size
    dist.init_process_group(backend='gloo',world_size=world_size,rank=rank)

    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    torch.cuda.set_device(device)

    ## load_params
    model,optimizer,start_epoch,train_loss,valid_loss=load_params(rank)

    if ddp:
        ddp_model=DDP(model,device_ids=[gpu],find_unused_parameters=False)

    ## data loader
    train_loader, valid_loader = load_data(args.datasetf, world_size, rank)

    ## iteration
    for epoch in range(start_epoch,args.max_epoch):
        ## train
        if ddp:
            ddp_model.train()
            temp_loss=train_one_epoch(ddp_model,optimizer,train_loader,rank,epoch,True)
        else:
            model.train()
            temp_loss=train_one_epoch(model,optimizer,train_loader,rank,epoch,True)
            
        for k in train_loss:
            train_loss[k].append(np.array(temp_loss[k]))
        optimizer.zero_grad()
        
        ## evaluate
        with torch.no_grad():
            if ddp:
                ddp_model.eval()
                temp_loss = train_one_epoch(ddp_model,optimizer,valid_loader,rank,epoch,False)
            else:
                temp_loss = train_one_epoch(model,optimizer,valid_loader,rank,epoch,False)
        
            for k in valid_loss:
                valid_loss[k].append(np.array(temp_loss[k]))

        print("***SUM***")
        print("Train loss | %7.4f | Valid loss | %7.4f"%((np.mean(train_loss['total'][-1]),np.mean(valid_loss['total'][-1]))))

        ## update the best model
        if rank==0:
            if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss}, join("models", args.modelname, "best.pkl"))
   
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss}, join("models", args.modelname, "model.pkl"))


### train_one_epoch
def train_one_epoch(model,optimizer,loader,rank,epoch,is_train):
    temp_loss={"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[], "struct":[], "mae":[], "spread":[], "Screen":[]}
    b_count,e_count=0,0
    accum=1
    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")

    for i, inputs in enumerate(loader):
        if inputs == None:
            e_count += 1
            continue

        (Grec, Glig, cats, masks, keyxyz, keyidx, blabel, info) = inputs
        if Grec == None:
            e_count += 1
            continue

        with torch.cuda.amp.autocast(enabled=False):
            with model.no_sync(): #should be commented if
            #if True:
                t0 = time.time()
                
                if Glig != None:
                    Glig = to_cuda(Glig,device)
                    keyxyz = to_cuda(keyxyz, device)
                    keyidx = to_cuda(keyidx, device)
                    nK = info['nK'].to(device)
                    blabel = to_cuda(blabel, device)
                else:
                    keyxyz, keyidx, nK, blabel = None, None, None, None                    
                    
                Grec = to_cuda(Grec, device)
                pnames  = info["pname"]
                grid = info['grid'].to(device)
                eval_struct = info['eval_struct'][0]
                grididx = info['grididx'].to(device)

                # Ggrid memory check -- otherwise 'x' and 'nsize' is sufficient
                t1 = time.time()
                Yrec_s, z, MotifP, aff = model(Grec, 
                                               Glig, keyidx, grididx,
                                               gradient_checkpoint=is_train,
                                               drop_out=is_train)
                
                if MotifP == None:
                    continue
        
                ## 1. GridNet loss related
                lossGc, lossGg, lossGr = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                lossGcontrast = torch.tensor(0.0).to(device)
                p_reg = torch.tensor(0.).to(device)
                
                if cats != None:
                    cats = to_cuda(cats, device)
                    masks = to_cuda(masks, device)
                    
                    MotifP = torch.sigmoid(MotifP) # Then convert to sigmoid (0~1)
                    MotifPs = [MotifP] # assume batchsize=1
                    t1 = time.time()
                
                    # 1-1. GridNet main losses; c-category g-group r-reverse contrast-contrast
                    
                    lossGc,lossGg,lossGr,bygrid = Loss.MaskedBCE(cats,MotifPs,masks)
                    lossGr = args.w_false*lossGr
                    lossGcontrast = args.w_contrast*Loss.ContrastLoss(MotifPs,masks) # make overal prediction low as possible

                    p_reg = torch.nn.functional.relu(torch.sum(MotifP*MotifP-25.0))
                
                ## 2. TRnet loss starts here
                Pbind = [] #verbose
                lossTs, mae, lossTr = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                lossScreen = torch.tensor(0.0).to(device)

                if Yrec_s != None and grid.shape[1] == z.shape[1]:
                    try:
                    #if True:
                        # 2-1. structural loss
                        if eval_struct:
                            nK = nK.squeeze() # hack, take the first one alone
                            lossTs, mae = Loss.structural_loss( Yrec_s, keyxyz, nK ) #both are Kx3 coordinates

                            lossTr = args.w_spread*Loss.spread_loss( keyxyz, z, grid, nK )
                            #lossTr2 = args.w_spread*spread_loss2( keyxyz, z, grid, nK )
                    
                        # 2-2.s screening loss
                        lossScreen = args.w_screen*Loss.ScreeningLoss( aff, blabel )
                        Pbind = ['%4.2f'%float(a) for a in torch.sigmoid(aff)]
                    except:
                        pass
                    
                t2 = time.time()
                    
                ## 3. Full regularizer
                l2_reg = torch.tensor(0.).to(device)
                if is_train:
                    for param in model.parameters(): l2_reg += torch.norm(param)
                
                ## final loss
                ## default loss
                loss = args.wGrid*(lossGc + lossGg + lossGr + lossGcontrast + \
                                   args.w_reg*(l2_reg+p_reg))  \
                      + args.wTR*(lossTs + lossTr + lossScreen)
                
                #store as per-sample loss
                temp_loss["total"].append(loss.cpu().detach().numpy()) 
                temp_loss["BCEc"].append(lossGc.cpu().detach().numpy()) 
                temp_loss["BCEg"].append(lossGg.cpu().detach().numpy()) 
                temp_loss["BCEr"].append(lossGr.cpu().detach().numpy()) 
                temp_loss["contrast"].append(lossGcontrast.cpu().detach().numpy())
                temp_loss["reg"].append((p_reg+l2_reg).cpu().detach().numpy())
                if lossTs > 0.0:
                    temp_loss["struct"].append(lossTs.cpu().detach().numpy())
                    temp_loss["mae"].append(mae.cpu().detach().numpy())
                    temp_loss["spread"].append(lossTr.cpu().detach().numpy())
                if lossScreen > 0.0:
                    temp_loss["Screen"].append(lossScreen.cpu().detach().numpy())
            
            # Only update after certain number of accululations.
            if is_train and (b_count+1)%accum == 0:
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()    
                optimizer.zero_grad()
                print("Rank %d TRAIN Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %8.3f (M/G/R/C/S(mae)/Sp/Scr: %6.1f/%6.1f/%6.1f/%6.3f/%6.1f(%5.2f)/%6.3f/%6.2f), %s"
                      %(rank,args.modelname, epoch, args.max_epoch, b_count, len(loader),
                        np.sum(temp_loss["total"][-1*accum:]),
                        np.sum(temp_loss["BCEc"][-1*accum:]),
                        np.sum(temp_loss["BCEg"][-1*accum:]),
                        np.sum(temp_loss["BCEr"][-1*accum:]),
                        np.sum(temp_loss["contrast"][-1*accum:]),
                        float(lossTs),
                        float(mae),
                        float(lossTr),
                        np.sum(temp_loss["Screen"][-1*accum:]),
                        pnames[0]
                      ), ' '.join(Pbind))
            elif (b_count+1)%accum == 0: # valid
                print("Rank %d VALID Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %8.3f (M/G/R/C/S(mae)/Sp/Scr: %6.1f/%6.1f/%6.1f/%6.3f/%6.1f(%5.2f)/%6.3f/%6.2f), %s"
                      %(rank, args.modelname, epoch, args.max_epoch, b_count, len(loader),
                        np.sum(temp_loss["total"][-1*accum:]),
                        np.sum(temp_loss["BCEc"][-1*accum:]),
                        np.sum(temp_loss["BCEg"][-1*accum:]),
                        np.sum(temp_loss["BCEr"][-1*accum:]),
                        np.sum(temp_loss["contrast"][-1*accum:]),
                        float(lossTs),
                        float(mae),
                        float(lossTr),
                        np.sum(temp_loss["Screen"][-1*accum:]),
                        pnames[0]
                      ), ' '.join(Pbind))
                t3 = time.time()
                        
            b_count += 1

    return temp_loss

## main
if __name__=="__main__":
    torch.cuda.empty_cache()
    mp.freeze_support()
    world_size=torch.cuda.device_count()
    print("Using %d GPUs.."%world_size)
    
    if ('MASTER_ADDR' not in os.environ):
        os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
    if ('MASTER_PORT' not in os.environ):
        os.environ['MASTER_PORT'] = '12320'

    os.system("touch GPU %d"%world_size)

    if ddp:
        mp.spawn(train_model,args=(world_size,0),nprocs=world_size,join=True)
    else:
        train_model(0, 1, None)
