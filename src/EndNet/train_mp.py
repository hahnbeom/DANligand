#!/usr/bin/env python
import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
import torch
from torch.utils import data
import time

## DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

#from src import *
from src.model import SE3TransformerWrapper
from src.src_Grid.myutils import N_AATYPE, count_parameters, to_cuda
from src.dataset import collate, Dataset
import src.src_Grid.motif as motif

ntypes = 6 #simplified types #len(motif.MOTIFS)
'''
mtype = 0
if i in A_lig and i in D_lig: mtype = 1
elif i in A_lig and i not in D_lig: mtype = 2
elif i not in A_lig and i in D_lig: mtype = 3
elif i in H_lig: mtype = 4
elif i in R_lig: mtype = 5
'''

wGrid = 1.0
w_reg = 1e-10
LR = 1.0e-4
BATCH = 1
w_contrast = 0.2
w_false = 1.0 #1.0/6.0 # 1.0/ngroups null contribution
#w_false = 0.0

# default setup
set_params={
# 'root_dir'     : "/home/hpark/data/HmapMine/features.forGridNet/",
'root_dir' : "/ml/motifnet/GridNet.ligand/",
'ball_radius'  : 8.0,
#'edgedist' : (2.6,4.5), # grid: 26 neighs
'edgedist' : (2.6,4.5), # grid: 18 neighs -- exclude cube-edges
'edgemode'     : 'dist',
#"upsample"     : sample1,
"randomize"    : 0.2, # Ang, pert the rest
"randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
"CBonly"       : False,
"ntype"        : ntypes,
'debug'        : ('-debug' in sys.argv),
}

params_loader={
    'shuffle':False,
    'num_workers':5 if '-debug' not in sys.argv else 1,
    'pin_memory':True,
    'collate_fn':collate,
    'batch_size':BATCH if '-debug' not in sys.argv else 1}

max_epoch=101
retrain=False
silent=False
modelname=sys.argv[1]
accum=1
rank=0

## DDP functions

### load_params / making model,optimizer,loss,etc.
def load_params(rank):
    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    ## model
    model=SE3TransformerWrapper(num_layers=4,
                                l0_in_features=65+N_AATYPE+3,
                                num_edge_features=3, #1-hot bond type x 2, distance 
                                l0_out_features=32, #category only
                                #l1_out_features=n_l1out,
                                num_degrees=1,
                                num_channels=16,
                                ntypes=ntypes,
                                n_trigonometry_module_stack=2)
    model.to(device)
    print("Nparams:",count_parameters(model))

    ## loss
    train_loss={"total":[],"BCEc":[],"BCEg":[],"BCEr":[],"contrast":[],"reg":[], "struct":[]}
    valid_loss={"total":[],"BCEc":[],"BCEg":[],"BCEr":[],"contrast":[],"reg":[], "struct":[]}
    
    epoch=0
    ## optimizer
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)

    if not retrain and os.path.exists("models/%s/model.pkl"%modelname):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(join("models", modelname, "model.pkl"))

        # trsf GridNet
        trained_dict = {}
        for key in checkpoint["model_state_dict"]:
            if key.startswith("module."):
                if 'Cblock' in key or 'graph_modules.8' in key: continue # do not succeed the last linear layers now
                newkey = key.replace('module.','se3_Grid.')
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
            else:
                trained_dict[key] = checkpoint["model_state_dict"][key]

         #model.load_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(trained_dict, strict=False)

        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = 1 #checkpoint["epoch"]+1 #force restart
        #train_loss = checkpoint["train_loss"] # ignore
        #valid_loss = checkpoint["valid_loss"] # ignore

        if not silent: print("Restarting at epoch", epoch)
         #assert(len(train_loss["total"]) == epoch)
         #assert(len(valid_loss["total"]) == epoch)
        restoreModel = True

        ## freeze here?

        
    else:
        if not silent: print("Training a new model")
        train_loss = {"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[], "struct":[]}
        valid_loss = {"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[], "struct":[]}
    
        best_models = []
        if not isdir(join("models", modelname)):
            if not silent: print("Creating a new dir at", join("models", modelname))
            os.mkdir(join("models", modelname))

        # initialize here
        for i, (name, layer) in enumerate(model.named_modules()):
            if isinstance(layer,torch.nn.Linear) and \
                "WOblock" in name: # or 'Rblock' in name:
                layer.weight.data.fill_(0.0)

    print("Loaded")

    return model,optimizer,epoch,train_loss,valid_loss


### train_model
def train_model(rank,world_size,dumm):
    count=np.zeros(ntypes)
    gpu=rank%world_size
    dist.init_process_group(backend='gloo',world_size=world_size,rank=rank)

    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    torch.cuda.set_device(device)

    ## load_params
    model,optimizer,start_epoch,train_loss,valid_loss=load_params(rank)

    ## DDP model
    ddp_model=DDP(model,device_ids=[gpu],find_unused_parameters=False)

    ## data loader
    train_set = Dataset(np.load("data/trainlist.combo.npy"), **set_params)
    train_sampler=data.distributed.DistributedSampler(train_set,num_replicas=world_size,rank=rank)
    train_loader=data.DataLoader(train_set,sampler=train_sampler,**params_loader)

    valid_set=Dataset(np.load("data/validlist.combo.npy"),**set_params)
    valid_sampler=data.distributed.DistributedSampler(valid_set,num_replicas=world_size,rank=rank)
    valid_loader=data.DataLoader(valid_set,sampler=valid_sampler,**params_loader)

    ## iteration
    for epoch in range(start_epoch,max_epoch):
        ## train
        ddp_model.train()
        temp_loss=train_one_epoch(ddp_model,optimizer,train_loader,rank,epoch,True)
        for k in train_loss:
            train_loss[k].append(np.array(temp_loss[k]))
        optimizer.zero_grad()
        
        ## evaluate
        ddp_model.eval()
        temp_loss = train_one_epoch(ddp_model,optimizer,valid_loader,rank,epoch,False)
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
                'valid_loss': valid_loss}, join("models", modelname, "best.pkl"))
   
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss}, join("models", modelname, "model.pkl"))


### train_one_epoch
def train_one_epoch(ddp_model,optimizer,loader,rank,epoch,is_train):
    temp_loss={"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[], "struct":[]}
    b_count,e_count=0,0
    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")

    for i, args in enumerate(loader):
        if args == None:
            e_count += 1
            continue

        (Grec, Glig, labelxyz, keyidx, info, b, node, edge) = args
        if Grec == None:
            e_count += 1
            continue

        with torch.cuda.amp.autocast(enabled=False):
            with ddp_model.no_sync():
                t0 = time.time()
                labelidx = [torch.eye(n)[idx] for n,idx in zip(Glig.batch_num_nodes(),keyidx)]
                labelxyz=to_cuda(labelxyz,device)
                labelidx=to_cuda(labelidx, device)
                edge=to_cuda(edge, device)

                Yrec_s, z, pred = ddp_model(to_cuda(Grec, device), to_cuda(node, device),
                                            to_cuda(Glig, device), to_cuda(labelidx, device), to_cuda(edge, device))
        
                if Yrec_s.shape[1] != 4: continue ##debug -- why?

                ## 1. GridNet loss related
                pred = torch.sigmoid(pred) # Then convert to sigmoid (0~1)
                preds = [pred] # assume batchsize=1
                t1 = time.time()

                # labels/mask have different size & cannot be Tensored -- better way?
                labels   = [torch.tensor(v["labels"], dtype=torch.float32).to(device) for v in info]
                masks     = [torch.tensor(v["mask"], dtype=torch.float32).to(device) for v in info] # to float
                Gsize     = torch.tensor([v["numnode"] for v in info]).to(device)
                pnames   = [v["pname"] for v in info]
                grids     = [v["grids"] for v in info]
            
                # 1-1. GridNet main losses; c-category g-group r-reverse contrast-contrast
                lossGc,lossGg,lossGr,bygrid = MaskedBCE(labels,preds,masks,device)
                lossGr = w_false*lossGr
                lossGcontrast =  w_contrast*ContrastLoss(preds,masks,device) # make overal prediction low as possible

                ## 1-2. GridNet-regularize pre-sigmoid value |x|<4
                p_reg = torch.nn.functional.relu(torch.sum(pred*pred-25.0)) #safe enough?

                ## 2. TRnet loss starts here
                # 2-1. structural loss
                lossTs, mae = structure_loss(Yrec_s, labelxyz,device) #both are Kx3 coordinates
            
                ## 3. Full regularizer
                l2_reg, p_reg = torch.tensor(0.).to(device),torch.tensor(0.).to(device)
                if is_train:
                    p_reg = torch.nn.functional.relu(torch.sum(pred*pred-25.0))
                    for param in ddp_model.parameters(): l2_reg += torch.norm(param)
                
                ## final loss
                ## default loss
                loss = wGrid*(lossGc + lossGg + lossGr + lossGcontrast + w_reg*(l2_reg+p_reg)) + lossTs
                loss.to(device)
    
                #store as per-sample loss
                temp_loss["total"].append(loss.cpu().detach().numpy()) 
                temp_loss["BCEc"].append(lossGc.cpu().detach().numpy()) 
                temp_loss["BCEg"].append(lossGg.cpu().detach().numpy()) 
                temp_loss["BCEr"].append(lossGr.cpu().detach().numpy()) 
                temp_loss["contrast"].append(lossGcontrast.cpu().detach().numpy()) 
                temp_loss["struct"].append(lossTs.cpu().detach().numpy())
                temp_loss["reg"].append((p_reg+l2_reg).cpu().detach().numpy()) 
            
            # Only update after certain number of accululations.
            if is_train and (b_count+1)%accum == 0:
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()    
                optimizer.zero_grad()
                print("TRAIN Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %8.3f (%6.3f/%6.3f/%6.3f/%6.3f/%6.3f) error %3d, %s"
                      %(modelname, epoch, max_epoch, b_count, len(loader),
                        np.sum(temp_loss["total"][-1*accum:]),
                        np.sum(temp_loss["BCEc"][-1*accum:]),
                        np.sum(temp_loss["BCEg"][-1*accum:]),
                        np.sum(temp_loss["BCEr"][-1*accum:]),
                        # bygrid[0],bygrid[1],bygrid[2],
                        np.sum(temp_loss["contrast"][-1*accum:]),
                        np.sum(temp_loss["struct"][-1*accum:]),
                        e_count, pnames[0]
                  ))
             #print("Train Rank %d | Epoch | %2d | Batch | [%2d/%2d] | Loss | %.3f | error | %d"%(rank,epoch,b_count,len(loader),np.sum(temp_loss['total'][-1*accum:]),e_count))

            b_count += 1

    return temp_loss


### loss calculation functions
def grouped_label(label,device):
#   print("grouped_label",device)
    labelG = torch.zeros(label.shape).to(device)

    # iter over 6 groups
    for k in range(1,7):
        js = np.where(np.array(motif.SIMPLEMOTIFIDX)==k)[0]
        if len(js) > 0:
            a = torch.max(label[:,js],dim=1)[0]
            if max(a).float() > 0.0:
                for j in js: labelG[:,j] = a

    # normalize
    norm = torch.sum(labelG, dim=1)[:,None].repeat(1,ntypes)+1.0e-6
    labelG = labelG / norm
    
    return labelG
            
def MaskedBCE(labels,preds,masks,device):
    # Batch
    # device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    #   print("maskedbce",device)

    lossC = torch.tensor(0.0).to(device)
    lossG = torch.tensor(0.0).to(device)
    lossR = torch.tensor(0.0).to(device)

    # iter through batches (actually not)
    bygrid = [0.0, 0.0, 0.0]
    for label,mask,pred in zip(labels,masks,preds):
        # "T": ntypes; "N": ngrids
        # label: NxT
        # mask : N
        # pred : NxT
        ngrid = label.shape[0]
        
        Q = pred[-ngrid:]
        
        a = -label*torch.log(Q+1.0e-6) #-PlogQ
        # old -- labeled ones still has < 1.0 thus penalized
        #b = -(1.0-label)*torch.log((1.0-Q)+1.0e-6)
        ilabel = (label<0.001).float()
        
        #labelG = grouped_label(label,device) # no such thing in ligand
        #g = -labelG*torch.log(Q+1.0e-6) # on group-label
        #ilabelG = (labelG<0.001).float()

        # transformed iQ -- 0~0.5->1, drops to 0 as x = 0.5->1.0
        # allows less penalty if x is 0.0 ~ 0.5
        #iQt = -0.5*torch.tanh(5*Q-3.0)+1)
        iQt = 1.0-Q+1.0e-6
        b  = -ilabel*torch.log(iQt) #penalize if high

        # normalize by num grid points & label points
        norm = 1.0
        
        lossC += torch.sum(torch.matmul(mask, a))*norm
        #lossG += torch.sum(torch.matmul(mask, g))*norm
        lossR += torch.sum(torch.matmul(mask, b))*norm

        bygrid[0] += torch.mean(torch.matmul(mask, a)).float()
        #bygrid[1] += torch.mean(torch.matmul(mask, g)).float()
        bygrid[2] += torch.mean(torch.matmul(mask, b)).float()
        
        if set_params['debug']:
            print("Label/Mask/Ngrid/Norm: %.1f/%d/%d/%.1f"%(float(torch.sum(label)), int(torch.sum(mask)), ngrid, float(norm)))
    return lossC, lossG, lossR, bygrid

def ContrastLoss(preds,masks,device):
    loss = torch.tensor(0.0).to(device)
    for mask,pred in zip(masks,preds):
        imask = 1.0 - mask
        ngrid = mask.shape[0]
        psum = torch.sum(torch.matmul(imask,pred[-ngrid:]))/ngrid

        loss += psum
    return loss

def structure_loss(Yrec, Ylig,device,prefix=""):
    dY = Yrec-Ylig
    loss = torch.sum(dY*dY,dim=1)
    
    N = Yrec.shape[0]
    loss_sum = torch.sum(loss)/N
    meanD = torch.mean(torch.sqrt(loss))

    return loss_sum, meanD
###


## main
if __name__=="__main__":
    torch.cuda.empty_cache()
    mp.freeze_support()
    world_size=torch.cuda.device_count()
    print("Using %d GPUs.."%world_size)
    
    if ('MASTER_ADDR' not in os.environ):
        os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
    if ('MASTER_PORT' not in os.environ):
        os.environ['MASTER_PORT'] = '12319'

    os.system("touch GPU %d"%world_size)
    mp.spawn(train_model,args=(world_size,0),nprocs=world_size,join=True)
