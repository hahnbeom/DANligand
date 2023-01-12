#!/usr/bin/env python
import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
import torch
from torch.utils import data
import time

#from src import *
from src.model import SE3TransformerWrapper
from src.src_Grid.myutils import N_AATYPE, count_parameters, to_cuda
from src.dataset import collate, Dataset
import src.src_Grid.motif as motif

ntypes=len(motif.MOTIFS)

w_reg = 1e-10
LR = 1.0e-4
BATCH = 1
w_contrast = 0.2
w_false = 1.0 #1.0/6.0 # 1.0/ngroups null contribution
#w_false = 0.0

model = SE3TransformerWrapper( num_layers=4,
                               l0_in_features=65+N_AATYPE+2,
                               num_edge_features=3, #1-hot bond type x 2, distance 
                               l0_out_features=32, #category only
                               #l1_out_features=n_l1out,
                               num_degrees=1,
                               num_channels=16,
                               ntypes=ntypes,
                               n_trigonometry_module_stack=2,
                               )

print("Nparams:", count_parameters(model))

params_loader = {
          'shuffle': False,
          'num_workers': 5 if '-debug' not in sys.argv else 1,
          'pin_memory': True,
          'collate_fn': collate,
          'batch_size': BATCH if '-debug' not in sys.argv else 1}

# default setup
set_params = {
    # 'root_dir'     : "/home/hpark/data/HmapMine/features.forGridNet/",
    'root_dir' : "/ml/motifnet/GridNet.ligand/",
    'ball_radius'  : 8.0,
    #'edgedist' : (2.6,4.5), # grid: 26 neighs
    'edgedist' : (2.2,4.0), # grid: 18 neighs -- exclude cube-edges 
    'edgemode'     : 'dist',
    #"upsample"     : sample1,
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    "CBonly"       : False,
    'debug'        : ('-debug' in sys.argv),
    }

#train_set = Dataset(np.load('data/GridNet.ligand/datalist.npy'), **set_params)
train_set = Dataset(np.load("data/GridNet.ligand/trainlist.npy"), **set_params)
train_loader = data.DataLoader(train_set,
                               worker_init_fn=lambda _: np.random.seed(),
                               **params_loader)

#valid_set = Dataset('data/GridNet.ligand/datalist.npy', **set_params)
valid_set = Dataset(np.load("data/GridNet.ligand/validlist.npy")[:1], **set_params)
valid_loader = data.DataLoader(valid_set,
                               worker_init_fn=lambda _: np.random.seed(),
                               **params_loader)

device      = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
optimizer   = torch.optim.Adam(model.parameters(), lr=LR)
max_epochs  = 101
accum       = 1
epoch       = 0
modelname   = sys.argv[1]
model.to(device)
retrain     = False
silent      = False

if not retrain and os.path.exists("models/%s/model.pkl"%modelname): 
    if not silent: print("Loading a checkpoint")
    checkpoint = torch.load(join("models", modelname, "model.pkl"))

    trained_dict = {}
    for key in checkpoint["model_state_dict"]:
        if key.startswith("module."):
            newkey = key[7:]
            trained_dict[newkey] = checkpoint["model_state_dict"][key]
        else:
            trained_dict[key] = checkpoint["model_state_dict"][key]
      
    #model.load_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(trained_dict)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]+1
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    if not silent: print("Restarting at epoch", epoch)
    #assert(len(train_loss["total"]) == epoch)
    #assert(len(valid_loss["total"]) == epoch)
    restoreModel = True
else:
    if not silent: print("Training a new model")
    epoch = 0
    train_loss = {"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[]}
    valid_loss = {"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[]}

    best_models = []
    if not isdir(join("models", modelname)):
        if not silent: print("Creating a new dir at", join("models", modelname))
        os.mkdir(join("models", modelname))

    # initialize here
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer,torch.nn.Linear) and \
           "WOblock" in name: # or 'Rblock' in name:
            layer.weight.data.fill_(0.0)

def grouped_label(label):
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
            
def MaskedBCE(labels,preds,masks):
    # Batch
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
        labelG = grouped_label(label)
        
        Q = pred[-ngrid:]
        
        a = -label*torch.log(Q+1.0e-6) #-PlogQ
        g = -labelG*torch.log(Q+1.0e-6) # on group-label
        # old -- labeled ones still has < 1.0 thus penalized
        #b = -(1.0-label)*torch.log((1.0-Q)+1.0e-6)

        # new: define ilabel == (labelG > 0.001)
        ilabel = (label<0.001).float()
        ilabelG = (labelG<0.001).float()

        # transformed iQ -- 0~0.5->1, drops to 0 as x = 0.5->1.0
        # allows less penalty if x is 0.0 ~ 0.5
        #iQt = -0.5*torch.tanh(5*Q-3.0)+1)
        iQt = 1.0-Q+1.0e-6
        b  = -(0.7*ilabel+0.3*ilabelG)*torch.log(iQt) #penalize if high

        # normalize by num grid points & label points
        norm = 1.0
        #norm = torch.sum(label)/torch.sqrt(torch.sum(mask))
        
        lossC += torch.sum(torch.matmul(mask, a))*norm
        lossG += torch.sum(torch.matmul(mask, g))*norm
        lossR += torch.sum(torch.matmul(mask, b))*norm

        bygrid[0] += torch.mean(torch.matmul(mask, a)).float()
        bygrid[1] += torch.mean(torch.matmul(mask, g)).float()
        bygrid[2] += torch.mean(torch.matmul(mask, b)).float()
        
        if set_params['debug']:
            print("Label/Mask/Ngrid/Norm: %.1f/%d/%d/%.1f"%(float(torch.sum(label)), int(torch.sum(mask)), ngrid, float(norm)))
    return lossC, lossG, lossR, bygrid

def ContrastLoss(preds,masks):
    loss = torch.tensor(0.0).to(device)
    for mask,pred in zip(masks,preds):
        imask = 1.0 - mask
        ngrid = mask.shape[0]
        psum = torch.sum(torch.matmul(imask,pred[-ngrid:]))/ngrid

        loss += psum
    return loss

def structure_loss(Yrec, Ylig, prefix=""):
    dY = Yrec-Ylig
    loss = torch.sum(dY*dY,dim=1)
    
    N = Yrec.shape[0]
    loss_sum = torch.sum(loss)/N
    meanD = torch.mean(torch.sqrt(loss))

    return loss_sum, meanD

start_epoch = epoch
count = np.zeros(ntypes)
for epoch in range(start_epoch, max_epochs):
    b_count,e_count=0,0
    temp_loss = {"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[]}
    for i, args in enumerate(train_loader):
        if args == None:
            e_count += 1
            continue
        
        (Grec, Glig, labelxyz, keyidx, info, b, node, edge) = args
        if Grec == None:
            e_count += 1
            continue
    
        t0 = time.time()
        
        labelidx = [torch.eye(n)[idx] for n,idx in zip(Glig.batch_num_nodes(),keyidx)]
        labelxyz = to_cuda( labelxyz, device )

        # predicted structure, Attention, category-for-GridNet
        Yrec_s, z, pred = model(to_cuda(Grec, device), to_cuda(node, device),
                                to_cuda(Glig, device), to_cuda(labelidx, device), to_cuda(edge, device))

        if Yrec_s.shape[1] != 4: continue ##debug -- why?

        ## 1. GridNet loss related -- TODO
        pred = torch.sigmoid(pred) # Then convert to sigmoid (0~1)
        preds = [pred] # assume batchsize=1
        t1 = time.time()

        # labels/mask have different size & cannot be Tensored -- better way?
        labels   = [torch.tensor(v["labels"], dtype=torch.float32).to(device) for v in info]
        mask     = [torch.tensor(v["mask"], dtype=torch.float32).to(device) for v in info] # to float
        Gsize     = torch.tensor([v["numnode"] for v in info]).to(device)
        pnames   = [v["pname"] for v in info]
        grids     = [v["grids"] for v in info]
       
        ## 2. Structure loss
        loss_struct, mae = structure_loss( Yrec_s, labelxyz ) #both are Kx3 coordinates
        
        ## 3. Regularization Loss --  regularize pre-sigmoid value |x|<4
        p_reg = torch.nn.functional.relu(torch.sum(pred*pred-25.0)) #safe enough?

        loss = loss_struct #TODO
        
        #header = [v["pname"]+" %8.3f"*3%tuple(v["xyz"].squeeze()) for v in info]
        print(info[0]['pname'], float(loss.cpu()))
            
        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        
        # Only update after certain number of accululations.
        if (b_count+1)%accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            # print("TRAIN Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %8.3f (%6.3f/%6.3f/%6.3f/%6.3f) error %3d, %s"
            #   %(modelname, epoch, max_epochs, b_count, len(train_loader),
            #     np.sum(temp_loss["total"][-1*accum:]), 
            #     #np.sum(temp_loss["BCEc"][-1*accum:]),
            #     #np.sum(temp_loss["BCEg"][-1*accum:]),
            #     #np.sum(temp_loss["BCEr"][-1*accum:]),
            #     # bygrid[0],bygrid[1],bygrid[2],
            #     np.sum(temp_loss["contrast"][-1*accum:]),
            #     e_count, pnames[0]
            #     ))
            
            b_count += 1

    # Empty the grad anyways
    optimizer.zero_grad()
    for k in train_loss:
        if k in temp_loss:
            train_loss[k].append(np.array(temp_loss[k]))

    b_count=0
    temp_loss = {"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[]}
    with torch.no_grad(): # wihout tracking gradients

        for i, args in enumerate(train_loader):
            if args == None:
                e_count += 1
                continue
        
            (Grec, Glig, labelxyz, keyidx, info, b, node, edge) = args
            if Grec == None:
                continue
                    
            labelidx = [torch.eye(n)[idx] for n,idx in zip(Glig.batch_num_nodes(),keyidx)]
            labelxyz = to_cuda( labelxyz, device )

            # predicted structure, Attention, category-for-GridNet
            Yrec_s, z, pred = model(to_cuda(Grec, device), to_cuda(node, device),
                                    to_cuda(Glig, device), to_cuda(labelidx, device), to_cuda(edge, device))

            if Yrec_s.shape[1] != 4: continue ##debug -- why?

            ## 1. GridNet loss related -- TODO
            pred = torch.sigmoid(pred) # Then convert to sigmoid (0~1)
            preds = [pred] # assume batchsize=1
            t1 = time.time()

            # labels/mask have different size & cannot be Tensored -- better way?
            labels   = [torch.tensor(v["labels"], dtype=torch.float32).to(device) for v in info]
            mask     = [torch.tensor(v["mask"], dtype=torch.float32).to(device) for v in info] # to float
            Gsize     = torch.tensor([v["numnode"] for v in info]).to(device)
            pnames   = [v["pname"] for v in info]
            grids     = [v["grids"] for v in info]
       
            ## 2. Structure loss
            loss_struct, mae = structure_loss( Yrec_s, labelxyz ) #both are Kx3 coordinates
            
            loss = loss_struct #TODO
        
            #header = [v["pname"]+" %8.3f"*3%tuple(v["xyz"].squeeze()) for v in info]
            print(info[0]['pname'], float(loss.cpu()))

            temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
            #temp_loss["BCEc"].append(loss1c.cpu().detach().numpy()) #store as per-sample loss 
            #temp_loss["BCEg"].append(loss1g.cpu().detach().numpy()) #store as per-sample loss 
            #temp_loss["BCEr"].append(loss1r.cpu().detach().numpy()) #store as per-sample loss 
            #temp_loss["contrast"].append(loss2.cpu().detach().numpy()) #store as per-sample loss
                
            if (b_count+1)%accum == 0:
                # print("VALID Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %.3f (%.3f/%.3f/%.3f/%.3f)"
                #       %(modelname, epoch, max_epochs, b_count, len(valid_loader),
                #         np.sum(temp_loss["total"][-1*accum:]),
                #         #np.sum(temp_loss["BCEc"][-1*accum:]),
                #         #np.sum(temp_loss["BCEg"][-1*accum:]),
                #         #np.sum(temp_loss["BCEr"][-1*accum:]),
                #         bygrid[0],bygrid[1],bygrid[2],
                #         np.sum(temp_loss["contrast"][-1*accum:])))
                b_count += 1
                
    for k in valid_loss:
        if k in temp_loss:
            valid_loss[k].append(np.array(temp_loss[k]))

    print("** SUMM, train/valid loss: %7.4f %7.4f"%((np.mean(train_loss['total'][-1]), np.mean(valid_loss['total'][-1]))))
    
    # Update the best model if necessary:
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

