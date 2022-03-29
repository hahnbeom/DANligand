#!/usr/bin/env python
import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
import torch
from torch.utils import data

#from src import *
from SE3.model import SE3TransformerWrapper
from src.myutils import N_AATYPE, count_parameters, to_cuda
from src.dataset import collate, Dataset
import src.motif as motif

ntypes=len(motif.MOTIFS)

w_reg = 1e-5
LR = 1.0e-4
w_contrast = 1.0
w_false = 0.2/ntypes # 0.2: releate importance; 1.0/ntypes null contribution

model = SE3TransformerWrapper( num_layers=3,
                               l0_in_features=65+N_AATYPE+3,
                               num_edge_features=3, #1-hot bond type x 2, distance 
                               l0_out_features=ntypes, #category only
                               #l1_out_features=n_l1out,
                               ntypes=ntypes)

print("Nparams:", count_parameters(model))

params_loader = {
          'shuffle': False,
          'num_workers': 5 if '-debug' not in sys.argv else 1,
          'pin_memory': True,
          'collate_fn': collate,
          'batch_size': 1 if '-debug' not in sys.argv else 1}
    
# default setup
set_params = {
    'root_dir'     : "/home/hpark/data/HmapMine/features.grid/",
    'ball_radius'  : 12.0,
    'edgedist'     : (2.0,4.5), 
    #"upsample"     : sample1,
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    "CBonly"       : False,
    'debug'        : ('-debug' in sys.argv),
    }

train_set = Dataset(np.load("data/train.npy"), **set_params)
train_loader = data.DataLoader(train_set,
                               worker_init_fn=lambda _: np.random.seed(),
                               **params_loader)

valid_set = Dataset(np.load("data/valid.npy"), **set_params)
valid_loader = data.DataLoader(valid_set,
                               worker_init_fn=lambda _: np.random.seed(),
                               **params_loader)

device      = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
optimizer   = torch.optim.AdamW(model.parameters(), lr=LR)
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
    model.load_state_dict(checkpoint["model_state_dict"])
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
    train_loss = {"total":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[]}
    valid_loss = {"total":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[]}

    best_models = []
    if not isdir(join("models", modelname)):
        if not silent: print("Creating a new dir at", join("models", modelname))
        os.mkdir(join("models", modelname))

    # initialize here
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer,torch.nn.Linear) and \
           "WOblock" in name: # or 'Rblock' in name:
            layer.weight.data.fill_(0.0)

def MaskedBCE(labels,preds,masks):
    # Batch
    lossG = torch.tensor(0.0).to(device)
    lossR = torch.tensor(0.0).to(device)
    for label,mask,pred in zip(labels,masks,preds):
        ngrid = label.shape[0]
        #print(label.shape, pred.shape, mask.shape)
        #pred: N x nmotiftype
        #label: N x nmotiftype
        #mask: N

        Q = pred[-ngrid:]
        a = -label*torch.log(Q+1.0e-6) #-PlogQ
        b = -(1.0-label)*torch.log((1.0-Q)+1.0e-6)

        # normalize by num grid points & label points
        norm = torch.sum(label)/torch.sqrt(torch.sum(mask))
        
        lossG += torch.sum(torch.matmul(mask, a))*norm
        lossR += torch.sum(torch.matmul(mask, b))*norm

    return lossG, lossR

def ContrastLoss(preds,masks):
    loss = torch.tensor(0.0).to(device)
    for mask,pred in zip(masks,preds):
        imask = 1.0 - mask
        ngrid = mask.shape[0]
        psum = torch.sum(torch.matmul(imask,pred[-ngrid:]))/ngrid

        loss += psum
    return loss

start_epoch = epoch
count = np.zeros(ntypes)
for epoch in range(start_epoch, max_epochs):
    b_count,e_count=0,0
    temp_loss = {"total":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[]}
    for G, node, edge, info in train_loader:
        if G == None: continue
        # Get prediction and target value
    
        pred = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))

        # labels/mask have different size & cannot be Tensored
        # better way?
        labels   = [torch.tensor(v["labels"], dtype=torch.float32).to(device) for v in info]
        mask     = [torch.tensor(v["mask"], dtype=torch.float32).to(device) for v in info] # to float
        Gsize     = torch.tensor([v["numnode"] for v in info]).to(device)
       
        #header = [v["pname"]+" %8.3f"*3%tuple(v["xyz"].squeeze()) for v in info]

        loss1g,loss1r = MaskedBCE(labels,pred,mask)
        loss1r = w_false*loss1r
        loss2 = ContrastLoss(pred,mask) # make overal prediction low as possible
        
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters(): l2_reg += torch.norm(param)
        loss = loss1g + loss1r + w_contrast*loss2 +  w_reg*l2_reg

        #print(loss1, loss2)
        loss.backward(retain_graph=True)
        
        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["BCEg"].append(loss1g.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["BCEr"].append(loss1r.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["contrast"].append(loss2.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["reg"].append(l2_reg.cpu().detach().numpy()) #store as per-sample loss
        
        # Only update after certain number of accululations.
        if (b_count+1)%accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("TRAIN Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %.3f (%.3f/%.3f/%.3f)"
              %(modelname, epoch, max_epochs, b_count, len(train_loader),
                np.sum(temp_loss["total"][-1*accum:]),
                np.sum(temp_loss["BCEg"][-1*accum:]),
                np.sum(temp_loss["BCEr"][-1*accum:]),
                np.sum(temp_loss["contrast"][-1*accum:])))
            
            b_count += 1
        else:
            e_count += 1
            
    # Empty the grad anyways
    optimizer.zero_grad()
    for k in train_loss:
        train_loss[k].append(np.array(temp_loss[k]))

    b_count=0
    temp_loss = {"total":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[]}
    with torch.no_grad(): # wihout tracking gradients

        for k in range(1): #repeat multiple times for stability
            for G, node, edge, info in valid_loader:
                if G == None:
                    continue
                    
                # Get prediction and target value
                pred = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))

                labels   = [torch.tensor(v["labels"], dtype=torch.float32).to(device) for v in info]
                mask     = [torch.tensor(v["mask"], dtype=torch.float32).to(device) for v in info] # to float
                Gsize     = torch.tensor([v["numnode"] for v in info]).to(device)
                
                grids     = [v["grids"] for v in info]
                tags = [v["pname"] for v in info]

                loss1g,loss1r = MaskedBCE(labels,pred,mask)
                loss1r = w_false*loss1r
                loss2 = ContrastLoss(pred,mask) # make overal prediction low as possible
        
                loss = loss1g + loss1r + w_contrast*loss2

                for tag,p,g,m,l in zip(tags,pred,grids,mask,labels):
                    m = np.array(m.cpu())
                    imask = np.where(m>0.0)[0]
                    l = np.array(l.cpu())
                    ngrid = l.shape[0]

                    p = np.array(p[-ngrid:].cpu())
                    if set_params['debug']:
                        np.savez("%s.prob.npz"%(tag), p=p, grids=g)
                        try:
                            for i in imask:
                                print("%-10s %3d"%(tag,int(np.where(l[i]>0)[0])),
                                      " %8.3f"*3%tuple(g[i]),
                                      " %4.2f"*len(p[i,:10])%tuple(p[i,:10])," | ",
                                      " %4.2f"*len(p[i,10:])%tuple(p[i,10:]))
                                  
                        except:
                            print("pass")

                #print(["%5.3f "%f for f in np.abs(ts-ps)])
                temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
                temp_loss["BCEg"].append(loss1g.cpu().detach().numpy()) #store as per-sample loss 
                temp_loss["BCEr"].append(loss1r.cpu().detach().numpy()) #store as per-sample loss 
                temp_loss["contrast"].append(loss2.cpu().detach().numpy()) #store as per-sample loss
                
                if (b_count+1)%accum == 0:
                    print("VALID Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %.3f (%.3f/%.3f/%.3f)"
                          %(modelname, epoch, max_epochs, b_count, len(train_loader),
                            np.sum(temp_loss["total"][-1*accum:]),
                            np.sum(temp_loss["BCEg"][-1*accum:]),
                            np.sum(temp_loss["BCEr"][-1*accum:]),
                            np.sum(temp_loss["contrast"][-1*accum:])))
                b_count += 1
                
    for k in valid_loss:
        valid_loss[k].append(np.array(temp_loss[k]))

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

