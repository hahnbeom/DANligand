#!/usr/bin/env python
from SE3.wrapper2 import *

import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
import torch
from FullAtomNet import *


ntypes=15
model = SE3TransformerWrapper( l0_in_features=65+N_AATYPE+2, num_edge_features=2,
                               l0_out_features=ntypes, #category only
                               l1_out_features=1,
                               ntypes=ntypes )
print("Nparams:", count_parameters(model))

w_reg = 1e-5

params_loader = {
          'shuffle': True,
          'num_workers': 3 if '-debug' not in sys.argv else 1,
          'pin_memory': True,
          'collate_fn': collate,
          'batch_size': 5}

# default setup
set_params = {
    'root_dir'     : "/projects/ml/ligands/motif/backbone/", #let each set get their own...
    'ball_radius'  : 12.0,
    #'ballmode'     : 'all',
    #'sasa_method'  : 'sasa',
    #'edgemode'     : 'distT',
    #'edgek'        : (0,0),
    #'edgedist'     : (10.0,6.0), 
    #'distance_feat': 'std',
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    #"CBonly"       : ('-CB' in sys.argv),
    #'aa_as_het'   : True,
    'debug'        : ('-debug' in sys.argv),
    }
base = "/projects/ml/ligands/motif/backbone/"

train_set = Dataset(np.load("data/train_proteinsv5or.npy"), **set_params)
train_loader = data.DataLoader(train_set,
                               worker_init_fn=lambda _: np.random.seed(),
                               **params_loader)

valid_set = Dataset(np.load("data/valid_proteinsv5or.npy"), **set_params)
valid_loader = data.DataLoader(valid_set,
                               worker_init_fn=lambda _: np.random.seed(),
                               **params_loader)

device      = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
optimizer   = torch.optim.AdamW(model.parameters(), lr=1e-4)
max_epochs  = 400
accum       = 10
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
    train_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    valid_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    best_models = []
    if not isdir(join("models", modelname)):
        if not silent: print("Creating a new dir at", join("models", modelname))
        os.mkdir(join("models", modelname))

def category_loss(cat, w, c, Gsize):
    s = 0
    loss1 = torch.tensor(0.0)
    loss2 = torch.tensor(0.0)
    ps = []
    ts = []

    for i,b in enumerate(Gsize):
        Ps = torch.tensor([float(k==cat[i]) for k in range(ntypes)]).repeat(2,1)
        #Ps = Ps.to(device)
        Ps = torch.transpose(Ps,0,1).to(device)
        Ps[:,1] = 1.0-Ps[:,0]
        
        wG = w[s:s+b]/b
        cG = c[s:s+b] #per-node per-category probability

        #sometimes q == [0,0,0,0]  -- how to penalize?
        q = torch.sum(wG*cG,dim=0) #node-mean; predict correct probability
        #print(" %5.3f"*ntypes%tuple(q)) #any value >0; higher means probable

        # take 1-exp(-q) so that high q == high Prob
        Qs = torch.exp(-q/2.0).repeat(2,1)
        Qs = torch.transpose(Qs,0,1)
        Qs[:,0] = 1.0-Qs[:,1]
        #Qs = torch.nn.functional.softmax(Qs, dim=1)
        L = torch.mean(-Ps*torch.log(Qs+1.0e-6))

        print("%2d %4.2f %6.3f %3d | "%(cat[i],Qs[cat[i],0],float(L),b),
              " %4.2f"*ntypes%tuple(Ps[:,0]),":",
              " %4.2f"*ntypes%tuple(Qs[:,0]))

        loss1 = loss1 + L
        s += b
        
    return loss1, loss2

def l1_loss(Y, BB, wo, wb, v, R, cat, Gsize):
    #w: nnode x ntype
    #v: nnode x 3
    s = 0
    lossY = torch.tensor(0.0).to(device)
    lossB = torch.tensor(0.0).to(device)
    #a = torch.tensor([1.0,1.0,1.0]).to(device)
    n = 0
    for i,b in enumerate(Gsize):
        y = Y[i].float()
        vG = v[s:s+b]
        if cat[i] == 0: continue

        vG = torch.squeeze(vG)
        
        n += 1
        rot = R[cat[i]]
        wOG = wo[s:s+b,cat[i]]/b # mean
        wvO = rot(torch.matmul(wOG,vG))

        #wBG = wb[s:s+b,cat[i]]/b # mean
        #wvB = rot(torch.matmul(wBG,vG))
        
        f = torch.nn.MSELoss()
        mag = f(torch.tensor(1.0).to(device),torch.sum(wvO*wvO))
        err = f(wvO,y)
        
        lossY = lossY + err + mag
        print("%1d %2d %5.2f %5.2f | %8.3f %8.3f %8.3f : %8.3f %8.3f %8.3f"%(i,cat[i],err,mag,
                                                                             float(wvO[0]),float(wvO[1]),float(wvO[2]),
                                                                             float(y[0]),float(y[1]),float(y[2])))
        
    return lossY, lossB, n 

start_epoch = epoch
for epoch in range(start_epoch, max_epochs):
    
    b_count,e_count=0,0
    temp_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    for G, node, edge, info in train_loader:
        if G == None: continue
        # Get prediction and target value

        # weights (orientation,bb,cat), cat, "v", pre-processed vector, per-cat Rotation matrix
        wo,wb,wc,c,v,R = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))

        #prediction = prediction[:, 0, 0]
        #truth      = G.ndata["lddt"].to(prediction.device)
        cat      = torch.tensor([v["motifidx"] for v in info]).to(device)
        yaxis     = torch.tensor([v["yaxis"] for v in info]).to(device) # y-vector
        bbxyz      = torch.tensor([v["dxyz"] for v in info]).to(device) # bb-vector
        Gsize      = torch.tensor([v["numnode"] for v in info]).to(device)
        b = int(Gsize.shape[0])
        print(cat)

        loss1,loss2 = category_loss(cat, wc, c, Gsize)
        #loss3,loss4,n  = l1_loss(yaxis, bbxyz, wo, wb, v, R, cat, Gsize)
        
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters(): l2_reg += torch.norm(param)

        #loss = loss1 + loss2 + loss3 + w_reg*l2_reg
        loss1 = loss1/b
        loss = loss1 + w_reg*l2_reg
        #if n == 0: continue
        
        loss.backward(retain_graph=True)

        temp_loss["cat"].append(loss1.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        
        # Only update after certain number of accululations.
        if (b_count+1)%accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("TRAIN Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %.3f, error %d"
                  %(modelname, epoch, max_epochs, b_count, len(train_loader), np.sum(temp_loss["total"][-1*accum:]),e_count)) 
            
            b_count += 1
        else:
            e_count += 1
            
    # Empty the grad anyways
    optimizer.zero_grad()
    for k in train_loss:
        train_loss[k].append(np.array(temp_loss[k]))

    b_count=0
    temp_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    with torch.no_grad(): # wihout tracking gradients

        for k in range(3): #repeat multiple times for stability
            for G, node, edge, info in valid_loader:
                if G == None:
                    continue
                
                # Get prediction and target value
                cat      = torch.tensor([v["motifidx"] for v in info]).to(device)
                yaxis     = torch.tensor([v["yaxis"] for v in info]).to(device) # y-vector
                bbxyz      = torch.tensor([v["dxyz"] for v in info]).to(device) # bb-vector
                Gsize      = torch.tensor([v["numnode"] for v in info]).to(device)
                b = int(Gsize.shape[0])
                
                loss1,loss2 = category_loss(cat, wc, c, Gsize)
                #loss3,loss4,n  = l1_loss(yaxis, bbxyz, wo, wb, v, R, cat, Gsize)
                
                loss = loss1/b #+ loss2 + loss3
                
                temp_loss["cat"].append(loss1.cpu().detach().numpy())
                temp_loss["total"].append(loss.cpu().detach().numpy())
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


# In[ ]:




