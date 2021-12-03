#!/usr/bin/env python
# coding: utf-8

# In[1]:
#from se3_transformer.model.transformer import *
#from se3_transformer.model.fiber import *
#from SE3.transformer import *
#from SE3.fiber import *
from SE3.wrapper import *

import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
#sys.path.insert(0,)
import torch
from FullAtomNet import *

'''
model = SE3Transformer(
    num_layers   = 2,
    num_heads    = 4,
    channels_div = 4,
    fiber_in=Fiber({0: 65}),
    fiber_hidden=Fiber({0: 32, 1:32, 2:32}),
    fiber_out=Fiber({0: 1}),
    fiber_edge=Fiber({0: 2}),
)
'''
model = SE3TransformerWrapper()
count_parameters(model)

params_loader = {
          'shuffle': True,
          'num_workers': 3 if '-debug' not in sys.argv else 1,
          'pin_memory': True,
          'collate_fn': collate,
          'batch_size': 10}
base = "/projects/ml/peptides/"
dirs = [join(base, i) for i in np.load("data/mylist.npy")]*30
train_set = Dataset(dirs, load_65types, max_atom_count=2900)
train_loader = data.DataLoader(train_set,
                               worker_init_fn=lambda _: np.random.seed(),
                               **params_loader)


base = "/projects/ml/peptides/"
dirs = [join(base, i) for i in np.load("data/valid_pep.npy")]
valid_set = Dataset(dirs, load_65types, max_atom_count=2900)
valid_loader = data.DataLoader(valid_set,
                               worker_init_fn=lambda _: np.random.seed(),
                               **params_loader)


device      = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
optimizer   = torch.optim.AdamW(model.parameters(), lr=1e-4)
max_epochs  = 400
accum       = 1
epoch       = 0
modelname   = "test"
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
    train_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
    valid_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
    best_models = []
    if not isdir(join("models", modelname)):
        if not silent: print("Creating a new dir at", join("models", modelname))
        os.mkdir(join("models", modelname))

def perGloss(w, c, truth, Gsize):
    s = 0
    loss = torch.tensor(0.0)
    ps = []
    ts = []
    #f = torch.nn.MSELoss()
    bins = to_cuda(torch.tensor([0.1*(k+0.5) for k in range(10)]), device)
                        
    for i,b in enumerate(Gsize):
        t = torch.mean(truth[s:s+b])
        tidx = int(t*10.0)
        Ps = torch.tensor([float(tidx==i) for i in range(10)]).repeat(2,1)
        Ps = to_cuda(torch.transpose(Ps,0,1), device)
        Ps[:,1] = 1.0-Ps[:,0]
        
        wG = w[s:s+b]
        cG = c[s:s+b]

        wG = torch.transpose(wG,0,1)/b
        p = torch.matmul(wG,cG) #[1,10]
        #Ps = torch.nn.functional.softmax(p, dim=1) + 1.0e-6
        # try independent prob.
        Qs = torch.nn.functional.sigmoid(p).repeat(2,1) #0~1
        Qs = torch.transpose(Qs,0,1)
        Qs[:,0] = 1.0-Qs[:,1]
        Qs = torch.nn.functional.softmax(Qs, dim=1)

        #ps.append(float(prob))
        wsum = torch.sum(Qs[:,0]*bins)
        ts.append(float(t))
        ps.append(float(wsum))
        
        #loss = loss + f(p,t)
        loss = torch.mean(-Ps*torch.log(Qs+1.0e-6))
        #print(Ps[:,0],Qs[:,0],loss)
        s += b
        
    return loss, np.array(ps), np.array(ts)
        
start_epoch = epoch
for epoch in range(start_epoch, max_epochs):
    
    b_count,e_count=0,0
    temp_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
    for G, node, edge, pname, tag, Gsize in train_loader:
        if G != None:
            # Get prediction and target value
            #x.to(device)
            #prediction = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))['0'][:, 0, 0]
            w,c = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))

            #prediction = prediction[:, 0, 0]
            #truth      = G.ndata["lddt"].to(prediction.device)
            truth      = G.ndata["lddt"].to(device)

            Loss       = torch.nn.MSELoss()
            #loss       = Loss(prediction, truth)/batchsize
            loss, ts, ps = perGloss(w, c, truth, Gsize)

            print(''.join(["%5.3f "%f for f in np.abs(ts-ps)]))
            loss.backward(retain_graph=True)
            
            temp_loss["total"].append(loss.cpu().detach().numpy())

            # Only update after certain number of accululations.
            if (b_count+1)%accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                print("TRAIN Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %.3f, error %d"
                       %(modelname, epoch, max_epochs, b_count, len(train_loader), np.sum(temp_loss["total"][-1*accum:]),e_count)) 
                
            b_count+=1
        else:
            e_count += 1
            
    # Empty the grad anyways
    optimizer.zero_grad()
    for k in train_loss:
        train_loss[k].append(np.array(temp_loss[k]))

    b_count=0
    temp_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
    with torch.no_grad(): # wihout tracking gradients

        for G, node, edge, pname, tag, Gsize in valid_loader:
            if G != None:
                optimizer.zero_grad()

                # Get prediction and target value
                w,c = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))

                #prediction = prediction[:, 0, 0]
                #truth      = G.ndata["lddt"].to(prediction.device)
                truth      = G.ndata["lddt"].to(device)
                
                Loss       = torch.nn.MSELoss()
                #loss       = Loss(prediction, truth)/batchsize
                loss, ts, ps = perGloss(w, c, truth, Gsize)
                
                print(["%5.3f "%f for f in np.abs(ts-ps)])
                loss.backward(retain_graph=True)
            
                temp_loss["total"].append(loss.cpu().detach().numpy())
                b_count+=1
                
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




