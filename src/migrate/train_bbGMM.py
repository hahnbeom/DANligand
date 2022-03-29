#!/usr/bin/env python
from SE3.wrapperB import * # sync l1_out_features for rot & WOblock

import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
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
TYPES = list(range(14))
ntypes=len(TYPES)
Ymode = 'node'
n_l1out = 8 # assume important physical properties e.g. dipole, hydrophobicity et al
nGMM = 3 # num gaussian functions

w_reg = 1e-5
LR = 1.0e-4

model = SE3TransformerWrapper( num_layers=6,
                               l0_in_features=65+N_AATYPE+3, num_edge_features=2,
                               l0_out_features=ntypes, #category only
                               l1_out_features=n_l1out,
                               ntypes=ntypes,
                               nGMM = nGMM )
print("Nparams:", count_parameters(model))

params_loader = {
          'shuffle': True,
          'num_workers': 5 if '-debug' not in sys.argv else 1,
          'pin_memory': True,
          'collate_fn': collate,
          'batch_size': 5 if '-debug' not in sys.argv else 1}


def sample1(cat):
    p = np.zeros(14)+0.01
    p[1] = 1.0
    #p[5] = 1.0
    w = p[cat]
    return w/np.sum(w)
    
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
    "xyz_as_bb"    : True,
    "upsample"     : upsample_category,
    #"upsample"     : sample1,
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    #"CBonly"       : ('-CB' in sys.argv),
    #'aa_as_het'   : True,
    'debug'        : ('-debug' in sys.argv),
    'origin_as_node': (Ymode == 'node')
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
optimizer   = torch.optim.AdamW(model.parameters(), lr=LR)
max_epochs  = 400
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
    train_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    valid_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    best_models = []
    if not isdir(join("models", modelname)):
        if not silent: print("Creating a new dir at", join("models", modelname))
        os.mkdir(join("models", modelname))

    # initialize here
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer,torch.nn.Linear) and \
           "WOblock" in name: # or 'Rblock' in name:
            layer.weight.data.fill_(0.0)
    
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

def l1_loss_sig(truth, w, v, sig, rot, cat, Gsize, header="",mode='node_weight'):
    #w: nnode x ntype
    #v: nnode x 3
    s = 0
    loss = torch.tensor(0.0).to(device)
    #a = torch.tensor([1.0,1.0,1.0]).to(device)
    n = 0
    for i,b in enumerate(Gsize):
        t = truth[i].float()
        vG = v[s:s+b] #N x channel x 3
        
        if cat[i] in TYPES and cat[i] not in [0]: # skip null
            n += 1
            icat = TYPES.index(cat[i])
            rot = R[icat] # 4 x nGMM
            
            #wG = w[s:s+b,cat[i]]/b # mean
            #wG = torch.squeeze(wG)
            if mode == 'simple':
                wv = torch.transpose(torch.mean(torch.squeeze(vG),dim=0),0,1) # 3x32
                wv = torch.squeeze(rot(wv))
                
            elif mode == 'simpler':
                wv = torch.transpose(torch.mean(vG,dim=0),0,1) # 3x1
                
            elif mode == 'node_weight':
                norm = vG.shape[0]*vG.shape[1]
                wG = w[s:s+b]/norm # N x n1out
                wv = torch.einsum('ij,ijk->k', wG, vG)
                #wv = rot(wv)

            elif mode == 'node':
                wv = torch.transpose(torch.einsum('i,ij->ij',w[s],vG[0]),0,1) # w[s]: n_l1out, vG: n_l1out x 3 -> n_1out x 3
                #-- universal to every category up to here (expressive enough?)
                
                # w/ GMM, rot should be n_1out x nGMM
                #print(w[s].shape, vG[0].shape, wv.shape)
                # 3x3
                wv = torch.transpose(rot(wv),0,1) # convert per category; n1_out x nGMM
                
            f1 = torch.nn.ReLU()
            f2 = torch.nn.MSELoss()
            # penalize only if smaller than 0.5
            mag = f1(2.25-torch.sum(wv*wv)) # make at least 1.5 Ang (1bond apart)
            
            #sigma = sig[s]+1.0
            sigma = torch.tensor([1.0,1.0,1.0]).to(device) #set it constant for simplicity....
            
            dev = torch.mean((wv-t)*(wv-t),dim=1)
            P = torch.exp(-dev/(sigma*sigma)) #>0
            dev = -torch.log(P+0.001)

            # allow negative? torch.sum(P)/3.0 ensures positive loss btw.
            err = -torch.log((torch.sum(P)+0.001)/3.0) # this will reward "converged Gaussian modes"...
            
            #reg = 0.1*(sigma*sigma-1.0)
            reg = 0.1*torch.mean(sigma*sigma-1.0,dim=0)
        
            loss = loss + torch.sum(err) + mag + reg
            try:
                l = "%s: %1d %2d %5.2f %5.2f | s %5.2f %5.2f %5.2f e %5.2f %5.2f %5.2f t %6.2f %6.2f %6.2f "%(header[i],i,cat[i],err,mag,
                                                                                                              sigma[0], sigma[1], sigma[2],
                                                                                                              dev[0],dev[1],dev[2],
                                                                                                              float(t[0,0]),float(t[0,1]),float(t[0,2]))                                                          
                #for k in range(nGMM):
                #    l += " , %6.2f %6.2f %6.2f"%(float(wv[k,0]),float(wv[k,1]),float(wv[k,2]))
                print(l)
            except:
                pass
                                     
            s = s + b
            
    if n > 1: loss = loss/n
        
    return loss, n 

start_epoch = epoch
count = np.zeros(ntypes)
for epoch in range(start_epoch, max_epochs):
    
    b_count,e_count=0,0
    temp_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    for G, node, edge, info in train_loader:
        if G == None: continue
        # Get prediction and target value

        #prediction = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))['0'][:, 0, 0]
        wo, wb,c,v,sig,R = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))

        #prediction = prediction[:, 0, 0]
        #truth      = G.ndata["lddt"].to(prediction.device)
        cat      = torch.tensor([v["motifidx"] for v in info]).to(device)
        yaxis     = torch.tensor([v["yaxis"] for v in info]).to(device) # y-vector
        bbxyz      = torch.tensor([v["dxyz"] for v in info]).to(device) # bb-vector
        Gsize      = torch.tensor([v["numnode"] for v in info]).to(device)
        
        header = [v["pname"]+" %8.3f"*3%tuple(v["xyz"].squeeze()) for v in info]

        loss,n  = l1_loss_sig(bbxyz, wb, v, sig, R, cat, Gsize, header, mode=Ymode)
        
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters(): l2_reg += torch.norm(param)
        loss = loss + w_reg*l2_reg
        
        #loss       = Loss(prediction, truth)/batchsize
        #loss, ts, ps = perGloss(w, c, truth, Gsize)
        
        #print(''.join(["%5.3f "%f for f in np.abs(ts-ps)]))
        if n == 0: continue
        
        for i in cat: count[int(i)] += 1
        #print(count)
        
        loss.backward(retain_graph=True)
        
        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["reg"].append(l2_reg.cpu().detach().numpy()) #store as per-sample loss
        
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
                wo, wb,c,v,sig,R = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))

                #prediction = prediction[:, 0, 0]
                cat      = torch.tensor([v["motifidx"] for v in info]).to(device)
                yaxis     = torch.tensor([v["yaxis"] for v in info]).to(device) # y-vector
                bbxyz      = torch.tensor([v["dxyz"] for v in info]).to(device) # bb-vector
                Gsize      = torch.tensor([v["numnode"] for v in info]).to(device)
                
                loss,n  = l1_loss_sig(bbxyz, wb, v, sig, R, cat, Gsize, mode=Ymode)

                #print(["%5.3f "%f for f in np.abs(ts-ps)])
                if n == 0: continue
                
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

