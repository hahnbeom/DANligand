#!/usr/bin/env python
from SE3.wrapperC2 import * # sync l1_out_features for rot & WOblock; + co-train BB & Yaxis

import time
import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
import torch
from FullAtomNet import *
MYPATH = os.path.dirname(os.path.abspath(__file__))

#if not os.path.exists('

TYPES = list(range(14))
ntypes=len(TYPES)
Ymode = 'node'
n_l1out = 8 # assume important physical properties e.g. dipole, hydrophobicity et al
nGMM = 3 # num gaussian functions

w_reg = 1e-5
LR = 1.0e-4

model = SE3TransformerWrapper( num_layers=10,
                               l0_in_features=65+N_AATYPE+3, num_edge_features=2,
                               l0_out_features=ntypes, #category only
                               l1_out_features=n_l1out,
                               ntypes=ntypes,
                               nGMM = nGMM )
print("Nparams:", count_parameters(model))
    
params_loader = {
    'shuffle': False,
    'num_workers': 8 if '-debug' not in sys.argv else 1,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 8 if '-debug' not in sys.argv else 1} # not much faster beyond 8

def sample1(cat):
    p = np.zeros(14)+0.01
    p[1] = 1.0
    #p[5] = 1.0
    w = p[cat]
    return w/np.sum(w)
    
# default setup
set_params = {
    #'root_dir'     : "/projects/ml/ligands/motif/exclchain/", #let each set get their own...
    'ball_radius'  : 12.0,
    "xyz_as_bb"    : True,
    "edgemode"     : "dist",
    "randomize"    : 0.0, # Ang, pert the rest
    "randomize_lig": 0.0, # Ang, pert the motif coord! #reduce noise...
    'debug'        : ('-debug' in sys.argv),
    'origin_as_node': True,
    'inference'    : True
    }

#valid_set = Dataset(np.load("data/valid_proteinsv5or.npy"), **set_params)
#valid_loader = data.DataLoader(valid_set,
#                               worker_init_fn=lambda _: np.random.seed(),
#                               **params_loader)

device      = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
modelname   = 'mpfix.rls'
model.to(device)
print("Running on device", device)

if os.path.exists("%s/models/%s/model.pkl"%(MYPATH,modelname)): 
    checkpoint = torch.load(join(MYPATH, "models", modelname, "model.pkl"), map_location=device)
    trained_dict = {}
    for key in checkpoint["model_state_dict"]:
        if key.startswith("module."):
            newkey = key[7:]
            trained_dict[newkey] = checkpoint["model_state_dict"][key]
        else:
            trained_dict[key] = checkpoint["model_state_dict"][key]
    model.load_state_dict(trained_dict)
else:
    sys.exit('no model found')

def is_same_simpleidx(label,idxs=np.arange(ntypes)):
    #print(label,idx)
    return torch.tensor([float(motif.SIMPLEMOTIFIDX[i]==motif.SIMPLEMOTIFIDX[label]) for i in idxs])
                    
def category_loss(cat, w, cs, Gsize, header="", out=sys.stdout):
    s = 0
    loss1 = torch.tensor(0.0)
    loss2 = torch.tensor(0.0)
    ps = []
    ts = []

    Ppreds = []
    for i,b in enumerate(Gsize):
        # ntype x 2
        wG = (w[s:s+b]/b)[:,:,None].repeat(1,1,2) # N x ntype
        cG = torch.transpose(cs[:,s:s+b,:],0,1) #N x ntype x 2
        q = torch.sum(wG*cG,dim=0) #node-mean; predict correct probability
        Qs = torch.nn.functional.softmax(q, dim=1)

        ## label: P
        Ps = torch.tensor([float(k==cat[i]) for k in range(ntypes)]).repeat(2,1)
        Ps = torch.transpose(Ps,0,1).to(device)
        Ps[:,1] = 1.0-Ps[:,0]

        Ps_cat = is_same_simpleidx(cat[i]).to(device)
        Ps_cat = Ps_cat.repeat(2,1)
        Ps_cat = torch.transpose(Ps_cat,0,1)
        Ps_cat[:,1] = 1-Ps_cat[:,0]
        
        L = -Ps*torch.log(Qs+1.0e-6)
        L1 = torch.mean(L[:,0]) # ture
        L2 = torch.mean(L[:,1]) # false
        L = 0.5*L1 + 0.5*L2

        Lcat = -Ps_cat*torch.log(Qs+1.0e-6)/(sum(Ps_cat)+1.0e-6)
        Lcat1 = torch.mean(Lcat[:,0])
        Lcat2 = torch.mean(Lcat[:,1])
        Lcat = torch.mean(Lcat)

        l = "C %s %2d %4.2f %6.3f %6.3f %6.3f %6.3f | "%(header[i],cat[i],Qs[cat[i],0],
                                                         float(L1),float(L2),float(Lcat1),float(Lcat2))+\
            " %4.2f"*ntypes%tuple(Ps[:,0])+":"+" %4.2f"*ntypes%tuple(Qs[:,0])

        out.write(l+'\n')
        loss1 = loss1 + L
        loss2 = loss2 + Lcat
        s += b
        Ppreds.append(np.array(Qs[:,0].cpu()))

    B  = len(Gsize)

    return loss1/B, loss2/B, np.array(Ppreds)

def l1_loss_Y(truth, w, x, R, cat, Gsize, header='', mode='node_weight', out=sys.stdout):
    #w: nnode x ntype
    #v: nnode x 3
    s = 0
    loss = torch.tensor(0.0).to(device)
    R = R.to(device)

    n = 0
    for i,b in enumerate(Gsize):
        t = truth[i].float()
        vG = x[1][s:s+b] #l1 pred: N x channel x 3
        if True: #cat[i] in TYPES:
            n += 1
            icat = cat[i]
            rot = R[icat] # 4x1 #weight on channel dim
            
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
                wv = torch.transpose(torch.einsum('i,ij->ij',w[s],vG[0]),0,1)
                #print(w[s], vG[0], wv)
                wv = rot(wv)
                
            f1 = torch.nn.ReLU()
            f2 = torch.nn.MSELoss()
            # penalize only if smaller than 0.5
            #mag = f(torch.tensor(1.0).to(device),torch.sum(wv*wv))
            mag = f1(1.0-torch.sum(wv*wv))
            err = f2(wv,t)
        
            loss = loss + err + mag
            l ="Y %s %1d %2d %5.2f %5.2f | %8.3f %8.3f %8.3f : %8.3f %8.3f %8.3f"%(header[i],i,cat[i],err,mag,
                                                                                    float(wv[0]),float(wv[1]),float(wv[2]),
                                                                                    float(t[0]),float(t[1]),float(t[2]))
            out.write(l+'\n')
            s = s + b
            
    if n > 1: loss = loss/n
        
    return loss, n 

def l1_loss_B(truth, w, x, R, cat, Gsize, header="",mode='node_weight',out=sys.stdout):
    s = 0
    loss = torch.tensor(0.0).to(device)
    n = 0
    A_s = []
    bb_s = []
    for i,b in enumerate(Gsize):
        t = truth[i].float()
        xG = x[0][s:s+b] #l0 pred: N x channel
        vG = x[1][s:s+b] #l1 pred: N x channel x 3
        icat = cat[i] # predicted category
        
        #if cat[i] in TYPES: # and cat[i] not in [0]: # skip null
        if True:
            n += 1
            ## USE estimated type
            rot = R['b'][icat].to(device) # 4 x nGMM
            sig = R['s'][icat].to(device) # 4 x nGMM
            
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
                wv = torch.transpose(rot(wv),0,1) # convert per category; n1_out x nGMM
                
            f1 = torch.nn.ReLU()
            f2 = torch.nn.MSELoss()
            
            # penalize only if smaller than 0.5
            mag = f1(2.25-torch.sum(wv*wv)) # make at least 1.5 Ang (1bond apart)
            
            # is this the best? shouldn't it communicate with wv somehow or pertype-weight?
            sigma = sig(xG[0])+1.0 # sigma = torch.tensor([1.0 for k in range(nGMM)])

            dev = torch.mean((wv-t)*(wv-t),dim=1)
            norm = 1.0/sigma
            P = norm*torch.exp(-dev/(sigma*sigma)) #>0
            dev = -torch.log(P+0.001)

            # allow negative? torch.sum(P)/3.0 ensures positive loss btw.
            err = -torch.log((torch.sum(P)+0.001)/3.0) # this will reward "converged Gaussian modes"...
            
            #reg = 0.1*(sigma*sigma-1.0)
            reg = 0.1*torch.mean(sigma*sigma-1.0,dim=0)
        
            loss = loss + torch.sum(err) + mag + reg

            A = 1.0/np.sqrt(2.0*np.pi)/sigma
            A_s.append(np.array(A.cpu()))
            bb_s.append(np.array(wv.cpu()))
            try:
                l = "B %s: %1d %2d %5.2f %5.2f | s %5.2f %5.2f %5.2f e %5.2f %5.2f %5.2f t %6.2f %6.2f %6.2f "%(header[i],i,cat[i],err,mag,
                                                                                                                sigma[0], sigma[1], sigma[2],
                                                                                                                dev[0],dev[1],dev[2],
                                                                                                                float(t[0,0]),float(t[0,1]),float(t[0,2]))                                                          
                #for k in range(nGMM):
                #    l += " , %6.2f %6.2f %6.2f"%(float(wv[k,0]),float(wv[k,1]),float(wv[k,2]))
                out.write(l+'\n')
            except:
                pass
                                     
            s = s + b
            
    if n > 1: loss = loss/n
        
    return loss, A_s, bb_s

def main(trgs, npzout=True, root_dir='./'):

    runset = Dataset(trgs, **set_params)
    set_params['root_dir'] = root_dir
    loader = data.DataLoader(runset,
                             worker_init_fn=lambda _: np.random.seed(),
                             **params_loader)
    
    with torch.no_grad(): # wihout tracking gradients
        t0 = time.time()
        Pvalues = []
        grids = []
        xyz_bb = []
        A_bb = []
        # iter through grid points
        for G, node, edge, info in loader:
            if G == None:
                continue

            out = open(info[0]["pname"]+'.pred.txt','a')

            # dummy
            cat      = torch.tensor([v["motifidx"] for v in info]).to(device)
            yaxis     = torch.tensor([v["yaxis"] for v in info]).to(device) # y-vector
            bbxyz      = torch.tensor([v["dxyz"] for v in info]).to(device) # bb-vector
            Gsize      = torch.tensor([v["numnode"] for v in info]).to(device)

            header = [v["pname"]+" %8.3f"*3%tuple(v["xyz"].squeeze()) for v in info]
            w,c,x,R = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))
        
            lossC,lossG,P = category_loss(cat, w['c'], c, Gsize, header, out)

            # update category based on prediction (if not null category)
            cat = np.argmax(P[:,1:],axis=1)+1 # TODO: this should be normalized!
            
            lossY,n  = l1_loss_Y(yaxis, w['y'], x, R['y'], cat, Gsize, header, mode=Ymode, out=out)
            lossB,A_s,bb_s = l1_loss_B(bbxyz, w['b'], x, R, cat, Gsize, header, mode=Ymode, out=out)
            out.close()

            Pvalues.append(P)

            if len(A_bb) == 0:
                A_bb = np.array(A_s)
                xyz_bb = np.array(bb_s)
            else:
                A_bb = np.concatenate([A_bb,np.array(A_s)])
                xyz_bb = np.concatenate([xyz_bb,np.array(bb_s)])

            grids += [v['xyz'] for v in info]

            #print(P.shape, len([v['xyz'] for v in info]))
            
        outnpz = info[0]["pname"]+".score.npz"
        Pvalues = np.concatenate(Pvalues)
        grids = np.array(grids)
        if npzout == True:
            if Pvalues.shape[0] != grids.shape[0]:
                print("Warning!! grid points and probability size don't match each other!")
            np.savez(outnpz, P=Pvalues, grids=grids, xyz_bb=xyz_bb, A_bb=A_bb)

        t1 = time.time()
        print("%s: Finished %d grid points in %.3f secs using batch=%d"%(info[0]["pname"],
                                                                         len(grids),t1-t0,
                                                                         params_loader['batch_size']))
                
if __name__ == "__main__":
    infile = sys.argv[1]
    if infile.endswith('.npz'):
        npzs = [sys.argv[1]]
    else:
        npzs = [l[:-1] for l in open(sys.argv[1])]
        if not npzs[0].endswith('.npz'):
            sys.exit("input file should either end with '.npz' or contain a list of '.npz' files")
        
    for npz in npzs:
        trgs = np.load(npz,allow_pickle=True)['name']
        print("Running ",npz,len(trgs))
        main(trgs, npzout=True)
