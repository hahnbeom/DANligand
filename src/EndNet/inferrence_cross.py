#!/usr/bin/env python
import os, sys
import numpy as np
from os.path import join, isdir, isfile
import torch
import time

MYPATH = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0,MYPATH)

## DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model_former import EndtoEndModel
from src.myutils import count_parameters, to_cuda, read_mol2_batch
from src.dataset_v3 import collate, DataSet
import src.Loss as Loss

from args3 import args_supd as args

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")

NTYPES = 6
ddp = False
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
    'maxedge'      : 40000,
    'maxnode'      : 2000,
    'max_subset'   : 20,
    'drop_H'       : True,
    'input_features': args.input_features

}

params_loader={
    'shuffle': False,
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

    modelf = "%s/models/%s/classifier.pkl"%(MYPATH,args.modelname)
    if os.path.exists(modelf):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(modelf,map_location=device)

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
        #if rank == 0: print("params", nnew, nexist)
        
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
        
    if rank == 0:
        print("Nparams:",count_parameters(model))
        print("Loaded")

    return model,optimizer,epoch,train_loss,valid_loss

def load_data(infile,type='txt',workpath=None):
    from torch.utils import data
    
    target_s = []
    ligands_s = []
    is_ligand_s = []
    if type == 'txt':
        for ln in open(infile,'r'):
            x = ln.strip().split()
            is_ligand = bool(x[0]) #1: PL, 0: PP
            target = x[1]
            liglist = x[2:]
        
            target_s.append(target)
            ligands_s.append(liglist)
            is_ligand_s.append(is_ligand)
            
    elif type == 'mol2':
        tags = read_mol2_batch(infile,tag_only=True)[-1]
        tags.sort() # let CHEMBL come first
        nsub = set_params['max_subset']
        targetname = infile.replace('.ligand.mol2','')
        
        for i,tag in enumerate(tags):
            if i%nsub == nsub-1 or (i == len(tags)-1):
                target_s.append(targetname)
                ligands_s.append(['infer']+tags[i+1-nsub:i+1])

        is_ligand_s = [True for _ in target_s]
        #is_ligand_s += [True for _ in tags]
        
    if workpath != None:
        set_params['datapath'] = workpath # override
    data_set = DataSet(target_s, is_ligand_s, ligands_s, **set_params)
    data_loader = data.DataLoader(data_set, **params_loader)
    return data_loader

def report_structural_attn( A, grid, Ps, keyatms, pnames ):
    # A: B x Nmax x K

    #for b, (x,k) in enumerate( zip(grid, nK) ): #ngrid: B x maxn
    #    n = x.shape[0]
    #    z = A[b,:n,:k] # N x K
        #com = torch.mean(z, x)
        #dcom = x - com

    form = 'HETATM %5d %3s UNK %1s%4d    %8.3f%8.3f%8.3f  1.00%6.2f\n'

    atypes = ['X','Don','Acc','Bot','Ali','Aro']
    for i in range(1,6):
        ctr = 0
        out = open('motifP.%s.pdb'%atypes[i],'w')
        for ps,x in zip(Ps,grid): # N x 6; N x 3
            if ps[i] > 0.3:
                ctr += 1
                out.write(form%(ctr,'O','X',i+1,x[0],x[1],x[2],ps[i]))
        out.close()

    for b,(atms,z,pname) in enumerate(zip(keyatms,A,pnames)): # batch dim
        out = open(pname+'.attn.pdb','w')

        for k,atm in enumerate(atms):
            chain = ['A','B','C','D','E','F','G','H','I','J','K'][k]
            highP = torch.where(z[:,k]>0.005)[0]
            #print(k, highP)
            ctr = 0
            
            if highP.shape[0] > 0 :
                out.write('MODEL %d\n'%k)
                for i in highP:
                    ctr += 1
                    x = grid[i]
                    out.write(form%(ctr,atm,chain,ctr,x[0],x[1],x[2],z[i,k]*100.0))
                out.write('ENDMDL\n')
        out.close()

### train_model
def inferrence(data_loader, report_attn=False):
    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    torch.cuda.set_device(device)

    ## load_params
    model,optimizer,start_epoch,train_loss,valid_loss=load_params(rank)

    ## iteration
    #if ddp:
    #    mp.spawn(train_one_epoch,args=(world_size,0),nprocs=world_size,join=True)
    #    with torch.no_grad():
    #        train_one_epoch(model,optimizer,data_loader,rank,0,report_attn)
    #else:
    
    with torch.no_grad():
        train_one_epoch(model,optimizer,data_loader,rank,0,report_attn)
        
### train_one_epoch
def train_one_epoch(model,optimizer,loader,rank,epoch,report_attn):
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
            if True:
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
                                               gradient_checkpoint=False,
                                               drop_out=False)
                
                if MotifP == None:
                    continue

                ## 1. GridNet loss related
                lossGc, lossGg, lossGr = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                lossGcontrast = torch.tensor(0.0).to(device)
                p_reg = torch.tensor(0.).to(device)
                
                MotifP = torch.sigmoid(MotifP) # Then convert to sigmoid (0~1)
                MotifPs = [MotifP] # assume batchsize=1
                
                if cats != None:
                    cats = to_cuda(cats, device)
                    masks = to_cuda(masks, device)
                    
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
                    atms = info['atms'][0]
                    keyidx = [np.array(torch.where(idx>0)[1].cpu()) for idx in keyidx]
                    keyatms = [np.array(atms[i])[idx] for i,idx in enumerate(keyidx)]

                    MotifP = MotifP[grididx]
                    if report_attn:
                        grid = grid[0] + info['com'][0].to(device)
                        report_structural_attn( z, grid, MotifP, keyatms, info['ligands'][0] )
                    
                    try:
                    #if True:
                        # 2-1. structural loss
                        if eval_struct:
                            nK = nK.squeeze() # hack, take the first one alone
                            lossTs, mae = Loss.structural_loss( Yrec_s, keyxyz, nK ) #both are Kx3 coordinates
                            lossTr = args.w_spread*Loss.spread_loss( keyxyz, z, grid, nK )
                            
                        if isinstance(aff,tuple):
                            aff = aff[0]
                        # 2-2.s screening loss
                        lossScreen = args.w_screen*Loss.ScreeningLoss( aff, blabel )
                        Pbind = ['%4.2f'%float(a) for a in torch.sigmoid(aff)]
                    except:
                    #else:
                        pass
                    
                t2 = time.time()
                    
                ## final loss
                ## default loss
                loss = args.wGrid*(lossGc + lossGg + lossGr + lossGcontrast) \
                    + args.wTR*(lossTs + lossTr + lossScreen)
                
                #store as per-sample loss
                temp_loss["total"].append(loss.cpu().detach().numpy()) 
                temp_loss["BCEc"].append(lossGc.cpu().detach().numpy()) 
                temp_loss["BCEg"].append(lossGg.cpu().detach().numpy()) 
                temp_loss["BCEr"].append(lossGr.cpu().detach().numpy()) 
                temp_loss["contrast"].append(lossGcontrast.cpu().detach().numpy())
                #temp_loss["reg"].append((p_reg+l2_reg).cpu().detach().numpy())
                if lossTs > 0.0:
                    temp_loss["struct"].append(lossTs.cpu().detach().numpy())
                    temp_loss["mae"].append(mae.cpu().detach().numpy())
                    temp_loss["spread"].append(lossTr.cpu().detach().numpy())
                if lossScreen > 0.0:
                    temp_loss["Screen"].append(lossScreen.cpu().detach().numpy())
            
            # Only update after certain number of accululations.
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
                  ), ' '.join(Pbind), ','.join(info['ligands'][0]))
            t3 = time.time()
                        
            b_count += 1

    return temp_loss

## main
if __name__=="__main__":
    workpath = sys.argv[1]
    report_attn = '-attn' in sys.argv

    if workpath.endswith('.txt'):
        data_loader = load_data(workpath,type='txt')
        print(len(data_loader))
    else:
        os.chdir(workpath)
        data_loader = load_data('target.ligand.mol2',type='mol2',workpath='./')
        
    inferrence(data_loader, report_attn=report_attn)
