import torch
import numpy as np
import sys,os
import time
import dgl

import torch.nn as nn
from src.model_merge import HalfModel
import src.myutils as myutils
import src.dataset_structonly as dataset

# DDP 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP 

import warnings
warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

# args
from args import args_grid15_scratch as args

set_params = {'K':args.K,
              'datapath':args.datapath,
              'neighmode': args.neighmode,
              'topk'  : args.topk,
              'mixkey': args.mixkey,
              'pert'  : args.pert,
              'version': 2,
              'maxnode': 1300, #important for trigonmetry pair using BNNc dimension; this limits to ~16G when stack=5,c=32,b=2
              'dropH' : True,
}

modelparams = {'num_layers_lig':3,
               'num_layers_rec':2,
               'num_channels':32, #within se3; adds small memory (~n00 MB?)
               'l0_in_features_lig':15, #19 -- drop dummy K-dimension
               'l0_in_features_rec':14, #18 -- drop dummy K-dimension
               'l1_in_features':0,
               'l0_out_features':64, #at Xatn
               'n_heads_se3':4,
               'embedding_channels':64,
               'c':32, #actual channel size within trigon attn?
               'n_trigonometry_module_stack':5,
               'div':4,
               'dropout':0.2,
               'classification_mode':args.classification_mode
}

generator_params = {
    'num_workers': 5, #1 if '-debug' in sys.argv else 10,
    'pin_memory': True,
    'collate_fn': dataset.collate,
    'batch_size': 5,
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def structural_loss(prefix, Yrec, Ylig, nK ):
    # Yrec: BxKx3 Ylig: K x 3
    
    N = Yrec.shape[0]
    dY = Yrec - Ylig # K x 3

    loss1 = torch.sum(dY*dY,dim=1) # K
    
    loss1_sum = torch.sum(loss1)/N 
    
    meanD = torch.sum(torch.sqrt(loss1))/torch.sum(nK) # mean over 

    return loss1_sum, meanD

def spread_loss(Ylig, A, G, nK, sig=2.0): #Ylig:label(B x K x 3), A:attention (B x Nmax x K)
    loss2 = torch.tensor(0.0)
    i = 0
    N = A.shape[0]
    for b,(n,k) in enumerate(zip(G.batch_num_nodes(),nK)): #batch
        x = G.ndata['x'][i:i+n,:] #N x 1 x 3
        z = A[b,:n,:k] # N x K
        y = Ylig[b][None,:k,:].squeeze() # 1 x K x 3
        
        #print(b,int(i),int(i+n),x.shape, z.shape)
        dX = x-y
        
        overlap = torch.exp(-torch.sum(dX*dX,axis=-1)/(sig*sig)) # N x K
        if z.shape[0] != overlap.shape[0]: continue

        loss2 = loss2 - torch.sum(overlap*z)

        i += n
    return loss2 # max -(batch_size x K)

def load_data(args_in):

    train_set = dataset.DataSet(np.load('data/trainlist.combo.npy'), **set_params)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args_in.world_size, rank=args_in.rank)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, **generator_params)
    
    valid_set = dataset.DataSet(np.load('data/validlist.combo.npy'), **set_params)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set, num_replicas=args_in.world_size, rank=args_in.rank)
    valid_loader = torch.utils.data.DataLoader(valid_set, sampler=valid_sampler, **generator_params)
    
    return train_loader, valid_loader

def load_model( args_in ):
    device = torch.device("cuda:%d"%args_in.rank if (torch.cuda.is_available()) else "cpu")

    modelname = args_in.modelname
    model = HalfModel(**modelparams)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_in.LR) #1.0e-3 < e-4

    if os.path.exists("models/%s/model.pkl"%modelname): 
        checkpoint = torch.load("models/%s/model.pkl"%modelname, map_location=device)
        
        model_dict = model.state_dict()
        model_keys = list(model_dict.keys())
        
        trained_dict = {}
        found = []
        new = []

        for key in checkpoint["model_state_dict"]:
            if key.startswith('module.') and key[7:] in model_keys:
                newkey = key[7:]
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
                found.append(newkey)
            elif 'trigon_module.'+key in model_keys:
                newkey = 'trigon_module.'+key
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
                found.append(newkey)
            else:
                trained_dict[key] = checkpoint["model_state_dict"][key]
                if key in model_keys:
                    found.append(key)
                else:
                    new.append(key)

        unclaimed = []
        for key in model_keys:
            if key not in found and key not in new:
                unclaimed.append(key)
                
        print(' Trsf\n'.join(found))
        print(' Added\n'.join(new))
        print(' Unclaimed\n'.join(unclaimed))
    
        model_dict.update(trained_dict)
        model.load_state_dict(model_dict,strict=False)
        
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        startepoch = 0 #checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]

        if 'mae' not in train_loss: train_loss['mae'] = []
        if 'reg' not in train_loss: train_loss['reg'] = []
        if 'mae' not in valid_loss: valid_loss['mae'] = []

        print("Loading a checkpoint at epoch %d"%startepoch)

    else:    
        if not os.path.isdir(os.path.join("models/", modelname)):
            print("Creating a new dir at", os.path.join("models", modelname))
            os.mkdir( os.path.join("models", modelname) )
        
        # zero initialize weighting FC layers
        for i, (name, layer) in enumerate(model.named_modules()):
            if isinstance(layer,torch.nn.Linear) and "phi" in name or "W" in name:
                if isinstance(layer,torch.nn.ModuleList):
                    for l in layer:
                        if hasattr(l,'weight'): l.weight.data.fill_(0.0)
                    else:
                        if hasattr(layer,'weight'):
                            layer.weight.data.fill_(0.0)

    
        train_loss = {'total':[],'mae':[],'reg':[],'loss1':[],'loss2':[]}
        valid_loss = {'total':[],'mae':[],'loss1':[],'loss2':[]}
        startepoch = 0

    return model, optimizer, train_loss, valid_loss, startepoch

def run_epoch(model, optimizer, generator, args_in, train=False):
    loss_t = {'total':[],'mae':[],'loss1':[],'loss2':[],'reg':[]} # temporary
        
    t0 = time.time()
    device = model.device
     
    with model.no_sync():
        with torch.cuda.amp.autocast(enabled=False):
            for i,(G1s,G2s,keyxyz,keyidx,info) in enumerate(generator):
                if G1s == None or keyxyz == None:
                    print(f"pass {info['name']}")
                    continue
        
                G1s = myutils.to_cuda(G1s, device) # lig
                G2s = myutils.to_cuda(G2s, device) # lig
                keyxyz = keyxyz.to(device)
                keyidx = myutils.to_cuda(keyidx, device)
        
                Yrec, _, z = model( G1s, G2s, keyidx, True)
    
                if train: prefix = "TRAIN-%d "%(args_in.rank)
                else: prefix = "VALID-%d "%(args_in.rank)
            
                loss1, mae = structural_loss( prefix, Yrec, keyxyz, info['nK'] )
                loss2 = args_in.w_spread*spread_loss( keyxyz, z, G1s, info['nK'] )

                print(f"{prefix} {i:5d}/{len(generator):5d} {float(loss1):8.3f} ({float(mae):8.3f}) {float(loss2):8.3f} {info['name']}" )
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters(): l2_reg += torch.norm(param)
                loss = loss1 + loss2 + args_in.w_reg*l2_reg 

                if train:
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step()
                    loss_t['reg'].append(l2_reg.cpu().detach().numpy())
        
                loss_t['total'].append(loss.cpu().detach().numpy())
                loss_t['loss1'].append(loss1.cpu().detach().numpy())
                loss_t['loss2'].append(loss2.cpu().detach().numpy())
                loss_t['mae'].append(mae.cpu().detach().numpy())

    return loss_t    

def main(rank):
    # ddp
    args.world_size = torch.cuda.device_count()
    args.rank = rank
    args.gpu = rank

    dist.init_process_group(backend='gloo', world_size=args.world_size, rank=args.rank) 
        
    model, optimizer, train_loss, valid_loss, startepoch = load_model(args)
    model = DDP(model, device_ids = [args.gpu], output_device=0, find_unused_parameters=False)

    train_generator, valid_generator = load_data(args)
    
    torch.cuda.empty_cache()
    
    for epoch in range(startepoch,args.maxepoch):
        model.train()
        loss_t = run_epoch(model, optimizer, train_generator, args, train=True)
        for key in train_loss:
            train_loss[key].append(np.array(loss_t[key]))
        
        model.eval()
        loss_v = run_epoch(model, optimizer, valid_generator, args, train=False)
        for key in valid_loss:
            valid_loss[key].append(np.array(loss_v[key]))

        if args.rank == 0:
            form = "Train/Valid: %3d %8.4f %8.4f %8.4f (%8.4f %8.4f %8.4f %8.4f)"
            print(form%(epoch, float(np.mean(loss_t['total'])), float(np.mean(loss_v['total'])),
                        float(np.mean(loss_t['reg'])),
                        float(np.mean(loss_t['mae'])), float(np.mean(loss_v['mae'])),
                        float(np.mean(loss_t['loss2'])), float(np.mean(loss_v['loss2'])),
            ))
    
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss}, os.path.join("models", args.modelname, "model.pkl"))

            if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss}, os.path.join("models", args.modelname, "best.pkl"))
                
if __name__ == "__main__":
    mp.freeze_support()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    if ('MASTER_ADDR' not in os.environ):
        os.environ['MASTER_ADDR'] = 'localhost' 
    if ('MASTER_PORT' not in os.environ):
        os.environ['MASTER_PORT'] = '12334'

    mp.spawn(main, nprocs=torch.cuda.device_count(), join=True )

