import torch
import numpy as np
import sys,os
import time
import dgl

import torch.nn as nn
import src.myutils as myutils

# efficient version
from src.model_e import HalfModel
import src.dataset_e as dataset

# memory-leak version
#from src.model_merge import HalfModel
#import src.dataset as dataset

# DDP 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP 

import warnings
warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
PARENTPATH = os.path.dirname(os.path.abspath(__file__))

# args
from args import args_dynamic2 as args
args.max_subset = 30 #for inferrencing

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

set_params = {'K':args.K,
              'datapath':'./',
              'neighmode': args.neighmode,
              'topk'  : args.topk,
              'mixkey': args.mixkey,
              'pert'  : args.pert,
              'maxnode': 2500, #can be bigger for inferrence
              'max_subset': args.max_subset,
              'dropH' : True,
              'version': 3 # for inferrence
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
               'dropout':0.0,
               'classification_mode':args.classification_mode,
}

generator_params = {
    'num_workers': 8, #1 if '-debug' in sys.argv else 10,
    'pin_memory': True,
    'collate_fn': dataset.collate,
    'batch_size': 1,
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model( args_in ):
    model = HalfModel(**modelparams)
    model.to(device)

    #print("model", model.state_dict().keys())
    model_dicts = list(model.state_dict().keys())
    #print("models/%s/model.pkl"%args_in.modelname, os.path.exists("models/%s/model.pkl"%args_in.modelname))
    modelpath = "%s/models/%s/model.pkl"%(PARENTPATH,args_in.modelname)
    if os.path.exists(modelpath): 
        checkpoint = torch.load(modelpath, map_location=device)
        
        trained_dict = {}
        for key in checkpoint["model_state_dict"]:
            if key.startswith("module."):
                newkey = key[7:]
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
            else:
                trained_dict[key] = checkpoint["model_state_dict"][key]
        
        model.load_state_dict(trained_dict)
    
    return model

def main(prefix):
    model = load_model( args )
    model.eval()
        
    #file check
    missing = []
    for f in ['%s.ligands.mol2'%prefix,'%s.keyatom.def.npz'%prefix,'%s.score.npz'%prefix]:
        if not os.path.exists(f): missing.append(f)
    if missing != []:
        sys.exit('missing files: '+ ' '.join(missing))

    tags = myutils.read_mol2_batch('%s.ligands.mol2'%prefix, tag_only=True)[-1]
    tags.sort() # CHEMBL first
    tags_batched = []
    for i,tag in enumerate(tags):
        if i%args.max_subset == 0:
            tags_batched.append([prefix,[]])
        tags_batched[-1][1].append(tag)
    
    runset = dataset.DataSet(tags_batched, **set_params)

    ## keyatom
    loader = torch.utils.data.DataLoader(runset,  **generator_params)

    torch.cuda.empty_cache()
    print("Launching %d molecules w/ batch %d..."%(len(tags),args.max_subset))

    ncount = 0
    t0 = time.time()
    with torch.no_grad():
        t_l0 = time.time()
        for i,(G1s,G2s,keyxyz,keyidx,blabel,info) in enumerate(loader):
            if G1s == None:
                print(f"pass {info['name']}")
                continue
        
            G1s = myutils.to_cuda(G1s, device) # lig
            G2s = myutils.to_cuda(G2s, device) # lig
        
            keyidx = myutils.to_cuda(keyidx, device)
            Yrec, affinity, z = model( G1s, G2s, keyidx, True, False)

            #print(G2s.batch_num_nodes().shape[0], affinity.shape[0])
            #print(' '.join(info['lig']), ":", " %8.3f"*affinity.shape[0]%tuple(torch.sigmoid(affinity)))
            #print(' '.join(info['lig']), ":", torch.sigmoid(affinity))
            ncount += len(info['lig'])
            
            t1 = time.time()
            if i%10 == 9:
                print("Scanned %d/%d molecules in %.3f secs; %.3f per mol"%(ncount,len(tags),t1-t0, (t1-t0)/ncount))
                
    t9 = time.time()
    print("Final: Scanned %d/%d molecules in %.3f secs; %.3f per mol"%(ncount,len(tags),t9-t0,(t9-t0)/ncount))

if __name__ == "__main__":
    prefix = sys.argv[1]
    main(prefix)
