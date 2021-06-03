#!/usr/bin/env python
import sys
import os

import numpy as np
import torch
import time

import matplotlib.pyplot as plt
#sys.path.insert(0, ".")
from src.myutils import *
from src.dataset import *
from src.model import SE3Transformer
import src.featurize as featurize

import multiprocessing as mp

MYPATH = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def read_options():
    import argparse
    parser = argparse.ArgumentParser()

    #required
    parser.add_argument("inputpath",
                        help='inputpath where pdbs are located')

    #optional
    parser.add_argument("-p",default="LG.params",
                        help='ligand params file located at inputpath')
    
    parser.add_argument("--mol2",default="ligand.mol2",
                        help='ligand mol2 file located at inputpath')
    
    parser.add_argument("--ncore",default=4,type=int,
                        help='num cpu cores to use for featurization')
    
    parser.add_argument('--root',default=MYPATH,
                        help='path to DANligand')
    
    parser.add_argument("--modelname","-m",default="flexGR3comm4",
                        help="DANligand model param name")
    
    parser.add_argument("--tag",default="run",help="temporary prefix to feature npz")
    
    parser.add_argument('--output','-o',default=None,
                        help='output npz file name')
    
    parser.add_argument('--extrapath', default='',
                        help='path to extra residue params files')
    
    parser.add_argument('--nbatch','-b', default=1,type=int,
                        help='number of batches')
    
    parser.add_argument('--screen', default=False, action="store_true",
                        help='screen through multiple ligands in the mol2file (after -m argument)')
    
    parser.add_argument('--debug',default=False, action="store_true",
                        help='debug option')
    
    option = parser.parse_args()
    if not option.inputpath.endswith('/'): option.inputpath += '/'
    
    # get lig residue name from ligand params
    '''
    l = open(option.inputpath+'/'+option.p).readlines()[0]
    if not l.startswith('NAME'): sys.exit("Error occurred processing ligand params %s... exit."%(option.p))
    option.ligname = l[:-1].split()[1]
    '''

    return option

def featurize_if_missing(option):
    current = os.getcwd()
    os.chdir(option.inputpath)
    print(option.inputpath)
    #if os.path.exists('%s.features.npz'%option.tag):
    #    print("Feature file exists... skip")
    #else:
    npzs = featurize.main(option.tag,verbose=True,decoytypes=[''], 
                          inputpath = './',
                          outpath = './',
                          store_npz=True,
                          same_answer=False,
                          debug=option.debug,
                          nper_npz=50)
    print("features stored at %d npz files: "%len(npzs), npzs)
    os.chdir(current)
    return npzs

def main(option):
    t0 = time.time()
    npzs = featurize_if_missing(option)
    
    t1 = time.time()
    model = SE3Transformer()
    model.to(device)

    checkpoint = torch.load(join("%s/models"%option.root, option.modelname, "best.pkl"),
                            map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    params_loader = {'shuffle': False, 'num_workers': option.ncore,
                     'pin_memory': True, 'collate_fn': collate, 'batch_size': option.nbatch}

    t2 = time.time()
    print("Featurization/Model loading in %.1f/%.1f secs"%(t1-t0,t2-t1))
    
    with torch.no_grad(): # without tracking gradients
        for npz in npzs:
            val_set = Dataset(npz, root_dir=option.inputpath)
            valid_generator = data.DataLoader(val_set, **params_loader)
        
            pnames = []
            n = 0
            for i, (G_bnd, G_atm, G_res, info) in enumerate(valid_generator):
                if not G_bnd:
                    print("skip %s %s"%(info[0]['pname'],info[0]['sname']))
                    continue
                pname = info[0]["pname"]
                sname = info[0]["sname"]
                pindex = info[0]["pindex"]
            
                if pname not in pnames:
                    pnames.append(pname)

                idx = {}
                idx['ligidx'] = info[0]['ligidx'].to(device)
                idx['r2a'] = info[0]['r2amap'].to(device)
                idx['repsatm_idx'] = info[0]['repsatm_idx'].to(device)
                fnat = info[0]['fnat'].to(device)
                lddt = info[0]['lddt'].to(device)[None,:]

                pred_fnat,pred_lddt = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), idx)
                
                if pname not in pnames: pnames.append(pname)
                print("%4s %4d %15s %6.3f %6.3f %6.3f"%(pname,pnames.index(pname),sname,
                                                        float(fnat),float(pred_fnat),
                                                        abs(float(fnat)-float(pred_fnat))))
                n += 1
    t3 = time.time()
    print("Processed %d samples, total elapsed time: %.1f secs"%(n,t3-t2))

if __name__ == "__main__":
    option = read_options()
    main(option)
