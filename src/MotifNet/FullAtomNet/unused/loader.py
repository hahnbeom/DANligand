import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile

sys.path.insert(0,'/home/hpark/programs/DANse3/migrate/src_general/')
import featurize
import myutils

import torch
from torch.utils import data
import dgl
from .utils import *
from .peratom_lddt import *
from .dataset import *

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,
                 dirs,
                 loader,
                 category           = [],
                 topk               = 16,
                 max_atom_count     = 3000,
                 verbose            = False,
                 native             = False,
                 cutoff             = 15):

        self.datadirs  = dirs
        self.topk      = topk
        self.verbose   = verbose
        self.loader    = loader
        self.category  = category if len(category)==len(dirs) else [""]*len(dirs)
        self.cutoff    = cutoff
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datadirs)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Selecting a sample cluster 
        dirname = self.datadirs[index]
        
        if not self.native:
            samples = [i for i in listdir(dirname) if i.endswith(".pdb")]
            pindex = np.random.choice(np.arange(len(samples)))
            sname = samples[pindex]
            pdbfilename = join(dirname, sname)
        else:
            pdbfilename = join(dirname, "native.pdb")
        nativefilename = join(dirname, "native.pdb")
        
        temp = self.loader(pdbfilename,
                           nativefilename=nativefilename,
                           verbose=self.verbose, 
                           topk=self.topk,
                           mac=self.mac,
                           cutoff=self.cutoff)
        temp.append(self.category[index])
        
        return tuple(temp)

