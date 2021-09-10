# simple_eq_test performs a simple test of SE3 l1 feature equivariance to
import numpy as np
import pickle as pkl
np.random.seed(42)
import random
random.seed(42)

import torch
torch.manual_seed(42)
from torch import nn
# rotation.
#!/usr/bin/env python
import sys
import copy
import os

from scipy.spatial.transform import Rotation

import numpy as np
import torch

sys.path.insert(0, ".")
from src.myutils import *
from src.dataset import *
import src.motif as motif
from src.model_simple_test import SE3Transformer, SE3TransformerJ

import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import dgl.function as dglF

import torch
from torch import nn
from torch.nn import functional as F
from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber

#import Transformer
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

HYPERPARAMS = {
    "modelname" : sys.argv[1], #"XGrepro2",
    "base_learning_rate" : 1.0e-3, #dflt 1e-3
    'num_layers': [1],
    "w_reg"     : 1.0e-6, # loss ~0.05~0.1
    "max_epochs": 100,
    "w_lossBin"   : 1.0, #motif or not
    "w_lossCat"   : 1.0, #which category
    "w_lossxyz"   : 1.0, #MSE
    "w_lossrot"   : 0.0, #MSE

    # misc options below
    "modeltype" : 'comm',
    "gradient_accum_step" : 10,
    'nchannels' : 1, #default 32
    'use_l1'    : 1,
    'setsuffix' : 'v5or',
    'ansidx'   : list(range(0,14)), #[int(word) for word in sys.argv[2].split(',')], #-1 for all-type category prediction
    'learn_OR'  : True,
    "transfer"  : False, #transfer learning starting from "start.pkl"
    'clip_grad' : -1.0, #set < 0 if don't want
}

# default setup
set_params = {
    # IMPORTANT OPTIONS:
    "randomize_lig": 1.0, # Ang, pert the motif coord!
    'root_dir'     : "/projects/ml/ligands/motif/backbone/", #let each set get their own...
    'xyz_as_bb'    : True, #request to predict backbone not the motif

    # default below
    'ball_radius'  : 12.0,
    'ballmode'     : 'all',
    'sasa_method'  : 'sasa',
    'edgemode'     : 'distT',
    'edgek'        : (0,0),
    'edgedist'     : (10.0,6.0),
    'distance_feat': 'std',
    "randomize"    : 0.2, # Ang, pert the rest
    "CBonly"       : False,
    'debug'        : ('-debug' in sys.argv),
    }


# # Instantiating a dataloader
generator_params = {
    'shuffle': True, #True,
    'num_workers': 4,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 1,
}

def load_model(silent=False):
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    nchannels = HYPERPARAMS['nchannels']

    outtype = 'category'
    if isinstance(HYPERPARAMS['ansidx'],list):
        outtype = HYPERPARAMS['ansidx'] #extension of binary

    # l0 features dropped -- "is_lig"
    print("loading model:")
    model = SE3TransformerJ(
            num_layers=1,
            atom_feature_size=101,
            num_channels=1,
            num_degrees=2,
            edge_dim=2,
            div=1,
            n_heads=4)
    #model = SE3Transformer(
    #    num_layers     = HYPERPARAMS['num_layers'],
    #    l0_in_features = [65+N_AATYPE+2], #no aa-type in atm graph
    #    l1_in_features = [0],
    #    num_channels   = [nchannels],
    #    modeltype      = HYPERPARAMS['modeltype'],
    #    #nntypes        = ('SE3T','SE3T','SE3T'),
    #    nntypes        = ['TFN'], # TODO: switch back
    #    outtype        = outtype,
    #    drop_out       = 0.,
    #    learn_orientation = HYPERPARAMS['learn_OR']
    #)
    print("finished loading model")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
    print("nparams: ", count_parameters(model))

    if not silent: print("Training a new model")
    epoch = 0
    train_loss = {"total":[], "Cat":[], "Bin":[], "xyz":[], "rot":[], "reg":[]}
    valid_loss = {"total":[], "Cat":[], "Bin":[], "xyz":[], "rot":[]}
    if not isdir(join("models", modelname)):
        if not silent: print("Creating a new dir at", join("models", modelname))
        os.mkdir(join("models", modelname))

    return epoch, model, optimizer, train_loss, valid_loss

def is_same_simpleidx(label,idxs):
    #print(label,idx)
    return np.array([[float(motif.SIMPLEMOTIFIDX[i]==motif.SIMPLEMOTIFIDX[j]) for i in idxs] for j in label])


def rotate_example(G_bnd, R, offset):
    """rotate_example updates the graphs corresponding to a rotation of the
    global reference frame.

    The underlying graphs are modified.

    Args:
        G_bnd, G_atm, G_res: bond, atom and residue graphs
        R, offset: rotation and translation of the global frame of shapes [3,3]
            and [3]

    Returns:
        updated graphs
    """
    G_bnd.edata['d'] = torch.einsum('kj,ij->ki', G_bnd.edata['d'], R)
    return G_bnd

def main():

    # Load tutorial graph from pickle
    f = "/home/btrippe/G.pkl"
    with open(f,'rb') as f:
        G = pkl.load(f)
    G.ndata['0'] = G.ndata['seq1hot']

    # Load graph from training set
    generators = load_dataset(set_params, generator_params, setsuffix=HYPERPARAMS['setsuffix'])
    train_generator = generators[0]
    # pull out first datapoint from generator.
    for i, (G_bnd_real, G_atm, G_res, info) in enumerate(train_generator):
        if not G_bnd_real:
            print("skip ", info['pname'],info['sname'])
            continue
        else:
            break

    if True:
        G_bnd = G_bnd_real
    else:
        G_bnd = G

    input_channels = G_bnd.ndata['0'].shape[1]
    print("num input channels :", input_channels)

    model = SE3TransformerJ(num_layers=1,
                   #atom_feature_size=101,
                   atom_feature_size=input_channels,
                   num_channels=1,
                   num_degrees=2,
                   edge_dim=2,
                   div=1,
                   n_heads=4)
    model.to(device)


    G_bnds, h_bnds = [], []

    h_bnd = model(G_bnd.to(device))
    G_bnds.append(copy.deepcopy(G_bnd))
    h_bnds.append(h_bnd)
    print(f"h_bnd['1'][:4,0] :\n{h_bnd['1'][:4,0]}")

    def rotate_graph(G, R):
        """rotate graph rotates the all edge displacements defined in a graph
        """
        G.edata['d'] = torch.einsum('kj,ij->ki', G.edata['d'], R)
        return G

    # Rotate and predict again
    R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float)
    G_bnd_rot = rotate_graph(G_bnd.to(device), R.to(device))
    h_bnd_rot  = model(G_bnd_rot.to(device))
    h_bnd1_rot_back = torch.einsum('ijk,kl->ijl',h_bnd_rot['1'], R.to(device))
    print(f"h_bnd1_rot_back[:4,0] :\n{h_bnd1_rot_back[:4,0]}")

    G_bnds.append(copy.deepcopy(G_bnd_rot))
    h_bnds.append(h_bnd_rot)

    import ipdb;ipdb.set_trace()
    return

if __name__ == "__main__":
    main()

