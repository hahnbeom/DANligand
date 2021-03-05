#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import numpy as np
import torch

import dgl
from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling, G1x1SE3
from equivariant_attention.fibers import Fiber
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
import matplotlib.pyplot as plt

import os

sys.path.insert(0, ".")
from PALETTE import *

params_loader = {
          'shuffle': True,
          'num_workers': 4,
          'pin_memory': True,
          'collate_fn': collate,
          'batch_size': 1}

k = 16

# Use top 16 for both networks
f = lambda x:get_dist_neighbors(x, top_k=k)
train_set = Dataset(np.load("data/train_proteins.npy"), f, f)
train_generator = data.DataLoader(train_set, **params_loader)

f = lambda x:get_dist_neighbors(x, top_k=k)
val_set = Dataset(np.load("data/valid_proteins.npy"), f, f)
valid_generator = data.DataLoader(val_set, **params_loader)

class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, 
             num_layers       = 4, 
             l0_features      = (82, 60), 
             num_degrees      = 2,
             num_channels     = (64, 64),
             edge_features    = (5, 5),
             div              = (4, 4) ,
             n_heads          = (4, 4),
             global_att_size  = 16,
             **kwargs):
        super().__init__()

        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.edge_features = edge_features
        self.div = div
        self.n_heads = n_heads
        self.num_degrees = num_degrees
        self.global_att_size = global_att_size

        # Linear projection layers for ha and ca
        self.linear1_ha = nn.Linear(l0_features[0], l0_features[0])
        self.linear2_ha = nn.Linear(l0_features[0], self.num_channels[0])
        
        self.linear1_ca = nn.Linear(l0_features[1], l0_features[1])
        self.linear2_ca = nn.Linear(l0_features[1], self.num_channels[1])

        # Define fibers
        self.fibers_ha = {'in': Fiber(1, self.num_channels[0]),
                          'mid1': Fiber(self.num_degrees, self.num_channels[0]),
                          'mid2': Fiber(self.num_degrees, self.num_channels[0]-self.global_att_size),
                          'out': Fiber(1, 1)}
        
        self.fibers_ca = {'in': Fiber(1, self.num_channels[1]),
                          'mid1': Fiber(self.num_degrees, self.num_channels[1]),
                          'mid2': Fiber(self.num_degrees, self.num_channels[1]),
                          'out': Fiber(1, 1)}

        # Build graph convolution net
        self.Gblock_ha = self._build_gcn(self.fibers_ha, 0)
        self.Gblock_ca = self._build_gcn(self.fibers_ca, 1)
        
    def _build_gcn(self, fibers, g_index):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(
                fin, 
                fibers['mid1'], 
                edge_dim=self.edge_features[g_index], 
                div=self.div[g_index], 
                n_heads=self.n_heads[g_index]))
            
            Gblock.append(GNormSE3(fibers['mid1']))
            
            # This might not be needed for the Ca network.
            Gblock.append(G1x1SE3(fibers['mid1'], fibers['mid2']))
            
            # 
            fin = fibers['mid1']
        Gblock.append(GConvSE3(fibers['mid1'], fibers['out'], self_interaction=True, edge_dim=self.edge_features[g_index]))

        return nn.ModuleList(Gblock)

    def forward(self, G_ha, G_ca):
        
        # Gradient checkpointing
        from torch.utils.checkpoint import checkpoint
        def runlayer(layer, G, r, basis):
            def custom_forward(*h):
                hd = {str(i):h_i for i,h_i in enumerate(h)}
                hd = layer(hd, G=G, r=r, basis=basis)
                h = tuple(hd[str(i)] for i in range(len(hd)))
                return (h)
            return custom_forward

        # Compute equivariant weight basis from relative positions
        basis_ha, r_ha = get_basis_and_r(G_ha, self.num_degrees-1)
        basis_ca, r_ca = get_basis_and_r(G_ca, self.num_degrees-1)

        # linear layers to condense to #channels
        l0_ha = F.elu(self.linear1_ha(G_ha.ndata['0'].squeeze()))
        l0_ha = self.linear2_ha(l0_ha).unsqueeze(2)
        h_ha = [l0_ha]
        
        l0_ca = F.elu(self.linear1_ca(G_ca.ndata['0'].squeeze()))
        l0_ca = self.linear2_ca(l0_ca).unsqueeze(2)
        h_ca = [l0_ca]
        
        count = 0
        attmaps = []
        for layer_ha, layer_ca in zip(self.Gblock_ha, self.Gblock_ca):
            count += 1
            h_ha = checkpoint(runlayer(layer_ha, G_ha, r_ha, basis_ha), *h_ha)
            h_ca = checkpoint(runlayer(layer_ca, G_ca, r_ca, basis_ca), *h_ca)
            
            # Do attention after every SE3 norm layer.
            # We also might just want to attend over l0 not l1.
            if count %3 == 0:
                l0, attmap1 = self.attend_with_scaled_dot(h_ha[0][:,:self.global_att_size],
                                                 h_ca[0][:,:self.global_att_size],
                                                 h_ca[0][:,self.global_att_size:2*self.global_att_size])
                attmaps.append(attmap1)
                
                l1, attmap2 = self.attend_with_scaled_dot(h_ha[1][:,:self.global_att_size],
                                                 h_ca[1][:,:self.global_att_size],
                                                 h_ca[1][:,self.global_att_size:2*self.global_att_size])
                attmaps.append(attmap2)
                
                h_ha = torch.cat([h_ha[0], l0], axis=1), torch.cat([h_ha[1], l1], axis=1)
            
        h_ha = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h_ha)}
        h_ca = {str(i):h_i.requires_grad_(True) for i,h_i in enumerate(h_ca)}
        
        return h_ha, h_ca, attmaps
    
    def attend_with_scaled_dot(self, queries, keys, values):
        queries = torch.flatten(queries, start_dim=-2, end_dim=-1)
        keys = torch.flatten(keys, start_dim=-2, end_dim=-1).T
        
        # Scaled dot attention weights
        QK = torch.softmax(torch.matmul(queries, keys)/np.sqrt(self.global_att_size), dim=-1)
        
        QKV = torch.matmul(QK, values.view(values.shape[0], -1))
        QKV = QKV.view(QKV.shape[0], -1, values.shape[-1])
        
        return QKV, QK


# In[6]:


# Instantiate the model

model = SE3Transformer()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
count_parameters(model)/1e6


# In[ ]:


modelname = "PALETTE_ver001"

c = 0
base_learning_rate = 1e-4*10
decay = 0.995
max_epochs = 1000
silent = False
batchsize = 64

if isdir(join("models", modelname)): 
    if not silent: print("Loading a checkpoint")
    checkpoint = torch.load(join("models", modelname, "best.pkl"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]+1
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    if not silent: print("Restarting at epoch", epoch)
    assert(len(train_loss["total"]) == epoch)
    assert(len(valid_loss["total"]) == epoch)
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
        
        
start_epoch = epoch
for epoch in range(start_epoch, max_epochs):  
    
    lr = base_learning_rate*np.power(decay, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Go through samples (batch size 1)
    b_count=0
    temp_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
    for G_ha, G_ca in train_generator:
            
        mask = [int(i)==1 for i in G_ha.ndata['mask']]
        
        # Get prediction and target value
        pred_ha, pred_ca, attmaps = model(G_ha.to(device), G_ca.to(device))
        prediction = pred_ha['0'][:,0,0]
        truth = G_ha.ndata["lddt"]
        
        prediction = prediction[mask]
        truth = truth[mask]
        
        Loss = torch.nn.MSELoss()            
        loss = Loss(prediction, truth.float())
        
        loss.backward(retain_graph=True)
        
        # Only update after certain number of accululations.
        if (b_count+1)%batchsize == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        temp_loss["total"].append(loss.cpu().detach().numpy())
        
        c+=1
        b_count+=1
        sys.stdout.write("\rEpoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %.2f"
                         %(modelname, epoch, max_epochs, b_count, len(train_generator), temp_loss["total"][-1]))
    
    # Empty the grad anyways
    optimizer.zero_grad()
        
    train_loss["total"].append(np.array(temp_loss["total"]))
    
    b_count=0
    temp_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
    with torch.no_grad(): # wihout tracking gradients
        
        # Loop over validation 10 times to get stable evaluation
        for G_ha, G_ca in valid_generator:

            optimizer.zero_grad()

            mask = [int(i)==1 for i in G_ha.ndata['mask']]

            # Get prediction and target value
            pred_ha, pred_ca, attmaps = model(G_ha.to(device), G_ca.to(device))
            prediction = pred_ha['0'][:,0,0]
            truth = G_ha.ndata["lddt"]

            prediction = prediction[mask]
            truth = truth[mask]

            Loss = torch.nn.MSELoss()            
            loss = Loss(prediction, truth.float())

            temp_loss["total"].append(loss.cpu().detach().numpy())

            b_count+=1
            
        valid_loss["total"].append(np.array(temp_loss["total"]))
        
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





# In[ ]:




