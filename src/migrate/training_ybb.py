#!/usr/bin/env python
import sys
import os

import numpy as np
import torch

from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from src.myutils import *
from src.dataset import *
from src.model_multiGMM import SE3Transformer
import src.motif as motif
# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

CHECK_EQUIVARIENCE = '-eq' in sys.argv

HYPERPARAMS = {
    "modelname" : sys.argv[1], #"XGrepro2",
    "transfer"   : False, #transfer learning starting from "start.pkl"
    "base_learning_rate" : 1.0e-4, #
    "gradient_accum_step" : 10,
    "max_epochs": 100,
    "w_lossBin"   : 1.0, #motif or not
    "w_lossCat"   : 1.0, #which category 
    "w_lossxyz"   : 0.0, #MSE
    "w_lossrot"   : 1.0, #MSE
    "w_reg"     : 1.0e-6, # loss ~0.05~0.1
    "modeltype" : 'comm',
    'num_layers': (1,1,1),
    'nchannels' : 32, #default 32
    'use_l1'    : 0,
    'nalt_train': 1,
    'drop_out'  : 0.1,
    'setsuffix' : 'v5or',
    #'ansidx'   : list(range(0,14)), #list(range(0,2)), #only Fake/ASP
    'ansidx'   : [int(a) for a in sys.argv[2].split(',')],
    #'hfinal_from': (1,1), #skip-connection, ligres
    'learn_OR'  : 'axis', # [gs/axis]
    'clip_grad' : -1.0, #set < 0 if don't want
    'include_fake': False,
    'bias'      : False,
    'nGMM'      : 3 #num inv rotamers for each 
    #'hfinal_from': (int(sys.argv[2]),int(sys.argv[3])), #skip-connection, ligres
    # only for VS
}

# default setup
set_params = {
    'root_dir'     : "/projects/ml/ligands/motif/backbone/", #let each set get their own...
    'ball_radius'  : 12.0,
    'ballmode'     : 'all',
    'sasa_method'  : 'sasa',
    'edgemode'     : 'distT',
    'edgek'        : (0,0),
    'edgedist'     : (10.0,6.0), 
    'distance_feat': 'std',
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    "CBonly"       : ('-CB' in sys.argv),
    #'aa_as_het'   : True,
    'debug'        : ('-debug' in sys.argv),
    }

# # Instantiating a dataloader
generator_params = {
    'shuffle': False, #True,
    'num_workers': 4,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 1,
}
if set_params['debug']: generator_params['num_workers'] = 1
if CHECK_EQUIVARIENCE: HYPERPARAMS['drop_out'] = 0.0

def rotate_example(G_bnd, G_atm, G_res, R, offset):
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
    G_atm.edata['d'] = torch.einsum('kj,ij->ki', G_atm.edata['d'], R)
    G_res.edata['d'] = torch.einsum('kj,ij->ki', G_res.edata['d'], R)

    #G_atm.ndata['x'] += offset[None, None]
    #G_res.ndata['x'] += offset[None, None]

    # TODO MAKE THIS NOT DUBM
    #G_atm.ndata['x'] *= 0.
    #G_res.ndata['x'] *= 0.
    return G_bnd, G_atm, G_res

def load_model(silent=False):
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    nchannels = HYPERPARAMS['nchannels']

    outtype = 'category'
    if isinstance(HYPERPARAMS['ansidx'],list):
        outtype = HYPERPARAMS['ansidx'] #extension of binary

    # l0 features dropped -- "is_lig"
    model = SE3Transformer(
        num_layers     = HYPERPARAMS['num_layers'], 
        l0_in_features = (65+N_AATYPE+2, N_AATYPE+1, nchannels+nchannels), #no aa-type in atm graph
        l1_in_features = (0,0,HYPERPARAMS['use_l1']),
        num_channels   = (nchannels,nchannels,nchannels),
        modeltype      = HYPERPARAMS['modeltype'],
        nntypes        = ('SE3T','SE3T','SE3T'),
        outtype        = outtype,
        drop_out       = HYPERPARAMS['drop_out'],
        learn_orientation = HYPERPARAMS['learn_OR'],
        #eqtype = HYPERPARAMS['eqtype'], #unused
        #bias = HYPERPARAMS['bias'], #unused
        nGMM = HYPERPARAMS['nGMM']
    )
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
    print("nparams: ", count_parameters(model))

    if os.path.exists('models/%s/best.pkl'%(modelname)):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(join("models", modelname, "best.pkl"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        if not silent: print("Restarting at epoch", epoch)
        
    else:
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

def LossGMM(preds,v0,As,sigs,w_rsr=1.0):
    # multiple Gaussians defined as HYPERPARAMS['nGMM']
    n = HYPERPARAMS['nGMM']
    # all n dimension
    preds = (preds - v0.repeat(1,n,1))/(sigs+0.5) #TODO make sigma range in a reasonable region
    '''
    print("sigma:", sigs)
    print("exponent",preds)
    print("ampls", As)
    print(preds)
    '''
    
    loss = torch.sum(As*torch.exp(preds*preds))

    # make preds at least 3Ang off  from origin
    loss_r = w_rsr*torch.nn.functional.relu(9.0-torch.sum(preds*preds))
    
    # reduce sigma as much
    loss_r += w_rsr*torch.sum(sigs*sigs)
    #print(loss_r)
    loss = loss + loss_r
    return loss

def enumerate_an_epoch(model, optimizer, generator,
                       w_loss, temp_loss, mode='QA',
                       is_training=True, header="",
                       check_equivarience=False):

    if temp_loss == {}: temp_loss = {"total":[],"Cat":[],"Bin":[],'xyz':[],'rot':[],'reg':[]}

    b_count=0
    w_reg = HYPERPARAMS['w_reg']
    gradient_accum_step = HYPERPARAMS['gradient_accum_step']
    ansidx = HYPERPARAMS['ansidx']
        
    # > 0 as motif  
    if isinstance(ansidx,list): #take simpler
        expected_nout = len(ansidx)
    else:
        expected_nout = len(motif.MOTIFS)

    if check_equivarience:
        #generator = generator[:1]
        R, offset = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float), torch.ones([3])
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float)
        Rs, offsets = [torch.eye(3), R], [torch.zeros([3]), torch.zeros([3])]
        
    for i, (G_bnd, G_atm, G_res, info) in enumerate(generator):
        # Get prediction and target value
        if not G_bnd:
            print("skip ", info['pname'],info['sname'])
            continue

        r2a = info['r2a'].to(device)
        motifidx   = info['motifidx'].long() #label
        if motifidx == 0 and not HYPERPARAMS['include_fake']: continue #waste of time?
        if motifidx not in HYPERPARAMS['ansidx']: continue #waste of time?

        Ps_cat = is_same_simpleidx(motifidx,ansidx)
        Ps_cat = torch.transpose(torch.tensor(Ps_cat).repeat(2,1),0,1)
        Ps_cat[:,0] = 1-Ps_cat[:,1]
        
        Ps_bin = [[float(idx==key) for key in ansidx] for idx in motifidx]
        Ps_bin = torch.transpose(torch.tensor(Ps_bin).repeat(2,1),0,1)
        Ps_bin[:,0] = 1-Ps_bin[:,1]

        Ps_cat = Ps_cat.float().to(device)
        Ps_bin = Ps_bin.float().to(device)

        if check_equivarience:
            for R, offset in zip(Rs, offsets):
                G_bnd, G_atm, G_res = rotate_example(G_bnd, G_atm, G_res, R, offset)
                R, offset = R.to(device), offset.to(device)
                
                Ps,dxyz_pred,rot_pred = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), r2a)
                print(f"\trot: {info['rot'][0]}")
                rot_pred_back = torch.einsum('ijk,kl->ijl',rot_pred, R)
                print(f"\trot_pred_back[0]:\n{rot_pred_back[0]}")

                dxyz_pred_back = torch.einsum('jk,kl->jl', dxyz_pred, R)
                print(f"\tdxyz_pred_back[:4]:\n{dxyz_pred_back[:4]}")
        else:
            Ps,dxyz_pred,rot_pred,ampl_pred,sigs_pred = model(G_bnd.to(device), G_atm.to(device), G_res.to(device), r2a)
        # go individually for a hack...
        if len(Ps.shape) < 3 or Ps.shape[1] != expected_nout: continue

        def MyLLloss(P,Q): #Q prediction, P answer
            loss = torch.mean(-P*torch.log(Q)) #mean would better have normalization across setups?
            return loss

        loss1,loss2,loss3,loss4 = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)

        Ps = Ps.to(device)
        loss1 = MyLLloss(Ps_cat,Ps)
        loss2 = MyLLloss(Ps_bin,Ps)

        form1 = " | %-15s %2d"+' |'
        l = form1%tuple([info['sname'][0],int(info['motifidx'])])
        Ps = [int(P[1]) for P in Ps_cat]
        l += ' %1d'*len(Ps)%tuple(Ps)
        Ps = [int(P[1]) for P in Ps_bin]
        l += ' %1d'*len(Ps)%tuple(Ps)

        #try:
        #idx = torch.tensor([HYPERPARAMS['ansidx'].index(motifidx)])
        idx = [HYPERPARAMS['ansidx'].index(motifidx)] #answer idx
        #print(HYPERPARAMS['ansidx'].index(motifidx), motifidx, idx)
        #idx = motifidx
        
        loss3 = torch.tensor(0.0)
        loss4 = torch.tensor(0.0)
        
        if int(motifidx) > 0:
            LossMSE = torch.nn.MSELoss()
            
            dxyz = info['dxyz'].to(device)
            rot  = info['bases'].to(device)[0][1] #get y-axis instead quaternion 'rot'
            
            # or probability weighted predictions -- P*preds?
            dxyz_pred = dxyz_pred[idx].to(device)
            ampl_pred = ampl_pred[idx].to(device)
            sigs_pred = sigs_pred[idx].to(device)
            rot_pred = rot_pred[idx].to(device)
       
            if w_loss[2] > 0:
                loss3 = LossGMM(dxyz_pred,dxyz,ampl_pred,sigs_pred)
            if w_loss[3] > 0:                
                loss4 = motif.LossAxis(rot,rot_pred) # compare simple vector similarity 
            #else ignore
                # measure magnitude
            #l += " %6.2f %6.2f"%(torch.sum(dxyz_pred*dxyz_pred).float(), torch.sum(rot_pred[motifidx]*rot_pred[motifidx]).float())

        loss = w_loss[0]*loss1 + w_loss[1]*loss2 + w_loss[2]*loss3 + w_loss[3]*loss4

        q = rot_pred[0]
        l += " %5.2f %5.2f %5.2f : %5.2f %5.2f %5.2f"%(rot[0],rot[1],rot[2],q[0],q[1],q[2]) 
        l += " | %5.3f (%5.3f %5.3f"%(float(loss),float(loss1),float(loss2))
        l += " %5.3f %5.3f)\n"%(float(loss3),float(loss4))

        ##label, topredict
        #            float(P[0,1]),
        #            float(loss), float(loss1), float(loss2)))
        
        if is_training:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters(): l2_reg += torch.norm(param)
            loss = loss + w_reg*l2_reg

            if not np.isnan(loss.cpu().detach().numpy()):
                loss.backward(retain_graph=True)
            else:
                print("nan loss encountered")
            temp_loss["reg"].append(l2_reg.cpu().detach().numpy())
            
            if i%gradient_accum_step == gradient_accum_step-1:
                if HYPERPARAMS['clip_grad'] > 0.0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), HYPERPARAMS['clip_grad'])

                optimizer.step()
                optimizer.zero_grad()
        
        b_count+=1
        temp_loss["Cat"].append(loss1.cpu().detach().numpy())
        temp_loss["Bin"].append(loss2.cpu().detach().numpy())
        temp_loss["xyz"].append(loss3.cpu().detach().numpy())
        temp_loss["rot"].append(loss4.cpu().detach().numpy())
        temp_loss["total"].append(loss.cpu().detach().numpy()) # append only

        if header != "":
            sys.stdout.write("\r%s, Batch: [%2d/%2d], loss: %8.4f %s"%(header,b_count,len(generator),temp_loss["total"][-1],l))
    return temp_loss
            
def main():
    decay = 0.98
    max_epochs = HYPERPARAMS['max_epochs']
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    
    start_epoch,model,optimizer,train_loss,valid_loss = load_model()

    generators = load_dataset(set_params, generator_params, setsuffix=HYPERPARAMS['setsuffix'])
    train_generator,valid_generator = generators[:2]
    
    w = (HYPERPARAMS['w_lossCat'],HYPERPARAMS['w_lossBin'],
         HYPERPARAMS['w_lossxyz'],HYPERPARAMS['w_lossrot'])
    
    for epoch in range(start_epoch, max_epochs):  
        lr = base_learning_rate*np.power(decay, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        # Go through samples (batch size 1)
        # training
        header = "Epoch(%s): [%2d/%2d] QA"%(modelname, epoch, max_epochs)
        
        temp_loss = {}
        temp_loss = enumerate_an_epoch(model, optimizer, train_generator, 
                                       w, temp_loss, 
                                       is_training=True, header=header,
                                       check_equivarience=CHECK_EQUIVARIENCE)
            
        for key in ['Bin','Cat','xyz','rot','reg']: train_loss[key].append(temp_loss[key])
        train_loss['total'].append(temp_loss['total'])

        # validation
        with torch.no_grad(): # without tracking gradients
            temp_loss = {}
            for i in range(3): # repeat multiple times for stable numbers
                temp_loss = enumerate_an_epoch(model, optimizer, valid_generator, 
                                               w, temp_loss, is_training=False)

            for key in ['Bin','Cat','xyz','rot']: valid_loss[key].append(temp_loss[key])
            valid_loss['total'].append(temp_loss['total'])

        # Update the best model if necessary:
        if epoch == 0 or (np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1])):
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

if __name__ == "__main__":
    main()

