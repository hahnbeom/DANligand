#!/usr/bin/env python
import sys
import os
import numpy as np
import torch

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

sys.path.insert(0, ".")
from deepAccNet_cross import *
from deepAccNet_cross.dataset import Dataset
from deepAccNet_cross.model import SE3Transformer

# trace anomal gradients
#torch.autograd.set_detect_anomaly(True)

HYPERPARAMS = {
    "modelname" : "crossMSEduo",
    #"modelname" : "crossTrfL4Slow",
    "transfer"   : True,
    "base_learning_rate" : 1e-4,
    "gradient_accum_step" : 10,
    "TOPK"      : 12,
    "RANDOMIZE" : 0.0, # Ang
    "NUM_LAYERS": 4,
    "BALLDIST"  : 10.0,
    "N_L0"      : 65+28+2, #natype+naatype+charge+sasa
    "w_lossG"   : 0.5, #global
    "w_lossL"   : 0.3, #per-atm
    "w_lossVS"  : 0.2, #classification
    "w_reg"     : 0.00001 # loss ~0.05~0.1
}

def upsample1(fnat):
    over06 = fnat>0.6
    over07 = fnat>0.7
    over08 = fnat>0.8
    p = over06 + over07 + over08 + 1.0 #weight of 1,2,3,4
    return p/np.sum(p)

def upsample2(fnat):
    over08 = fnat>0.8
    p = over08 + 0.01
    return p/np.sum(p)

def upsampleX(fnat):
    over08 = fnat>0.8
    over07 = fnat>0.7
    under01 = fnat<0.8
    p = over08 + over07 + under01 + 0.01
    return p/np.sum(p)

def load_dataset():
    ## setup Virtual Screening (VS) generators
    f = lambda x:get_dist_neighbors(x, top_k=HYPERPARAMS["TOPK"])
    val_set = Dataset(np.load("data/valid_proteins5.npy"), f,
                      root_dir="/projects/ml/ligands/v4.reps/",
                      tag_substr=['rigid','flex'])
    cross_set_valid = Dataset(np.load("data/validVS_proteins5.npy"), f,
                              root_dir="/projects/ml/ligands/v4.reps/",
                              randomize=HYPERPARAMS["RANDOMIZE"], stack=True, nsamples_per_p = 100)
    
    # Instantiating a dataloader
    params_loader = {
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True,
        'collate_fn': collate,
        'batch_size': 1,
    }
    
    valid_generator = data.DataLoader(val_set,
                                      worker_init_fn=lambda _: np.random.seed(),
                                      **params_loader)
    validVS_generator = data.DataLoader(cross_set_valid,
                                        worker_init_fn=lambda _: np.random.seed(),
                                        **params_loader)
    
    return (None,valid_generator,None,validVS_generator)

def load_model(silent=False):
    modelname = HYPERPARAMS['modelname']
    base_learning_rate = HYPERPARAMS['base_learning_rate']
    transfer = HYPERPARAMS['transfer']
    
    model = SE3Transformer(
        num_layers      = HYPERPARAMS['NUM_LAYERS'], 
        l0_in_features  = HYPERPARAMS['N_L0'],
        l1_in_features  = 1,
        num_degrees     = 2, #3
        num_channels    = 32, #32
        edge_features   = 2,
        div             = 2, #TFN
        n_heads         = 2, #TFN
        pooling         = "avg",
        chkpoint        = True,
    )
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
        #print(train_loss["total"], len(train_loss["total"]))
        #assert(len(train_loss["total"]) == epoch)
        #assert(len(valid_loss["total"]) == epoch)
        
    elif transfer and (os.path.exists('models/%s/start.pkl'%(modelname))):
        checkpoint = torch.load(join("models", modelname, "start.pkl"))
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = 0
        train_loss = {"total":[], "global":[], "local":[], "VS":[], "reg":[]}
        valid_loss = {"total":[], "global":[], "local":[], "VS":[]}
        
    else:
        if not silent: print("Training a new model")
        epoch = 0
        train_loss = {"total":[], "global":[], "local":[], "VS":[], "reg":[]}
        valid_loss = {"total":[], "global":[], "local":[], "VS":[]}
        if not isdir(join("models", modelname)):
            if not silent: print("Creating a new dir at", join("models", modelname))
            os.mkdir(join("models", modelname))
    
    return epoch, model, optimizer, train_loss, valid_loss

def validation(model, generator, kernels, header=""):
    
    for i, (G, lddt, fnat, info) in enumerate(generator):
        # Get prediction and target value
        if not info[0]['stat'] or not G:
            print("skip %s %s"%(info[0]['pname'],info[0]['sname']))
            continue
            
        idx = info[0]['ligidx'].to(device)
        pred_fnat,pred_lddt = model(G.to(device), idx)
        fnat = fnat.to(device)[:, None]
        lddt = lddt.to(device)
        
        if lddt.size() != pred_lddt.size(): continue
        pred_fnat = kernels[0](pred_fnat).to(device)
        pred_lddt = kernels[1](pred_lddt).to(device)

        print("%-10s %-10s %3d %6.3f %6.3f"%(info[0]['pname'],info[0]['sname'],info[0]['pindex'],
                                             float(fnat[0]),float(pred_fnat)))

    
def enumerate_an_epoch(model, optimizer, generator,
                       kernels, w_loss,
                       is_training, header=""):    
    temp_loss = {"total":[], "global":[], "local":[], "reg":[]}
    b_count=0
    w_reg = HYPERPARAMS['w_reg']
    gradient_accum_step = HYPERPARAMS['gradient_accum_step']

    for i, (G, lddt, fnat, info) in enumerate(generator): 
        # Get prediction and target value
        if not info[0]['stat'] or not G:
            print("skip %s %s"%(info[0]['pname'],info[0]['sname']))
            continue
            
        idx = info[0]['ligidx'].to(device)
        pred_fnat,pred_lddt = model(G.to(device), idx)
        fnat = fnat.to(device)[:, None]
        lddt = lddt.to(device)

        if lddt.size() != pred_lddt.size(): continue
        Loss = torch.nn.MSELoss()

        pred_fnat = kernels[0](pred_fnat).to(device)
        pred_lddt = kernels[1](pred_lddt).to(device)

        loss1 = Loss(pred_fnat, fnat.float())
        loss2 = Loss(pred_lddt, lddt.float())
        loss = w_loss[0]*loss1 + w_loss[1]*loss2

        print("%-10s %-10s %3d %6.3f %6.3f %6.3f %6.3f"%(info[0]['pname'],info[0]['sname'],info[0]['pindex'],
                                                         float(loss),float(loss1),float(fnat[0]),float(pred_fnat)))

        if is_training:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters(): l2_reg += torch.norm(param)
            loss +=  w_reg*l2_reg

            if not np.isnan(loss.cpu().detach().numpy()):
                loss.backward(retain_graph=True)
            else:
                print("nan loss encountered", prediction.float(), fnat)
            temp_loss["reg"].append(l2_reg.cpu().detach().numpy())

            if i%gradient_accum_step == gradient_accum_step-1:
                optimizer.step()
                optimizer.zero_grad()
        
        b_count+=1
        temp_loss["global"].append(loss1.cpu().detach().numpy())
        temp_loss["local"].append(loss2.cpu().detach().numpy())
        temp_loss["total"].append(loss.cpu().detach().numpy()) # append only

        if header != "":
            sys.stdout.write("\r%s, Batch: [%2d/%2d], loss: %.2f"%(header,b_count,len(generator),temp_loss["total"][-1]))
            
    return temp_loss

def main():
    start_epoch,model,optimizer,train_loss,valid_loss = load_model()
    model.to(device)

    f = lambda x:get_dist_neighbors(x, top_k=HYPERPARAMS["TOPK"])

    def identity(x):
        return x
    def logistic(x):
        return 1.0/(1.0+torch.exp(-20*(x-0.5)))
    
    kernels_QA = [identity,identity] #global/local
    kernels_VS = [logistic,identity] #global/local
    w_QA = (HYPERPARAMS['w_lossG'],HYPERPARAMS['w_lossL'])
    w_VS = (HYPERPARAMS['w_lossVS'],0.0)
    balldist = HYPERPARAMS['BALLDIST']
    randomize = HYPERPARAMS['RANDOMIZE']
    
    # validation
    #proteins = np.load("data/validVS_proteins5.npy")
    if sys.argv[1] == '-s':
        proteins = [sys.argv[2]]
    elif sys.argv[1] == '-l':
        proteins = [l[:-1] for l in open(sys.argv[2])]

    if len(sys.argv) > 3:
        out = open(sys.argv[3],'w')
    else:
        out = sys.stdout
        
    feature_path = '/net/scratch/hpark/VSset/features.test'
    #feature_path = '/net/scratch/hpark/VSset/vina/features.test'
    for pname in proteins:
        prop    = np.load("%s/%s.prop.npz"%(feature_path,pname))
        samples = np.load("%s/%s.lig.npz"%(feature_path,pname),allow_pickle=True)
        nsample = len(samples['name'])
        
        for pindex in range(nsample):
            info = {}
            G, lddt, fnat = graph_from_npz(prop, samples, pindex, info,
                                           "sasa", balldist, f, randomize)
            if not G: continue

            idx = info['ligidx'].to(device)
            pred_fnat,pred_lddt = model(G.to(device), idx)
            pred_fnat = kernels_VS[0](pred_fnat)
        
            #print(len(lddt),pred_lddt.size())
            #if len(lddt) != pred_lddt.size(): continue
            #pred_lddt = kernels_VS[1](pred_lddt)
            
            out.write("%-10s %-10s %3d %6.3f %6.3f\n"%(pname,samples['name'][pindex],pindex,
                                                       fnat,float(pred_fnat)))
                

if __name__ == "__main__":
    main()
