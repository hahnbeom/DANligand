import numpy as np
import torch
import sys,os
import torch.nn.functional as F
from torch import nn
from src.model import MyModel
from src.dataset import DataSet, collate
from torch.utils import data
import gc

BATCH_SIZE = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)
MAXEPOCHS = 100
LR = 1.0e-4
W_REG = 0.0001
NTYPES = 128
DEBUG = '-debug' in sys.argv

set_params = {'datapath' : '/ml/CSD/set.50k/trainable',
              'ntypes'   : NTYPES,
              'max_nodes': 400,
              'datatype' : 'ZINC',
              'neighmode':'conn', #based on connectivity only 
              'debug'    : DEBUG,
              }#use default

model_params = {'modeltype'     : 'GAT',
                'num_node_feats': 11 + 5 + 1 + 4, #1hot-elemtype(11) + (1hot-Bnd) + (isPredSite) + 1hot-nH
                'num_edge_feats': 10, # conn (5: self,bnd,ang,tors,farther) + bnd order 1hot(5: 0,1,2,3,aro)
                'num_channels'  : 32,
                'n_out_emb'     : 128, # encoded embedding dim
                'num_layers'    : 4,
                'ntypes'        : NTYPES,
                'drop_out'       : 0.1
                }

loader_params = {
    'shuffle': False,
    'num_workers': 10 if DEBUG else 1,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': BATCH_SIZE,
    'worker_init_fn': lambda _: np.random.seed()
}

def run_an_epoch(loader,model,optimizer,epoch,train):
    temp_loss = {'total':[],'loss1':[],'loss2':[]}
    
    for i, (G, label, info) in enumerate(loader):
        
        if train: optimizer.zero_grad()
        if not G or G == None: continue
        optimizer.zero_grad()

        G = G.to(device)
        label = label.to(device)
        
        pred = model(G)
        loss1 = torch.tensor(0.0).to(device)
        loss2 = torch.tensor(0.0).to(device)

        # Category Entropy
        # locally softmaxing, minval(128) = 3.8654; why?
        lossfunc = torch.nn.CrossEntropyLoss()
        loss1 = lossfunc( pred, label )

        # Regularizer to minimize sum(pred)
        loss2 = torch.mean(pred)

        suffix = " per-entry (label;imax;P[lab]): "
        for p,l in zip(pred,label):
            imax = int(torch.argmax(p))
            Plabel = float(p[l])
            suffix += " %4d %4d %5.3f"%(int(l),imax,Plabel)
        
        loss = loss1 + loss2
        if train:
            loss.backward()
            optimizer.step()
            print(f"TRAIN Epoch {epoch} | {i}/{len(loader)} Loss: {loss.item():6.3f}: {loss1.item():6.3f} + {loss2.item():6.3f} {suffix}")
        else:
            print(f"VALID Epoch {epoch} | {i}/{len(loader)} Loss: {loss.item():6.3f}: {loss1.item():6.3f} + {loss2.item():6.3f} {suffix}")

        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["loss1"].append(loss1.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["loss2"].append(loss2.cpu().detach().numpy()) #store as per-sample loss

        if i%10 == 9:
            torch.cuda.empty_cache()
        
    return temp_loss
                
def load_model(modelname):
    model = MyModel( **model_params )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    
    train_loss = {'total':[],'loss1':[],'loss2':[]}
    valid_loss = {'total':[],'loss1':[],'loss2':[]}
    epoch = 0

    model.to(device)
    optimizer   = torch.optim.SGD(model.parameters(), lr=LR)
    
    if os.path.exists("models/%s/model.pkl"%modelname): 
        checkpoint = torch.load("models/"+modelname+"/model.pkl", map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        print("Restarting at epoch", epoch)
    else:
        if not os.path.exists("models/%s"%(modelname)):
            print("Creating a new dir at models/%s"%modelname) 
            os.mkdir("models/"+modelname)
                    
    return model, optimizer, epoch, train_loss, valid_loss

def main():    
    train_set = DataSet(np.load("data/trainlist.npy"), **set_params)
    train_loader = data.DataLoader(train_set, **loader_params)

    valid_set = DataSet(np.load("data/validlist.npy")[:100], **set_params)
    valid_loader = data.DataLoader(valid_set, **loader_params)

    modelname = sys.argv[1]
    model, optimizer, start_epoch, train_loss, valid_loss = load_model( modelname )

    gc.collect()

    for epoch in range(start_epoch,MAXEPOCHS):
        temp_loss = run_an_epoch(train_loader, model, optimizer, epoch, True)
        for key in temp_loss:
            train_loss[key].append(temp_loss[key])

        with torch.no_grad():
            temp_loss = run_an_epoch(valid_loader, model, optimizer, epoch, False)
            for key in temp_loss:
                valid_loss[key].append(temp_loss[key])
            
        print("Epoch %d, train/valid loss: %7.4f %7.4f"%((epoch,
                                                          np.mean(train_loss['total'][-1]),
                                                          np.mean(valid_loss['total'][-1]))))

        # Update the best model if necessary:
        if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss}, "models/"+modelname+"/best.pkl")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss}, "models/"+modelname+"/model.pkl")
        
if __name__ =="__main__":
    main()
