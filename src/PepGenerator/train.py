import numpy as np
import torch
import sys,os
import torch.nn.functional as F
from torch import nn
from model import SE3TransformerAutoEncoder
from dataset import DataSet, collate
from torch.utils import data

BATCH_SIZE = 1
T = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
MAXEPOCHS = 100
LR = 5.0e-5

set_params = {'datapath':'/ml/pepbdb/setH/'
              }#use default

model_params = {'num_node_feats': 22, # 20 aas + sep2cen + is_pep
                'num_edge_feats': 2, # distance + seqsep
                'n_layers_encoder': 3,
                'n_layers_decoder': 3,
                'latent_dim': 8,
                }

loader_params = {
    'shuffle': False,
    'num_workers': 5 if '-debug' not in sys.argv else 1,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 1}

def distance_loss(xyz):
    n = xyz.shape[0]
    #guarantee at least 3.8 Ang dev from prv
    dxyz = xyz[1:,:] - xyz[:-1,:]
    dv = torch.sum(dxyz*dxyz,dim=0) - 3.8
    loss = torch.sum(dv*dv)/(n-1)

    return loss

def run_an_epoch(loader,model,optimizer,epoch,train):
    temp_loss = {'total':[]}
    
    for i, (G, info) in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue

        G = G.to(device)
        dXpred = model(G)
        dX = info['xyz_lig'].to(device)
        pepidx = info['pepidx'] #.to(device)
        loss = torch.tensor(0.0).to(device)

        # iter through batch dim
        for a,p,idx in zip(dX,dXpred,pepidx):
            loss1 = F.mse_loss( a, p[idx,:] )
            loss2 = 0.0*distance_loss(p[idx,:])
            loss = loss + loss1 + loss2
            '''
            form = "%3d %8.3f %8.3f %8.3f : %8.3f %8.3f %8.3f, loss %8.5f %8.5f"
            for i in range(len(a)):
                print(form%(i,a[i,0],a[i,1],a[i,2],
                            p[idx,:][i,0],p[idx,:][i,1],p[idx,:][i,2],
                            float(loss1),float(loss2)))
            '''
        
        if train:
            loss.backward()
            optimizer.step()
            print(f"TRAIN Epoch {epoch} | Loss: {loss.item()} ")
        else:
            print(f"VALID Epoch {epoch} | Loss: {loss.item()} ")

        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
    return temp_loss
                
def load_model(modelname):
    model = SE3TransformerAutoEncoder( **model_params )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    
    train_loss = {'total':[]}
    valid_loss = {'total':[]}
    epoch = 0

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR)
    
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
                    
    model.to(device)
    return model, optimizer, epoch, train_loss, valid_loss

def main():    
    train_set = DataSet(np.load("data/trainlist.H.npy"), **set_params)
    train_loader = data.DataLoader(train_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    valid_set = DataSet(np.load("data/validlist.H.npy")[:10], **set_params)
    valid_loader = data.DataLoader(valid_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    modelname = sys.argv[1]
    model, optimizer, start_epoch, train_loss, valid_loss = load_model( modelname )

    for epoch in range(start_epoch,MAXEPOCHS):
        temp_loss = run_an_epoch(train_loader, model, optimizer, epoch, True)
        train_loss['total'].append(temp_loss['total'])

        with torch.no_grad():
            temp_loss = run_an_epoch(valid_loader, model, optimizer, epoch, False)
            valid_loss['total'].append(temp_loss['total'])
            
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
