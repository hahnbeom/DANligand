import numpy as np
import torch
import sys,os
import torch.nn.functional as F
from torch import nn
from src.model import 
from src.dataset import DataSet, collate
from torch.utils import data

BATCH_SIZE = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
MAXEPOCHS = 100
LR = 1.0e-5
W_REG = 0.0001

set_params = {'datapath':'/ml/'
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

def run_an_epoch(loader,model,optimizer,epoch,train):
    temp_loss = {'total':[],'loss1':[],'loss2':[]}
    
    for i, (seq_Ab, G_Ag, info) in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G_Ag or G_Ag == None: continue

        G_Ag = G_Ag.to(device)
        pred = model(G_Ag, seq_Ab)
        label = info['label'].to(device)
        loss1 = torch.tensor(0.0).to(device)
        loss2 = torch.tensor(0.0).to(device)

        # weighted cosine similarity
        loss1 = custom_loss(pred, label)

        # regularization term
        loss2 = torch.tensor(0.).to(device)
        for param in model.parameters(): loss2 += torch.norm(param)
        loss2 = W_REG*loss2

        if DEBUG:
            for s,l in zip(seq_Ab, label):
                form = "%3d %8.3f %8.3f %8.3f : %8.3f %8.3f %8.3f, loss %8.5f %8.5f"
                for i in range(len(a)):
                    print(form%(i,a[i,0],a[i,1],a[i,2],
                                p[idx,:][i,0],p[idx,:][i,1],p[idx,:][i,2],
                                float(loss1),float(loss2)))
        
        loss = loss1 + loss2
        if train:
            loss.backward()
            optimizer.step()
            print(f"TRAIN Epoch {epoch} | Loss: {loss.item()} ")
        else:
            print(f"VALID Epoch {epoch} | Loss: {loss.item()} ")

        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["loss1"].append(loss1.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["loss2"].append(loss2.cpu().detach().numpy()) #store as per-sample loss
    return temp_loss
                
def load_model(modelname):
    model = MyModel( **model_params )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    
    train_loss = {'total':[],'loss1':[],'loss2':[]}
    valid_loss = {'total':[],'loss1':[],'loss2':[]}
    epoch = 0

    model.to(device)
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
                    
    return model, optimizer, epoch, train_loss, valid_loss

def main():    
    train_set = DataSet(np.load("data/trainlist.npy"), **set_params)
    train_loader = data.DataLoader(train_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    valid_set = DataSet(np.load("data/validlist.npy"), **set_params)
    valid_loader = data.DataLoader(valid_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    modelname = sys.argv[1]
    model, optimizer, start_epoch, train_loss, valid_loss = load_model( modelname )

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
