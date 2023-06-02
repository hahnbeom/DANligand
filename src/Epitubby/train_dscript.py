import numpy as np
import torch
import sys,os
import torch.nn.functional as F
from torch import nn
from src.model import DSCRIPTModel
from src.dataset import DataSet, collate
from torch.utils import data

torch.set_printoptions(precision=3,sci_mode=False)

BATCH_SIZE = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
MAXEPOCHS = 150
LR = 1.0e-4
W_REG = 0.0001
W_FP = 1.0
DEBUG = '-debug' in sys.argv

set_params = {'datapath':'/ml/motifnet/HmapPPDB/trainable/',
              'debug'   : DEBUG
              }#use default

model_params = {'num_node_feats': 24+1024, # 20 aas + 4 pos-encode + Embedding_s(1024)
                }

loader_params = {
    'shuffle': False,
    'num_workers': 5 if DEBUG else 1,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': BATCH_SIZE}

def custom_loss( pred, label ):
    entropy1 = torch.tensor(0.0).to(device)
    entropy2 = torch.tensor(0.0).to(device)
    
    for p,lidx in zip(pred,label):
        lidx = lidx.to(device)
        l = torch.zeros(p.shape[0]).to(device)
        l[lidx] = 1.0
        
        entropy1 += -torch.sum(l*torch.log(p+1.0e-6))
        # penalize false positives
        entropy2 = -torch.sum((1.0-l)*torch.log(1.0-p+1.0e-6))
    
    return entropy1, entropy2

def run_an_epoch(loader,model,optimizer,epoch,train):
    temp_loss = {'total':[],'loss1':[],'loss2':[]}
    
    for i, (x_frag, G_rec, info) in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G_rec or G_rec == None: continue

        x_frag = x_frag.to(device)
        G_rec = G_rec.to(device)
        
        pred = model(x_frag, G_rec)
        label = info['label']
        
        # weighted cosine similarity
        loss1,loss2 = custom_loss(pred, label)
        loss2 = loss2*W_FP

        # regularization term
        loss_reg = torch.tensor(0.).to(device)
        for param in model.parameters(): loss_reg += torch.norm(param)
        loss_reg = W_REG*loss_reg

        if DEBUG:
            for p,l in zip(pred,label):
                print(torch.sum(l))
        
        loss = loss1 + loss2 + loss_reg
        if train:
            loss.backward()
            optimizer.step()
            print(f"TRAIN Epoch {epoch} | Loss: {loss.item():9.4f} {loss1.item():9.4f} / {loss2.item():9.4f} ")
        else:
            print(f"VALID Epoch {epoch} | Loss: {loss.item():9.4f} {loss1.item():9.4f} / {loss2.item():9.4f} ")

        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["loss1"].append(loss1.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["loss2"].append(loss2.cpu().detach().numpy()) #store as per-sample loss
    return temp_loss
                
def load_model(modelname):
    model = DSCRIPTModel( **model_params )
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
    train_set = DataSet(np.load("data/trainlist.v1.npy"), **set_params)
    train_loader = data.DataLoader(train_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    valid_set = DataSet(np.load("data/validlist.v1.npy"), **set_params, same_frag=True) # to be consistent
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
