import numpy as np
import torch
import sys,os
import torch.nn.functional as F
from torch import nn
from src.model import MyModel
from src.dataset import DataSet, collate
from torch.utils import data

BATCH_SIZE = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
MAXEPOCHS = 100
LR = 1.0e-4
W_REG = 0.0001
DEBUG = '-debug' in sys.argv
VERBOSE = '-v' in sys.argv

set_params = {'datapath':'/ml/motifnet/HmapPPDB/trainable/',
              'debug'   : DEBUG,
              'same_frag': True
              }#use default

model_params = {'num_node_feats': 24+1024, # 20 aas + 4 pos-encode + Embedding_s(1024)
                'num_edge_feats': 1, # distance 
                'n_layers_rec': 3,
                'n_layers_frag': 3,
                'num_channels'  : 64
                }

loader_params = {
    'shuffle': False if DEBUG else True,
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
        
        entropy1 += -torch.sum(l*torch.log(p+1.0e-8))
        
        # penalize false positives
        entropy2 = -torch.sum((1.0-l)*torch.log(1.0-p+1.0e-8))
    
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
        loss2 = loss2*0.1

        if VERBOSE:
            #HETATM    1  O1  LG1 X   1       0.176   1.258  15.572  1.00 0.00
            form = 'ATOM  %5d  %3s%4s A%4d    %8.3f%8.3f%8.3f  1.00 %4.2f\n'
            b = 0
            for p,l,t,n in zip(pred,label,info['tag'],G_rec.batch_num_nodes()):
                print(torch.mean(p),p)
                out = open(t+'.pdb','w')
                xyz = G_rec.ndata['x'][b:b+n].squeeze()
                for i,(x,p_) in enumerate(zip(xyz,p)):
                    out.write(form%(i,'CA','GLY',i,x[0],x[1],x[2],float(p_)))
                    if i in list(l):
                        out.write(form%(i,'Zn','Zn',i,x[0],x[1],x[2],0.0))
                out.close()
                b += n

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
    valid_set = DataSet(np.load("data/trainlist.v1.npy")[:100], **set_params)
    valid_loader = data.DataLoader(valid_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    modelname = sys.argv[1]
    model, optimizer, start_epoch, train_loss, valid_loss = load_model( modelname )

    with torch.no_grad():
        temp_loss = run_an_epoch(valid_loader, model, optimizer, start_epoch, False)
        for key in temp_loss:
            valid_loss[key].append(temp_loss[key])
            
if __name__ =="__main__":
    main()
