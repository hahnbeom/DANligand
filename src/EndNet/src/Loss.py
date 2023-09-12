import torch

ScreeningLoss = torch.nn.BCEWithLogitsLoss() 

### loss calculation functions
def grouped_cat(cat):
    import src.src_Grid.motif as motif
    import numpy as np
    
    catG = torch.zeros(cat.shape).to(device)

    # iter over 6 groups
    for k in range(1,7):
        js = np.where(np.array(motif.SIMPLEMOTIFIDX)==k)[0]
        if len(js) > 0:
            a = torch.max(cat[:,js],dim=1)[0]
            if max(a).float() > 0.0:
                for j in js: catG[:,j] = a

    # normalize
    norm = torch.sum(catG, dim=1)[:,None].repeat(1,NTYPES)+1.0e-6
    catG = catG / norm
    
    return catG
            
def MaskedBCE(cats,preds,masks,debug=False):
    device = masks.device
    
    lossC = torch.tensor(0.0).to(device)
    lossG = torch.tensor(0.0).to(device)
    lossR = torch.tensor(0.0).to(device)

    # iter through batches (actually not)
    bygrid = [0.0, 0.0, 0.0]
    for cat,mask,pred in zip(cats,masks,preds):
        # "T": NTYPES; "N": ngrids
        # cat: NxT
        # mask : N
        # pred : NxT
        ngrid = cat.shape[0]
        
        Q = pred[-ngrid:]

        a = -cat*torch.log(Q+1.0e-6) #-PlogQ
        # old -- cated ones still has < 1.0 thus penalized
        #b = -(1.0-cat)*torch.log((1.0-Q)+1.0e-6)
        icat = (cat<0.001).float()
        
        #catG = grouped_cat(cat,device) # no such thing in ligand
        #g = -catG*torch.log(Q+1.0e-6) # on group-cat
        #icatG = (catG<0.001).float()

        # transformed iQ -- 0~0.5->1, drops to 0 as x = 0.5->1.0
        # allows less penalty if x is 0.0 ~ 0.5
        #iQt = -0.5*torch.tanh(5*Q-3.0)+1)
        iQt = 1.0-Q+1.0e-6
        b  = -icat*torch.log(iQt) #penalize if high

        # normalize by num grid points & cat points
        norm = 1.0

        lossC += torch.sum(torch.matmul(mask, a))*norm
        #lossG += torch.sum(torch.matmul(mask, g))*norm
        lossR += torch.sum(torch.matmul(mask, b))*norm


        bygrid[0] += torch.mean(torch.matmul(mask, a)).float()
        #bygrid[1] += torch.mean(torch.matmul(mask, g)).float()
        bygrid[2] += torch.mean(torch.matmul(mask, b)).float()
        
        if debug:
            print("Label/Mask/Ngrid/Norm: %.1f/%d/%d/%.1f"%(float(torch.sum(cat)), int(torch.sum(mask)), ngrid, float(norm)))
    return lossC, lossG, lossR, bygrid

def ContrastLoss(preds,masks):
    loss = torch.tensor(0.0).to(masks.device)
    
    for mask,pred in zip(masks,preds):
        imask = 1.0 - mask
        ngrid = mask.shape[0]
        psum = torch.sum(torch.matmul(imask,pred[-ngrid:]))/ngrid

        loss += psum
    return loss

def structural_loss( Yrec, Ylig, nK ):
    # Yrec: BxKx3 Ylig: K x 3
    dY = Yrec[0,:nK[0],:] - Ylig[0] # hack

    N = 1
    loss1 = torch.sum(dY*dY,dim=0) # sum over K
    loss1_sum = torch.sum(loss1)/N
    
    mae = torch.sum(torch.abs(dY))/nK[0] # this is correct mae...

    return loss1_sum, mae

###
def spread_loss(Ylig, A, grid, nK, sig=2.0): #Ylig:label(B x K x 3), A:attention (B x Nmax x K), grid: B x Nmax x 3
    # actually B == 1
    loss2 = torch.tensor(0.0)

    for b, (x,k) in enumerate( zip(grid, nK) ): #ngrid: B x maxn
        n = x.shape[0]
        #z = A[0,:n,:Ylig.shape[0]] # N x K
        z = A[0,:n,:k] # N x K
        x = x[:,None,:]
        
        dX = x-Ylig
        overlap = torch.exp(-torch.sum(dX*dX,axis=-1)/(sig*sig)) # N x K
        if z.shape[0] != overlap.shape[0]: continue

        loss2 = loss2 - torch.sum(overlap*z)

    #loss2 = -torch.sum(overlap*z)

    return loss2 # max -(batch_size x K)

# penalty 
#Ylig:label(B x K x 3), A:attention (B x Nmax x K), grid: B x Nmax x 3
def spread_loss2(Ylig, A, grid, nK, sig=2.0): 
    # actually B == 1
    loss2 = torch.tensor(0.0)

    for b, (x,k) in enumerate( zip(grid, nK) ): #ngrid: B x maxn
        n = x.shape[0]
        z = A[0,:n,:k] # N x K
        x = x[:,None,:] # N x 1 x 3
        
        dX = (x-Ylig)/sig # N x K x 3
        dev = torch.sum(dX*dX, axis=-1) # N x K
        
        if z.shape[0] != dev.shape[0]: continue

        loss2 = loss2 + torch.sum(dev*z)

    return loss2 # max -(batch_size x K)

'''
def spread_loss( Ylig, A, grid, nK, sig=2.0): #key(B x K x 3), attention (B x Nmax x K)
    loss2 = torch.tensor(0.0)
    i = 0
    N = A.shape[0]
    
    for b, (x,k,y) in enumerate( zip(grid, nK, Ylig) ): #ngrid: B x maxn
        n = x.shape[0]
        z = A[b,:n,:k] # Nmax x Kmax

        dX = x[:,None,:] - y[None,:k,:]
        
        overlap = torch.exp(-torch.sum(dX*dX,axis=-1)/(sig*sig)) # N x K
        if z.shape[0] != overlap.shape[0]: continue

        loss2 = loss2 - torch.sum(overlap*z)

        i += n
    return loss2 # max -(batch_size x K)
'''

