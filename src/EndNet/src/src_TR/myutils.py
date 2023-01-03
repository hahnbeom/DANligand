import torch
import numpy as np
#import matplotlib.pyplot as plt

def rmsd(Y,Yp): # Yp: require_grads
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    Y = Y - Y.mean(axis=0)
    Yp = Yp - Yp.mean(axis=0)

    # Computation of the covariance matrix
    # put a little bit of noise to Y
    C = torch.mm(Y.T, Yp)
    
    # Computate optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign( det(V)*det(W) ) to ensure right-handedness
    d = torch.ones([3,3]).to(device)
    d[:,-1] = torch.sign(torch.det(V) * torch.det(W))
    
    # Rotation matrix U
    U = torch.mm(d*V, W.T)

    rY = torch.einsum('ij,jk->ik', Y, U)
    dY = torch.sum( torch.square(Yp-rY), axis=1 )

    rms = torch.sqrt( torch.sum(dY) / Yp.shape[0] )
    return rms, U

def make_pdb(atms,xyz,outf,header=""):
    out = open(outf,'w')
    if header != "":
        out.write(header)
        
    #ATOM      1  N   VAL A  33     -15.268  78.177  37.050  1.00 92.09      A    N
    form = "HETATM %5d %-3s UNK X %3d   %8.3f %8.3f %8.3f 1.00  0.00\n"
    for i,(atm,x) in enumerate(zip(atms,xyz)):
        #out.write("%-3s  %8.3f %8.3f %8.3f\n"%(atm,x[0],x[1],x[2]))
        out.write(form%(i,atm,1,x[0],x[1],x[2]))

    if header != "":
        out.write("ENDMDL\n")
    out.close()

def generate_pose(Y, keyidx, xyzfull, atms=[], epoch=0, report=False):
    make_pdb(atms,xyzfull,"init.pdb")
    Yp = xyzfull[keyidx]
    # find rotation matrix mapping Y to Yp

    T = torch.mean(Yp - Y, dim=0)

    com = torch.mean(Yp,dim=0)
    rms,U = rmsd(Y,Yp)
    
    Z = xyzfull - com
    T = torch.mean(Y - Yp, dim=0) + com
    
    Z = torch.einsum( 'ij,jk -> ik', Z, U.T) + T

    outf = "epoch%d.pdb"%epoch
    if report: make_pdb(atms,Z,outf,header="MODEL %d\n"%epoch)
    


def report_attention(grids, A, epoch, modelname):
    K=A.shape[1]
    print(A.shape)
    print(K)
    form = "HETATM %5d %-3s UNK X %3d    %8.3f%8.3f%8.3f 1.00%6.2f\n"
    #ATOM      1  N   VAL A  33     -15.268  78.177  37.050  1.00 92.09      A    N
    #HETATM     0 O   UNK X   1       0.000   64.000  71.000 1.00  0.00
    for k in range(K):
        out = open("pdbs/attention%d.epoch%02d.pdb"%(k,epoch),'w')
        out.write("MODEL %d\n"%epoch)
        for i,(x,p) in enumerate(zip(grids,A[k])):
            # print("x:",x,'\n','p:',p)
            out.write(form%(i,"O",1,x[0],x[1],x[2],p))
        out.write("ENDMDL\n")
        out.close()
        
    out = open("pdbs//attentionAll.epoch%02d.pdb"%(epoch),'w')
    out.write("MODEL %d\n"%epoch)

    print(grids, '\n', A)
    # print(max(A[:,i]))
    for i,x in enumerate(grids):
        # print(i,x)
        # print((i,"O",1,x[0],x[1],x[2],max(A[:,i])))
        out.write(form%(i,"O",1,x[0],x[1],x[2],max(A[:,i])))
    out.write("ENDMDL\n")
    out.close()

def make_batch_vec(size_vec):
    batch_vector = []
    n=0
    for i in size_vec:
        for k in range(i):
            batch_vector.append(n)
        n+=1

    return torch.tensor(batch_vector)


def show_how_attn_moves(Z, epoch):

    # Z = np.load(npyfile)
    Z = Z[:int(len(Z)/2)]
    nrows, ncols = Z.shape
    X = np.linspace(1, ncols, ncols)
    Y = np.linspace(1, nrows, nrows)
    X,Y = np.meshgrid(X,Y)

    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    
    # Creating plot
    surf = ax.plot_surface(X, Y, Z , cmap='viridis')
    ax.set_zticks([0, 0.0005, 0.05])

    fig.colorbar(surf, shrink=0.6, aspect=8)

    plt.tight_layout()
    # plt.show()
    plt.savefig('../plotpngs/epoch_%d.png'%epoch)

    print('plotpngs/epoch_%d.png with %d points saved'%(epoch,int(len(Z))))



# z = np.load('../fortest.npy')[4]
# show_how_attn_moves(z, epoch=14)
