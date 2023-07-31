import torch
import numpy as np
from scipy.spatial.transform import Rotation
#import matplotlib.pyplot as plt

ELEMS = ['Null','H','C','N','O','Cl','F','I','Br','P','S'] #0 index goes to "empty node"

def to_cuda(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, tuple):
        return (to_cuda(v, device) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=device)

def rmsd(Y,Yp): # Yp: require_grads
    device = Y.device
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
    form = "HETATM %5d%-4s UNK X %3d   %8.3f %8.3f %8.3f 1.00  0.00\n"
    for i,(atm,x) in enumerate(zip(atms,xyz)):
        #out.write("%-3s  %8.3f %8.3f %8.3f\n"%(atm,x[0],x[1],x[2]))
        if len(atm) < 4:
            atm = ' '+atm
        else:
            atm = atm[3]+atm[:3]
        out.write(form%(i,atm,1,x[0],x[1],x[2]))

    if header != "":
        out.write("ENDMDL\n")
    out.close()

def generate_pose(Y, keyidx, xyzfull, atms=[], prefix=None):
    Yp = xyzfull[keyidx]
    # find rotation matrix mapping Y to Yp
    T = torch.mean(Yp - Y, dim=0)

    com = torch.mean(Yp,dim=0)
    rms,U = rmsd(Y,Yp)
    
    Z = xyzfull - com
    T = torch.mean(Y - Yp, dim=0) + com
    
    Z = torch.einsum( 'ij,jk -> ik', Z, U.T) + T # aligned xyz

    if prefix != 'None':
        make_pdb(atms,xyzfull,"%s.input.pdb"%prefix)
        make_pdb(atms[keyidx.cpu().detach().numpy()],Y,'%s.predkey.pdb'%prefix)
        make_pdb(atms, Z, "%s.al.pdb"%prefix) #''',header="MODEL %d\n"%epoch''')
        
    return rms, atms[keyidx.cpu().detach().numpy()]

def RandRG(Glig,maxT=2.0):
    com = torch.mean(Glig.ndata['x'],axis=0)

    q = torch.rand(4) # random rotation
    R = torch.tensor(Rotation.from_quat(q).as_matrix()).float()

    t = maxT*(2.0*torch.rand(3)-1.0)
    xyz = torch.matmul(Glig.ndata['x']-com,R) + com + t

    #xyz = xyz.squueze()
    #for x in xyz:
    #    x = x.squeeze()
    #    print('C %8.3f %8.3f %8.3f'%(x[0],x[1],x[2]))    
    Glig.ndata['x'] = xyz
    
    return Glig
    

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

def read_mol2(mol2,drop_H=False):
    read_cont = 0
    qs = []
    elems = []
    xyzs = []
    bonds = []
    borders = []
    atms = []
    
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        if l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            break
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        words = l[:-1].split()
        if read_cont == 1:

            idx = words[0]
            if words[1].startswith('BR'): words[1] = 'Br'
            if words[1].startswith('Br') or  words[1].startswith('Cl') :
                elem = words[1][:2]
            else:
                elem = words[1][0]

            if elem == 'A' or elem == 'B' :
                elem = words[5].split('.')[0]
            
            if elem not in ELEMS: elem = 'Null'
            
            atms.append(words[1])
            elems.append(elem)
            qs.append(float(words[-1]))
            xyzs.append([float(words[2]),float(words[3]),float(words[4])]) 
                
        elif read_cont == 2:
            # if words[3] == 'du' or 'un': rint(mol2)
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'0':0,'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
            borders.append(bondtypes[words[3]])

    nneighs = [[0,0,0,0] for _ in qs]
    for i,j in bonds:
        if elems[i] in ['H','C','N','O']:
            k = ['H','C','N','O'].index(elems[i])
            nneighs[j][k] += 1.0
        if elems[j] in ['H','C','N','O']:
            l = ['H','C','N','O'].index(elems[j])
            nneighs[i][l] += 1.0

    # drop hydrogens
    if drop_H:
        nonHid = [i for i,a in enumerate(elems) if a != 'H']
    else:
        nonHid = [i for i,a in enumerate(elems)]

    borders = [b for b,ij in zip(borders,bonds) if ij[0] in nonHid and ij[1] in nonHid]
    bonds = [[nonHid.index(i),nonHid.index(j)] for i,j in bonds if i in nonHid and j in nonHid]

    return np.array(elems)[nonHid], np.array(qs)[nonHid], bonds, borders, np.array(xyzs)[nonHid], np.array(nneighs,dtype=float)[nonHid], list(np.array(atms)[nonHid])

def read_mol2_batch(mol2,tags_read=None,drop_H=True,tag_only=False):
    read_cont = 0
    qs,elems,xyzs,bonds,borders,atms = {},{},{},{},{},{}

    for l in open(mol2):
        if l.startswith('@<TRIPOS>MOLECULE'):
            read_cont = 3
            # reset
            continue

        if read_cont == 3:
            tag = l[:-1]
            if tags_read == None or tag in tags_read:
                read_cont = 4
                qs[tag] = []
                elems[tag] = []
                xyzs[tag] = []
                bonds[tag] = []
                borders[tag] = []
                atms[tag] = []
            else:
                read_cont = -1
            
        if read_cont < 0 or tag_only: continue
        
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        elif l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        elif l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            break
        elif l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        if read_cont == 1:
            words = l[:-1].split()
            idx = words[0]
            if words[1].startswith('BR'): words[1] = 'Br'
            if words[1].startswith('Br') or  words[1].startswith('Cl') :
                elem = words[1][:2]
            else:
                elem = words[1][0]

            if elem == 'A' or elem == 'B' :
                elem = words[5].split('.')[0]
            
            if elem not in ELEMS: elem = 'Null'
            
            atms[tag].append(words[1])
            elems[tag].append(elem)
            qs[tag].append(float(words[-1]))
            xyzs[tag].append([float(words[2]),float(words[3]),float(words[4])])
                
        elif read_cont == 2:
            words = l[:-1].split()
            bonds[tag].append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'0':0,'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
            borders[tag].append(bondtypes[words[3]])

    tags = list(bonds.keys())
    if tags_read != None:
        tags_order = [tag for tag in tags_read if tag in tags] # reorder following input
    else:
        tags_order = tags

    qs_s, elems_s, xyzs_s, bonds_s, borders_s, atms_s, nneighs_s =[],[],[],[],[],[],[]
    if tag_only:
        return elems_s, qs_s, bonds_s, borders_s, xyzs_s, nneighs_s, atms_s, tags_order
    
    for tag in tags_order:
        nneighs = [[0,0,0,0] for _ in qs[tag]]
        for i,j in bonds[tag]:
            if elems[tag][i] in ['H','C','N','O']:
                k = ['H','C','N','O'].index(elems[tag][i])
                nneighs[j][k] += 1.0
            if elems[tag][j] in ['H','C','N','O']:
                l = ['H','C','N','O'].index(elems[tag][j])
                nneighs[i][l] += 1.0

        # drop hydrogens
        if drop_H:
            nonHid = [i for i,a in enumerate(elems[tag]) if a != 'H']
        else:
            nonHid = [i for i,a in enumerate(elems[tag])]

        _borders = [b for b,ij in zip(borders[tag],bonds[tag]) if ij[0] in nonHid and ij[1] in nonHid]
        _bonds = [[nonHid.index(i),nonHid.index(j)] for i,j in bonds[tag] if i in nonHid and j in nonHid]
                    
        # append to list
        elems_s.append(np.array(elems[tag])[nonHid])
        qs_s.append(np.array(qs[tag])[nonHid])
        xyzs_s.append(np.array(xyzs[tag])[nonHid])
        atms_s.append(np.array(atms[tag])[nonHid])
                
        nneighs_s.append(np.array(nneighs,dtype=float)[nonHid])
        bonds_s.append(_bonds)
        borders_s.append(_borders)
        
    return elems_s, qs_s, bonds_s, borders_s, xyzs_s, nneighs_s, atms_s, tags_order

def read_mol2s_xyzonly(mol2):
    read_cont = 0
    xyzs = []
    atms = []
    
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            xyzs.append([])
            atms.append([])
            continue
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue
        
        if l.startswith('@<TRIPOS>BOND'): 
            read_cont = 2
            continue

        words = l[:-1].split()
        if read_cont == 1:
            is_H = (words[1][0] == 'H')
            if not is_H:
                atms[-1].append(words[1])
                xyzs[-1].append([float(words[2]),float(words[3]),float(words[4])]) 

    return np.array(xyzs), atms



# z = np.load('../fortest.npy')[4]
# show_how_attn_moves(z, epoch=14)
