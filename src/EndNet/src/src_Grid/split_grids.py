import numpy as np
import sys
import torch
import scipy
import scipy.spatial
import multiprocessing as mp

def cluster(X,dcut):
    kd = scipy.spatial.cKDTree(X)
    neighs = kd.query_ball_tree(kd, dcut)

    n = X.shape[0]
    neighM = np.zeros((n,n),dtype=int)
    for i,v in enumerate(neighs):
        neighM[i,v] = 1

    ncls,clusters = scipy.sparse.csgraph.connected_components(csgraph=neighM,directed=False,return_labels=True)

    return ncls, clusters

def dynamic_assign_clusters(Xlabel):
    
    dcut = 6.0
    for k in range(8):
        ncls,clusters = cluster(Xlabel,dcut=dcut)
        if ncls < 2:
            dcut -= 1.0
        elif ncls > 3:
            dcut += 1.0
        else:
            break

    ncls,clusters = cluster(Xlabel,dcut=dcut)

    if ncls == 1:
        return False,None,None
        
    #print(Xlabel.shape[0],ncls,k,dcut)
    Xcl = [[] for _ in range(ncls)]
    icl = [[] for _ in range(ncls)]
    for i,clid in enumerate(clusters):
        Xcl[clid].append(Xlabel[i])
        icl[clid].append(i)

    return Xcl,icl,dcut

def main(trg,validate=False):
    dat = np.load('%s.grid.npz'%trg,allow_pickle=True)
    ngrid = dat['labels'].shape[0]
    grids =  dat['xyz']
    labels = dat['labels']
    ilabels = np.where(labels > 0.001)[0]

    Xlabel = grids[ilabels]
    
    # cluster by labeled positions
    Xcl,icl,dcut = dynamic_assign_clusters(Xlabel)

    if not Xcl:
        print("Not enough broken graph found: skip %s"%trg)
        return
    icl = [ilabels[a] for a in icl]
    print(trg,len(Xcl),dcut)
    
    # collect grid points nearby each cluster points
    kd_grid = scipy.spatial.cKDTree(grids)
    for i,X in enumerate(Xcl):
        kd = scipy.spatial.cKDTree(X)
        a = kd.query_ball_tree(kd_grid, dcut)
        neighs = np.unique(np.concatenate(a)).astype(int)

        #combine labels & neighpoints
        idx = np.concatenate([icl[i],[a for a in neighs if a not in icl[i]]]).astype(int)
        nlabel = len(icl[i])
        ngrids = len(idx)-nlabel

        # skip if has too few points
        if len(idx) < 20:
            print("skip %s%d due to small size"%(trg,i))
            continue

        #assert(np
        # make a copy for the subset grid
        np.savez('split/%s%d.npz'%(trg,i),
                 labels=labels[idx],
                 xyz=grids[idx],
                 name='%s%d'%(trg,i)
        )
        
        # validate
        if validate:
            if i >= 10: continue
            #ATOM      1  N   SER A   3      59.419  26.851  14.790  1.00  0.00
            chain = 'ABCDEFGHIJKLMNOPQ'[i]
            form = "ATOM %6d %-4s %-3s %1s %4d   %8.3f%8.3f%8.3f"
            ctr = 0
            for x in X:
                ctr += 1
                print(form%(ctr,'ZN','ZN',chain,ctr,x[0],x[1],x[2]))
    
            chain = chains[i]
            for x in grids[neighs]:
                ctr += 1
                print(form%(ctr,'O','O',chain,ctr,x[0],x[1],x[2]))
    


trgs_short = np.concatenate([np.load('trim3.train.npy'),np.load('trim3.valid.npy')])
trgs_full  = [l[:6] for l in open('npzs.full')]
trgs_long = [trg for trg in trgs_full if trg not in trgs_short]

a = mp.Pool(processes=10)
a.map(main,trgs_long)
#for trg in trgs_long[:50]:
#    main(trg)



