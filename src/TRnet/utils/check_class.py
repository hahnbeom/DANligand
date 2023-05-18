import sys
import numpy as np

def sigmoid(vals):
    p = np.exp(-vals)
    p = 1.0/(1.0+p)
    return p

log = open(sys.argv[1]).readlines()
crit = float(sys.argv[2])
valid = True
if '-train' in sys.argv: valid = False

epochls = []
for i,l in enumerate(log):
    if l.startswith('Train/Valid'): epochls.append(i)

b = 0
for i,e in enumerate(epochls):
    nTP,nFP,nP,nC,nW,nT = 0,0,0,0,0,0
    ps_all,ls_all = [],[]
    for l in log[b:e]:
        if valid and not l.startswith('VALID'): continue
        if not valid and not l.startswith('TRAIN'): continue
        words = np.array([float(a) for a in l.split(':')[1].split()])
        nT += 1
        ps = sigmoid(words)
        ps_all.append(ps)
        ls = np.zeros(len(ps))
        ls[0] = 1.0
        ls_all.append(ls)
        if np.argmax(ps) == 0:
            nC += 1
        else:
            nW += 1

        iP = np.where(ps>crit)[0]
        nTP += np.sum(iP==0)
        nFP += np.sum(iP>0)
        nP += len(iP)

    ps_all = np.concatenate(ps_all)
    ls_all = np.concatenate(ls_all)

    isort = np.argsort(-ps_all)
    i5 = int(len(isort)*0.05)
    i1 = int(len(isort)*0.01)

    randomP = np.sum(ls_all)/len(ls_all)
    selP1 = np.sum(ls_all[isort[:i1]])/i1
    selP5 = np.sum(ls_all[isort[:i5]])/i5
    
    print(f"{i:4d} {nTP/nP:5.3f} {nTP/nT:5.3f} {nP:5d} {nC:5d} {nW:5d} {nC+nW:5d}; EF1: {selP1/randomP:5.2f} EF5: {selP5/randomP:5.2f}")
    b = e
