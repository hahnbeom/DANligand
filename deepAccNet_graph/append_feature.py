import sys
import numpy as np

def append_npz(npz,outnpz,featv,featname):
    a = np.load(npz,allow_pickle=True)
    cont = {}
    for key in a.keys():
        cont[key] = a[key]
    cont[featname] = featv
    
    np.savez(outnpz,**cont)

def convert_pK(fnats,pK):
    # use a logistic function
    #x = np.exp(10.0*(fnats-0.5))
    x = np.exp(5.0*(fnats-0.7))
    logistic = 1.5*x/(1+x)
    over1 = np.where(logistic>1)
    logistic[over1] = 1.0
    return pK*logistic
    
if __name__ == "__main__":
    npz = sys.argv[1]
    trg = npz[:4]
    outnpz = npz.replace('.npz','.1.npz')

    data = np.load(npz)
    fnats = data['fnat']
    pK = np.load('../data/bindingaffinity.npz',allow_pickle=True)
    print(pK[trg])
    featv = convert_pK(fnats,pK[trg])

    append_npz(npz,outnpz,featv,"affinity")
             
