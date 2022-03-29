import numpy as np
import os,sys

npz = sys.argv[1]

#Pcut = 0.3 #consider only those P higher than this
Pb = 0.3

ks = [1,2,3,4,6,7,8,9,11,12,13]
Ps_by_k = [[] for k in ks]
motifs = ['ASP','ARG','LYS','ALC','AMD','NHS','NTR','PHE','PHO','PH2', 'MET']
Pcut =   [   Pb,   Pb,   Pb,  0.9,  Pb,  0.3,   0.9,  0.4,  0.4,  0.6,   0.4] #"reference correction"

a = np.load(npz, allow_pickle=True)
for Ps,xyz in zip(a['p'],a['grids']):
    for i,k in enumerate(ks):
        if Ps[k] > Pcut[i]: Ps_by_k[i].append((Ps[k],xyz))

form = "HETATM %4d  %3s UNK A %3d    %8.3f%8.3f%8.3f  1.00 %5.2f\n"
for i,k in enumerate(ks):
    outf = '.'.join(npz.split('.')[:-1])+'.'+motifs[i]+'.pdb'
    out = open(outf,'w')
    for iatm,(P,xyz) in enumerate(Ps_by_k[i]):
        out.write(form%(i+1,motifs[i],i+1,xyz[0],xyz[1],xyz[2],P))
    out.close()
    
            
