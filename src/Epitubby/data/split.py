import numpy as np

trgs = []
for l in open('interfaces.legit'):
    trgs.append(l[:4]+'_'+l[4]+'_'+l[5])

n8 = int(0.8*len(trgs))
n9 = int(0.9*len(trgs))

np.save('trainlist.v1.npy',trgs[:n8])
np.save('validlist.v1.npy',trgs[n8:n9])
np.save('testlist.v1.npy',trgs[n9:])
