import sys
import numpy as np

trgs = [l[:-1] for l in open('trgs.all') if not l.startswith('#')]

ntrain = int(len(trgs)*0.7)

np.save('trainlist.npy',np.array(trgs[:ntrain]))
np.save('validlist.npy',np.array(trgs[ntrain:]))
