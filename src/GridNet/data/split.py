import numpy as np

trgs = [l[:-1] for l in open('pick.list')]

k80 = int(len(trgs)*0.8)

train = trgs[:k80]
valid = trgs[k80:]

np.save('train.npy',train)
np.save('valid.npy',valid)
