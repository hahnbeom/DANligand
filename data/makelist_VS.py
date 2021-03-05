import os,sys
import numpy as np
import random

K = 5

ftrn = 0.8
fval = 0.2
tags = [l[:-1] for l in open(sys.argv[1])] 

random.shuffle(tags)

n = len(tags)
tags_trn = tags[:int(n*ftrn)]
tags_val = tags[int(n*ftrn):]

print('Tr/V: %d/%d'%(len(tags_trn),len(tags_val)))
                
np.save("trainVS_proteins%d.npy"%K,tags_trn)
np.save("validVS_proteins%d.npy"%K,tags_val)

