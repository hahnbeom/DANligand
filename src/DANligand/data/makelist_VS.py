import os,sys
import numpy as np
import random

K = 5

ftrn = 0.8
fval = 0.2
tags = [l[:-1] for l in open('trgs.VS')] 

# train/valid contains 5/3 COF 
k_bar = tags.index('ppard.0')
tags_trn = tags[:k_bar]
tags_val = tags[k_bar:]

print('Tr/V: %d/%d'%(len(tags_trn),len(tags_val)))
                
np.save("trainVS_proteins%d.npy"%K,tags_trn)
np.save("validVS_proteins%d.npy"%K,tags_val)

