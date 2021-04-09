import os,sys
import numpy as np
import random

K = 6

symm = [l[:-1] for l in open('trgs.symm')]

ftrn = 0.9
fval = 0.1
ftst = 0.0001
tags = []
tags_symm = []
for l in open(sys.argv[1]):
    tag = l[:-1].split('/')[-1].split('.')[0]
    if tag in symm:
        tags_symm.append(tag)
    else:
        tags.append(tag)

random.shuffle(tags)
random.shuffle(tags_symm)
tags += tags_symm

n = len(tags)
tags_trn = tags[:int(n*ftrn)]
tags_val = tags[int(n*ftrn):-int(n*ftst)]
tags_tst = tags[-int(n*ftst):]

print('Tr/V/Ts: %d/%d/%d'%(len(tags_trn),len(tags_val),len(tags_tst)))
                
np.save("train_proteins%d.npy"%K,tags_trn)
np.save("valid_proteins%d.npy"%K,tags_val)
np.save("test_proteins%d.npy"%K, tags_tst)
