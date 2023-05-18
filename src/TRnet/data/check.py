import os
import numpy as np

n,x = 0,0
for l in open('trgs.biolip'):
    key = l[:-1]
    f = '/ml/motifnet/TRnet.combo/biolip/%s.score.npz'%key
    if os.path.exists(f):
        n += 1
    else:
        x += 1

print(n,x)
