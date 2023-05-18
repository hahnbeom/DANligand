import sys
import numpy as np

trgs_train = list(np.load('trainlist.npy'))+[l[:-1] for l in open('trgs.biolip')]

#ntrain = int(len(trgs)*0.7)
np.save('trainlist.combo.npy',np.array(trgs_train))
print(len(trgs_train))
#np.save('validlist.combo.npy',np.array(trgs[ntrain:])) #share the same validation set
