import sys
import numpy as np

a = np.load(sys.argv[1])

n = len(a['affinity'])
for i in range(n):
    print("%8.3f %8.3f"%(a['affinity'][i],a['fnat'][i]))

