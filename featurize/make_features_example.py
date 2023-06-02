import sys
import numpy as np
sys.path.insert(0,'/home/hpark/projects/DANligand/featurize')

from featurize_usage import main

mode = 'com'

if mode == 'global': 
    pdb = sys.argv[1]
    trg = pdb[:-4]
    main(pdb,trg,gridsize=2.5,gridoption='global')
    
elif mode == 'local':
    pdb = sys.argv[1]
    recpdb = sys.argv[2]
    trg = pdb[:-4]
    main(pdb,trg,recpdb=recpdb,gridsize=1.0,ligchain='B')

elif mode == 'com':
    pdb = sys.argv[1]
    x = float(sys.argv[2])
    y = float(sys.argv[3])
    z = float(sys.argv[4])

    xyz = np.array([x,y,z])
    trg = pdb[:-4]
    main(pdb,outprefix=trg,gridsize=1.5,com=xyz,padding=12.0,gridoption='com')

    
