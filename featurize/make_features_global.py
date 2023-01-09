import sys
sys.path.insert(0,'/applic/MotifNet/featurize')
from featurize_usage import main

pdb = sys.argv[1]
trg = pdb[:-4]
main(pdb,trg,gridsize=2.5,gridoption='global')
    
