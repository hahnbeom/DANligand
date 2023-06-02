import sys
sys.path.insert(0,'/applic/MotifNet/featurize')
from featurize_usage import main

pdb = sys.argv[1]
recpdb = sys.argv[2]
ligchain = sys.argv[3]
trg = pdb[:-4]
main(pdb,trg,recpdb=recpdb,gridsize=1.0,ligchain=ligchain)

