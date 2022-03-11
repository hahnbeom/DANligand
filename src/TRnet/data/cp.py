import sys,os

trgs = [l[:-1] for l in open(sys.argv[1])]
for trg in trgs:
    os.system('cp ~/projects/BindingSite/result.1ang/%s/complex.score.npz ./%s.score.npz'%(trg,trg))
    os.system('cp ~/projects/BindingSite/result.1ang/%s/crystal_ligand.mol2 ./%s.ligand.mol2'%(trg,trg))
