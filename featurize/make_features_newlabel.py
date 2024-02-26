import sys,os
sys.path.insert(0,'/home/hpark/projects/DANligand/featurize')
import featurize_ligand
import multiprocessing as mp

def runner(trg):
    if os.path.exists(trg+'.grid.npz'): return

    pdb  = '%s.m.pdb'%trg
    if not os.path.exists(pdb):
        pdb = '../holo/%s.holo.pdb'%trg
    mol2 = '../ligands/full/%s.mol2'%trg
    try:
        featurize_ligand.main(pdb,mol2,inputpath='./',outprefix=trg,gridsize=1.5,
                              padding=5.0,verbose=False,debug=False)
    except:
        print("pass %s"%trg)
    
trgs = [l.split('/')[1][:8] for l in open('../filt3')]
#trgs = ['10mh.SAH']
#for trg in trgs:
#    print(trg)
#    runner(trg)

a = mp.Pool(processes=30)
a.map(runner,trgs)
