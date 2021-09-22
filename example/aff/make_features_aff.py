import os,sys,glob
sys.path.insert(0,'/home/hpark/programs/DANse3/ligand/deepAccNet_XG')
import featurize
import multiprocessing as mp

def run(tag):
    #params = ['%s/X%02d.params'%(tag,k) for k in range(30) if os.path.exists('%s/X%02d.params'%(tag,k))]
    curr = os.getcwd()
    workpath = curr+'/'+tag

    '''
    with open('%s/featurize.log'%workpath,'w') as outf:
        pdbs = glob.glob('%s/*pdb'%workpath)
        for pdb in pdbs:
            pdbtag = pdb.split('/')[-1][:-4]
            featurize.main(tag,verbose=True,decoytypes=[pdbtag],
                           ligres=tag,
                           out=outf,
                           inputpath=curr,
                           outpath=workpath,
                           outprefix=pdbtag,
                           refligand=-1,
                           store_npz=True,
                           debug=True,
                           same_answer=True)
    '''
    try:
        with open('%s/featurize.log'%workpath,'w') as outf:
            pdbs = glob.glob('%s/*pdb'%workpath)
            for pdb in pdbs:
                pdbtag = pdb.split('/')[-1][:-4]
                featurize.main(tag,verbose=True,decoytypes=[pdbtag],
                               ligres=tag,
                               out=outf,
                               inputpath=curr,
                               outpath=workpath,
                               outprefix=pdbtag,
                               refligand=-1,
                               store_npz=True,
                               debug=True,
                               same_answer=True)
    except:
        print("failed", tag)
        return
        
    
if __name__ == "__main__":
    #run(sys.argv[1])
    trgs = [l[:-1] for l in open(sys.argv[1])]
    a = mp.Pool(processes=20)
    a.map(run,trgs)
    #for trg in trgs[:1]: run(trg)
