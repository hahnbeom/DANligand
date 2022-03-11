import sys
import numpy as np

KEYATOMS = {'ada17':['C10','C9','C13','N1'],
            'vgfr2':['N1','C23','C19','C1'],
            'egfr': ['C3','C18','C22','C16'],
            'hs90a':['N1','C6','C1','C2'],
            'try1': ['N1','C16','C28','C19'],
            'aces': ['C1','C2','N1','C5'],
            'bace1' : ['C1','C19','N1','O2'],
            'bace1r': ['C1','C18','N4','O1'], # bace1-conf-from-scratch
            ## added
            'abl1': ['C19','C1','C3','O1'],
            'akt1': ['C12','N1','N2', 'C4'],
            'akt2': ['C16','O1','N6','N1'],
            #'aofb': ['C13','O1','N6','N1'],
            'braf': ['C16','O1','N3','N1'],
            'cah2': ['S1','F3','O2','N1'],
            'casp3': ['C10','C1','O2','N1'],
            'cp3a4': ['C35','C29','N1','C6','C7'],
            'csf1r': ['C16','N1','N4','C5'],
            #'def': ['N1','O2','O1',''],
            'dhi1': ['C18','F3','N1','C6'],
            'dpp4': ['C8','F3','C19','N1'],
            'esr2': ['C22','C4','O1','N1'],
            'fa7': ['C24','N2','O1','C2'],
            'fabp4': ['C30','C8','C2','C31','C3'],
            'fak1': ['C11','O1','S1','C2'],
            'fkb1a': ['N1','C1','C2','S1'],
            'fnta': ['C21','C2','C5','C6','N1'],
            'fpps': ['C5','C1','O6','O5'],
            'glcm': ['N1','O4','O1','O2'],
            'gria2': ['C2','O3','O6','P1'],
            'grik1': ['C6','O2','O1','C11'],
            'hdac2': ['C18','C1','C2','N1'],
            'hdac8': ['C7','C2','O1','O3'],
            'hivint': ['C10','Cl1','C7','C17'],
            'hmdh': ['N3','C4','O3','F1','C23' ],
            'hxk4': ['C6','C5','C4','N1'],
            'igf1r': ['C20','C8','C6','C3'],
            'ital': ['C26','O1','N1','C8'],
            'jak2': ['N2','S1','C24','C5'],
            'kif11': ['N2','C1','C9','C3'],
            'kit': ['C6','N1','F1','O1'],
            'kpcb': ['C17','N4','O1','C6'],
            'lck': ['C19','N2','C5','C1','C12'],
            'lkha4': ['C7','C1','O1','N1'],
            'mapk2': ['C20','C3','C1','O1'],
            'mcr': ['C15','O1','O3','C3'],
            'met': ['C7','O3','F1','F2'],
            'mk01': ['C8','C5','C1','N2'],
            'mk10': ['C23','O1','Cl1','C2'],
            'mmp13': ['C16','Cl1','O5','O2'],
            'mp2k1': ['C15','C2','C5','N5'],
            #
            'nos1': ['C3','C2','O1'],
            'pa2ga': ['C13','O3','C2'],
            'plk1': ['O2','C1','O1'],
            'ppara': ['C18','O3','F3'],
            'ppard': ['N1','O1','F2'],
            'ptn1': ['C6','O3','C3'],
            'pyrd': ['C17','F1','C2'],
            'reni': ['C30','Cl1','C16'],
            'rock1': ['C12','N2','N1'],
            'tgfr1': ['N4','C1','C11'],
            'thb': ['C19','O3','C5'],
            'tryb1': ['C10','C3','N1'],
            'tysy': ['C9','C1','O5'],
            'urok': ['C14','O1','N1'],
            'wee1': ['C22','C2','C5'],
            'xiap': ['C28','C10','C3'],
            'ace': ['O1','C14','C1'],
            'ada': ['C12','N1','C7'],
            'aldr': ['S1','F2','C1'],
            'ampc': ['C7','O1','O6'],
            'andr': ['C15','O1','O2'],
            'cdk2': ['C6','C2','F1'],
            'dyr': ['C12','C5','N2'],
            'esr1': ['C4','C14','O1'],
            'fa10': ['O1','C10','Cl1'],
            'fgfr1': ['C14','C1','C7'],
            'gcr': ['C22','C5','O1'],
            'hivrt': ['C20','N1','O1'],
            'inha': ['O2','C10','C3'],
            'kith': ['N2','O1','O3'],
            'mk14': ['C17','F1','C8'],
            'nram': ['C14','O5','C6'],
            'pgh1': ['C14','O1','Cl1'],
            'pgh2': ['C13','C1','N1'],
            'pnph': ['C7','O2','O3'],
            'pparg': ['C19','C4','O1'],
            'prgr': ['N3','N1','O2'],
            'pur2': ['C2','N2','O6'],
            'rxra': ['C6','C5','O1'],
            'sahh': ['N5','O1','N1'],
            'src': ['C8','F3','C1'],
            'thrb': ['C25','C12','N1']
}

def read_mol2(mol2):
    read_cont = 0
    xyzs = []
    atms = []
    
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            break

        words = l[:-1].split()
        if read_cont == 1:
            idx = words[0]
            if words[1][0] == 'H': continue
            
            atms.append(words[1])
            xyzs.append([float(words[2]),float(words[3]),float(words[4])]) 
                
    return np.array(xyzs), np.array(atms)

def pick(xyz,n):
    com = np.mean(xyz,axis=0)
    N = len(xyz)

    refps = []
    keys = []
    
    for k in range(n):
        D2cen = np.zeros(N)
        if k == 0:
            D2cen = np.sqrt(np.sum((xyz-com)*(xyz-com),axis=1))
            i = np.argmin(D2cen)
        else:
            for p in refps:
                D2cen += np.sqrt(np.sum((xyz-p)*(xyz-p),axis=1))
            D2cen[keys] -= 1e6
            i = np.argmax(D2cen)
            
        refps.append(xyz[i])
        
        keys.append(i)
    return keys

#trgs = [l[:-1] for l in open(sys.argv[1]) if not l.startswith('#')]

#for trg in trgs:
#    xyz,atoms = read_mol2('%s.ligand.mol2'%trg)
#    keyidx = pick(xyz,3)
#    print("'%s':"%trg,atoms[keyidx],",")
keyatms = {}
for trg in KEYATOMS:
    if len(KEYATOMS[trg]) >= 4:
        keyatms[trg] = KEYATOMS[trg]

np.savez('keyatom.def.npz',keyatms=keyatms)
