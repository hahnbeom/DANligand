import sys
import os
import glob
import numpy as np
from math import dist
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose, BreakBRICSBonds
import random
import multiprocessing as mp
import tempfile

OBABEL = 'obabel'

#######################
# function definition #
#######################

def retrieve_frag_name(input):
    m = Chem.MolFromPDBFile(input)

    if m == None:
        return None
    else:
        orgnames = []
        frag_dic = {}
        for i,atm in enumerate(m.GetAtoms()):
            ri = atm.GetPDBResidueInfo()
            orgnames.append(ri.GetName())
            atm.SetAtomMapNum(i)

        natm = len(m.GetAtoms())
        res = list(BRICSDecompose(m))

        m2 = BreakBRICSBonds(m)
        frags = Chem.GetMolFrags(m2,asMols=True)
        a = 0
        for fragno,f in enumerate(frags):
            frag_atm_list = []
            for atm in f.GetAtoms():
                if atm.GetSymbol() == '*':
                    continue
                i = atm.GetAtomMapNum()
                frag_atm_list.append(orgnames[i])
            frag_dic[fragno] = frag_atm_list
        return frag_dic #{[a1,a2,a3],[]...}

def get_atom_lines(mol2_file):
    with open(mol2_file,'r') as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if ln.startswith('@<TRIPOS>ATOM'):
            first_atom_idx = i+1
        if ln.startswith('@<TRIPOS>BOND'):
            last_atom_idx = i-1

    return lines[first_atom_idx:last_atom_idx+1]

def xyz_from_mol2(mol2):
    coordinates = []
    atms = []
    lines = get_atom_lines(mol2file)
    for ln in lines:
        for atm in frag_atm_list:
            if atm in ln:
               x = ln.strip().split()
               Rx = float(x[2])
               Ry = float(x[3])
               Rz = float(x[4])
               R = np.array([Rx,Ry,Rz])
               coordinates.append(R)
               atms.append([atm,R])
    center = np.average(np.array(coordinates),axis=0)

def select_atm_from_frag(xyz,frag_atm_list):
    min_dist_to_center = None
    selected_atm = None
    center = np.zero(3)
    
    for atm in atms:
        atm_R = atm[1]
        distance = dist(xyz,center)
        if min_dist_to_center == None:
            min_dist_to_center = distance
            selected_atm = atm[0].strip()
        else:
            if distance <= min_dist_to_center:
                min_dist_to_center = distance
                selected_atm = atm[0].strip()

    return selected_atm

def select_random_atm(mol2file, pre_selected_atms):
    atm_pool = []
    tobe_selected = 4 - len(pre_selected_atms)
    lines = get_atom_lines(mol2file)
    for ln in lines:
        x = ln.strip().split()
        atm_name = x[1]
        atmtype = x[5]
        if atmtype != 'H':
            if not atm_name in pre_selected_atms:
                atm_pool.append(atm_name)
    selected = pre_selected_atms+(random.sample(atm_pool,tobe_selected))
    return selected

def frag_from_pdb(pdb, mol2xyz, trg):
    BRICSfragments = retrieve_frag_name(pdb)

    if BRICSfragments == None:
        print (i, trg)
        return

    key_atm_list = []
    for fragno in BRICSfragments:
        fragatms = BRICSfragments[fragno]
        selected_atm = select_atm_from_frag(mol2xyz, fragatms)
        key_atm_list.append(selected_atm)

    if len(BRICSfragments) < 4:
        key_atm_list = select_random_atm(mol2, key_atm_list)

    KEYATOMS[trg] = key_atm_list

    return KEYATOMS

def split_pdb(pdb, workpath):
    ligpdbs = []
    
    for l in open(pdb):
        if l.startswith('COMPND'):
            tag = l[:-1].split()[-1]
            ligpdb = '%s.pdb'%tag
            ligpdbs.append([tag,ligpdbs])
            out = open(workpath+'/%s.pdb'%tag,'w')
        if l.startswith('ENDMDL'):
            out.close()
        if l.startswith('ATOM') or l.startswith('CONECT'):
            out.write(l)
    return ligpdbs

##############################
# save key atoms using BRICS #
##############################

def main(mol2s):
    KEYATOMS = {}
    for i, mol2 in enumerate(mol2s):
        trg = mol2.split('/')[-1][:-5].replace('.ligand','')
        ligpdb = mol2[:-5]+'.pdb'
        
        # take pdb instead of mol2 due to stupid rdkit...
        #m = Chem.MolFromMol2File(mol2,sanitize=True)
        #Chem.MolToPDBFile(m, ligpdb)

        # use obabel instead
        os.system(f'{OBABEL} {mol2} -O {ligpdb} 2>/dev/null') 

        is_multi = len(os.popen('grep ^MODEL %s'%ligpdb).readlines())>0
        if is_multi:
            workpath = tempfile.mkdtemp()
            ligpdbs = split_pdb(ligpdb, workpath)

            print("?",ligpdbs)
            for pdb,trg in ligpdbs:
                frag_from_pdb(pdb,trg)
            os.system('rm -rf %s'%workpath)
            
        else:
            frag_from_pdb(ligpdb,trg)
            

def launch(mol2s,N=10,save_separately=True,collated_npz='keyatom.def.npz'):
    a = mp.Pool(processes=N)
    mol2s_split = [[] for i in range(N)]
    print("processing %s mol2s in %d processors"%(len(mol2s), N))
    for i,m in enumerate(mol2s):
        mol2s_split[i%N].append(m)

    #main(mol2s_split[0])
    
    ans = a.map(main, mol2s_split)
    keyatms = {}
    for an in ans:
        for trg in an:
            if len(an[trg]) >= 4:
                keyatms[trg] = an[trg]
            else:
                print(f"skip key atom for {trg} due to insufficient number")
            if save_separately:
                np.savez('%s.keyatom.def.npz'%trg,**{trg:keyatms[trg]}) #keyatm may contain multiple entries for VS

    if not save_separately:
        np.savez(collated_npz,**keyatms) #keyatm may contain multiple entries for VS
            
if __name__ == "__main__":
    mol2s = [l[:-1] for l in open(sys.argv[1])]
    N = 5
    if len(sys.argv) > 3:
        N = int(sys.argv[3])
    #launch(mol2s,N,save_separately=True)
    
    # for saving multiple ligand into a single keynpz
    launch(mol2s,N,save_separately=False,collated_npz='keyatom.def.npz')

    
