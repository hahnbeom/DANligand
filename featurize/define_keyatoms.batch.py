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
                frag_atm_list.append(orgnames[i].strip())
            frag_dic[fragno] = frag_atm_list
        return frag_dic #{[a1,a2,a3],[]...}

def get_atom_lines(mol2_file):
    with open(mol2_file,'r') as f:
        lines = f.readlines()

    atminfo = {}
    for i, ln in enumerate(lines):
        if ln.startswith('@<TRIPOS>MOLECULE'):
            cmpd = lines[i+1][:-1]
        if ln.startswith('@<TRIPOS>ATOM'):
            first_atom_idx = i+1
        if (ln.startswith('@<TRIPOS>BOND') or ln.startswith('@<TRIPOS>UNITY')) and (cmpd not in atminfo):
            last_atom_idx = i-1
            atminfo[cmpd] = lines[first_atom_idx:last_atom_idx+1]

    return atminfo

def xyz_from_mol2(mol2):
    lines = get_atom_lines(mol2)
    atms = {}
    xyz = {}

    for key in lines:
        xyz[key] = []
        atms[key] = []

        coordinates = []
        for ln in lines[key]:
            x = ln.strip().split()
            atm = x[1]
            Rx = float(x[2])
            Ry = float(x[3])
            Rz = float(x[4])
            R = np.array([Rx,Ry,Rz])
            coordinates.append(R)
            atms[key].append(atm)
        coordinates = np.array(coordinates)
        center = np.average(coordinates,axis=0)
        xyz[key] = coordinates - center
    return xyz, atms

def select_atm_from_frag(xyz,frag_atm_list):
    # xyz are origin-translated
    xyz_f = np.array([xyz[atm] for atm in frag_atm_list])
    com_f = np.mean(xyz_f,axis=0)
    xyz_f -= com_f
    
    d2 = [np.dot(x,x) for x in xyz_f]
    imin = np.argsort(d2)[0]
    selected_atm = frag_atm_list[imin]

    return selected_atm

def frag_from_pdb(pdb, mol2xyz, atms, trg):
    BRICSfragments = retrieve_frag_name(pdb)

    if BRICSfragments == None:
        return

    key_atm_list = []
    for fragno in BRICSfragments:
        fragatms = BRICSfragments[fragno]
        atmxyz = {atm:x for atm,x in zip(atms, mol2xyz)}
        selected_atm = select_atm_from_frag( atmxyz, fragatms)
        key_atm_list.append(selected_atm)

    if len(key_atm_list) < 4:
        npick = 4-len(key_atm_list)
        toadd = list(np.random.choice([a for a in atms if (a not in key_atm_list and a[0] != 'H')],npick))
        key_atm_list += toadd

    return key_atm_list

def split_pdb(pdb, workpath):
    ligpdbs = []
    
    for l in open(pdb):
        if l.startswith('COMPND'):
            tag = l[:-1].split()[-1].replace('.pdb','')
            ligpdb = workpath+'/%s.pdb'%tag
            ligpdbs.append([tag,ligpdb])
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
            mol2xyz,atms = xyz_from_mol2(mol2)
            print(mol2xyz.keys())

            for trg,pdb in ligpdbs:
                try:
                    KEYATOMS[trg] = frag_from_pdb(pdb, mol2xyz[trg], atms[trg], trg)
                except:
                    print("skip %s"%trg)
                    continue
                
            os.system('rm -rf %s'%workpath)
            
        else:
            mol2xyz,atms = xyz_from_mol2(mol2)
            KEYATOMS[trg] = frag_from_pdb(ligpdb, mol2xyz, atms, trg)

    return KEYATOMS

def local_runner(args):
    pdb,xyz,atms,trg = args
    try:
        keyatoms = {trg:frag_from_pdb(pdb, xyz, atms, trg)}
    except:
        return None
    return keyatoms
    
def launch_batch_mol2(mol2, N=10, collated_npz='keyatom.def.npz'):
    trg = mol2.split('/')[-1][:-5].replace('.ligand','')
    ligpdb = mol2[:-5]+'.pdb'
    os.system(f'{OBABEL} {mol2} -O {ligpdb} 2>/dev/null')
        
    workpath = tempfile.mkdtemp()
    ligpdbs = split_pdb(ligpdb, workpath)
    mol2xyz,atms = xyz_from_mol2(mol2)

    args = []
    for trg,pdb in ligpdbs:
        args.append((pdb,mol2xyz[trg],atms[trg],trg))
        
    a = mp.Pool(processes=N)
    ans = a.map(local_runner,args)
    keyatms = {}
    for an in ans:
        if an == None: continue
        for tag in an:
            keyatms[tag] = an[tag]
        
    os.system('rm -rf %s'%workpath)
    np.savez(collated_npz,keyatms=keyatms) #keyatm may contain multiple entries for VS
 
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
            if an[trg] == None: continue
            
            if len(an[trg]) >= 4:
                keyatms[trg] = an[trg]
            else:
                print(f"skip key atom for {trg} due to insufficient number")
            if save_separately:
                np.savez('%s.keyatom.def.npz'%trg,**{trg:keyatms[trg]}) #keyatm may contain multiple entries for VS

    if not save_separately:
        np.savez(collated_npz,**keyatms) #keyatm may contain multiple entries for VS
            
if __name__ == "__main__":
    if sys.argv[1].endswith('.mol2'):
        mol2s = [sys.argv[1]]
    else:
        mol2s = [l[:-1] for l in open(sys.argv[1])]
    N = 5
    if len(sys.argv) > 2:
        N = int(sys.argv[2])

    # option1. save output separately
    #launch(mol2s,N,save_separately=True)
    
    # option2. for saving multiple ligand into a single keynpz
    #launch(mol2s,N,save_separately=False,collated_npz='%s.keyatom.def.npz'%(mol2s[0].split('.')[0]))

    # option3. if mol2 is a batch mol2
    launch_batch_mol2(mol2s[0],N,collated_npz='%s.keyatom.def.npz'%(mol2s[0].split('.')[0]))

