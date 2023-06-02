import sys
import os
import glob
import numpy as np
from math import dist
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Draw
from brics_hpark import retrieve_frag_name
import random 


others = glob.glob('/home/j2ho/DB/pdbbind/v2020-others/*/')
refined = glob.glob('/home/j2ho/DB/pdbbind/v2020-refined/*/')
#pdbs = [sys.argv[1]]

PDB_run = []

#with open('/home/j2ho/vs/motif/pdbv2020_refined_5316.list', 'r') as f: 
#    lines = f.readlines() 
#    for ln in lines:
#        ln = ln.strip() 
#        PDB_run.append(ln)

#for pdbdir in pdbs: 
#    pdbid = pdbdir.split('/')[-2]
#    if pdbid == 'index':
#        continue
#    if pdbid == 'readme':
#        continue
#    PDB_run.append(pdbid)
#    print (PDB_run)
#    sys.exit()



#######################
# function definition #
#######################


def get_atom_lines(mol2_file): 
    with open(mol2_file,'r') as f: 
        lines = f.readlines() 
    for i, ln in enumerate(lines): 
        if ln.startswith('@<TRIPOS>ATOM'): 
            first_atom_idx = i+1
        if ln.startswith('@<TRIPOS>BOND'): 
            last_atom_idx = i-1

    return lines[first_atom_idx:last_atom_idx+1]


def select_atm_from_frag(mol2file,frag_atm_list):
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
    min_dist_to_center = None
    selected_atm = None
    for atm in atms: 
        atm_R = atm[1]
        distance = dist(atm_R,center)
#        print (atm, distance)
        if min_dist_to_center == None:
            min_dist_to_center = distance
            selected_atm = atm[0].strip()
        else: 
            if distance <= min_dist_to_center: 
                min_dist_to_center = distance
                selected_atm = atm[0].strip() 
#    print (min_dist_to_center)
#    print (selected_atm)

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


##############################
# save key atoms using BRICS # 
##############################


KEYATOMS = {} 

for i, elem in enumerate(others):
    pdbid = elem.split('/')[-2]
    if pdbid == 'index':
        continue
    if pdbid == 'readme':
        continue
    key_atm_list = []
    ligpdb = '/home/j2ho/DB/pdbbind/v2020-others/%s/%s_ligand_renamed.pdb'%(pdbid, pdbid)
#    if not pdbid in PDB_run:
#        continue
    BRICSfragments = retrieve_frag_name(ligpdb)
    if BRICSfragments == None: 
        print (i, pdbid) 
    else: 
        if len(BRICSfragments) >= 4:
            for elem in BRICSfragments: 
                fraglist = BRICSfragments[elem]
                selected_atm = select_atm_from_frag('/home/j2ho/DB/pdbbind/v2020-others/%s/%s_ligand_renamed.mol2'%(pdbid, pdbid), fraglist)
                key_atm_list.append(selected_atm)
            KEYATOMS[pdbid] = key_atm_list
        if len(BRICSfragments) < 4: 
            for elem in BRICSfragments: 
                fraglist = BRICSfragments[elem]
                selected_atm = select_atm_from_frag('/home/j2ho/DB/pdbbind/v2020-others/%s/%s_ligand_renamed.mol2'%(pdbid, pdbid), fraglist)
                key_atm_list.append(selected_atm)
            new_key_atm_list = select_random_atm('/home/j2ho/DB/pdbbind/v2020-others/%s/%s_ligand_renamed.mol2'%(pdbid,pdbid), key_atm_list) 
            KEYATOMS[pdbid] = new_key_atm_list


for i, elem in enumerate(refined):
    pdbid = elem.split('/')[-2]
    if pdbid == 'index':
        continue
    if pdbid == 'readme':
        continue
    key_atm_list = []
    ligpdb = '/home/j2ho/DB/pdbbind/v2020-refined/%s/%s_ligand_renamed.pdb'%(pdbid, pdbid)
#    if not pdbid in PDB_run:
#        continue
    BRICSfragments = retrieve_frag_name(ligpdb)
    if BRICSfragments == None: 
        print (i, pdbid) 
    else: 
        if len(BRICSfragments) >= 4:
            for elem in BRICSfragments: 
                fraglist = BRICSfragments[elem]
                selected_atm = select_atm_from_frag('/home/j2ho/DB/pdbbind/v2020-refined/%s/%s_ligand_renamed.mol2'%(pdbid, pdbid), fraglist)
                key_atm_list.append(selected_atm)
            KEYATOMS[pdbid] = key_atm_list
        if len(BRICSfragments) < 4: 
            for elem in BRICSfragments: 
                fraglist = BRICSfragments[elem]
                selected_atm = select_atm_from_frag('/home/j2ho/DB/pdbbind/v2020-refined/%s/%s_ligand_renamed.mol2'%(pdbid, pdbid), fraglist)
                key_atm_list.append(selected_atm)
            new_key_atm_list = select_random_atm('/home/j2ho/DB/pdbbind/v2020-refined/%s/%s_ligand_renamed.mol2'%(pdbid,pdbid), key_atm_list) 
            KEYATOMS[pdbid] = new_key_atm_list


keyatms = {}
a = 0
for trg in KEYATOMS:
    if len(KEYATOMS[trg]) >= 4:
        keyatms[trg] = KEYATOMS[trg]
    else: 
        a += 1
print (a) 
np.savez('keyatom.def.npz',keyatms=keyatms)

sys.exit() 




########################################
#    OLD VERSION USING FINDBRICSBONDS  #
########################################



for elem in ligpdbs:
    pdbid = elem[:-4].split('/')[-1].split('_')[0]
    if not pdbid in PDB_1000: 
        continue
    print (elem)
    BRICSfragments = retrieve_frag_name(elem)
    m = Chem.MolFromPDBFile(elem)
    bonds = []
    for bond in m.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx() 
        a2 = bond.GetEndAtom().GetIdx() 
        bonds.append([a1,a2])
    print (bonds)
    BRICSbonds = list(BRICS.FindBRICSBonds(m))
    BRICS_atms = [] #atoms from BRICS bonds 
    for Bbond in BRICSbonds: 
        if not Bbond[0][0] in BRICS_atms: 
            BRICS_atms.append(Bbond[0][0])
        if not Bbond[0][1] in BRICS_atms: 
            BRICS_atms.append(Bbond[0][1])
    print (BRICSbonds) 
    print (BRICS_atms)


if __name__==' __main__': 

    m = Chem.MolFromSmiles('CC[NH+](CC)CCOc1ccc(cc1)Cc2ccccc2')
    bonds = m.GetBonds() 
    for bond in bonds: 
        a1 = bond.GetBeginAtom().GetIdx() 
        a2 = bond.GetEndAtom().GetIdx() 
        print (a1, a2)
    BRICSbonds = list(BRICS.FindBRICSBonds(m))
    print (BRICSbonds)
    sys.exit()
    import numpy as np

    keyatms = np.load('keyatom.def.npz',allow_pickle=True)['keyatms'].item()

    print (keyatms)
