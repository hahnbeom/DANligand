import torch
import dgl
import numpy as np
from scipy.spatial import KDTree
from torch import Tensor

'''
def collate(samples):
    valid = [v[0] != None for v in samples]
    #if samples[0][0] != None:
    try:
        names  = [s[1] for s in samples]
        tags = [s[2] for s in samples]
        samples = [s[0] for s in samples]
        bG = dgl.batch(samples)

        node_feature = {"0": bG.ndata["attr"][:,:,None].float()}
        edge_feature = {"0": bG.edata["attr"][:,:,None].float()}
    
        return bG, node_feature, edge_feature, names, tags
    except:
        #print("Error happened:",)# samples[0][1])
        return None, {}, {}, [""], [""]
'''
    
def getsubgraph(G, max_atom_count=3000):
    L = G.ndata["pos"].shape[0]
    if L > max_atom_count:
        pivot = np.random.choice(np.arange(L))
        tree = KDTree(G.ndata["pos"])
        inds = tree.query(G.ndata["pos"][pivot], max_atom_count)[-1]
        inds.sort()
        G2 = dgl.node_subgraph(G, inds)
        return G2
    else:
        return G

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_cuda(x, device):
    if isinstance(x, Tensor):
        return x.to(device)
    elif isinstance(x, tuple):
        return (to_cuda(v, device) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=device)
    
# Tip atom definitions
AA_to_tip = {"ALA":"CB", "CYS":"SG", "ASP":"CG", "ASN":"CG", "GLU":"CD",
             "GLN":"CD", "PHE":"CZ", "HIS":"NE2", "ILE":"CD1", "GLY":"CA",
             "LEU":"CG", "MET":"SD", "ARG":"CZ", "LYS":"NZ", "PRO":"CG",
             "VAL":"CB", "TYR":"OH", "TRP":"CH2", "SER":"OG", "THR":"OG1"}

# Residue number definition
residues= ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',\
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',\
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(residues[i], i) for i in range(len(residues))])

# minimal sc atom representation (Nx8)
aa2short={
    "ALA": (" N  "," CA "," C  "," CB ",  None,  None,  None,  None), 
    "ARG": (" N  "," CA "," C  "," CB "," CG "," CD "," NE "," CZ "), 
    "ASN": (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), 
    "ASP": (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), 
    "CYS": (" N  "," CA "," C  "," CB "," SG ",  None,  None,  None), 
    "GLN": (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), 
    "GLU": (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), 
    "GLY": (" N  "," CA "," C  ",  None,  None,  None,  None,  None), 
    "HIS": (" N  "," CA "," C  "," CB "," CG "," ND1",  None,  None),
    "ILE": (" N  "," CA "," C  "," CB "," CG1"," CD1",  None,  None), 
    "LEU": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), 
    "LYS": (" N  "," CA "," C  "," CB "," CG "," CD "," CE "," NZ "), 
    "MET": (" N  "," CA "," C  "," CB "," CG "," SD "," CE ",  None), 
    "PHE": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "PRO": (" N  "," CA "," C  "," CB "," CG "," CD ",  None,  None), 
    "SER": (" N  "," CA "," C  "," CB "," OG ",  None,  None,  None),
    "THR": (" N  "," CA "," C  "," CB "," OG1",  None,  None,  None),
    "TRP": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "TYR": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "VAL": (" N  "," CA "," C  "," CB "," CG1",  None,  None,  None),
}

# Atom types:
atypes = {('ALA', 'CA'): 'CAbb', ('ALA', 'CB'): 'CH3', ('ALA', 'C'): 'CObb', ('ALA', 'N'): 'Nbb', ('ALA', 'O'): 'OCbb', ('ARG', 'CA'): 'CAbb', ('ARG', 'CB'): 'CH2', ('ARG', 'C'): 'CObb', ('ARG', 'CD'): 'CH2', ('ARG', 'CG'): 'CH2', ('ARG', 'CZ'): 'aroC', ('ARG', 'NE'): 'Narg', ('ARG', 'NH1'): 'Narg', ('ARG', 'NH2'): 'Narg', ('ARG', 'N'): 'Nbb', ('ARG', 'O'): 'OCbb', ('ASN', 'CA'): 'CAbb', ('ASN', 'CB'): 'CH2', ('ASN', 'C'): 'CObb', ('ASN', 'CG'): 'CNH2', ('ASN', 'ND2'): 'NH2O', ('ASN', 'N'): 'Nbb', ('ASN', 'OD1'): 'ONH2', ('ASN', 'O'): 'OCbb', ('ASP', 'CA'): 'CAbb', ('ASP', 'CB'): 'CH2', ('ASP', 'C'): 'CObb', ('ASP', 'CG'): 'COO', ('ASP', 'N'): 'Nbb', ('ASP', 'OD1'): 'OOC', ('ASP', 'OD2'): 'OOC', ('ASP', 'O'): 'OCbb', ('CYS', 'CA'): 'CAbb', ('CYS', 'CB'): 'CH2', ('CYS', 'C'): 'CObb', ('CYS', 'N'): 'Nbb', ('CYS', 'O'): 'OCbb', ('CYS', 'SG'): 'S', ('GLN', 'CA'): 'CAbb', ('GLN', 'CB'): 'CH2', ('GLN', 'C'): 'CObb', ('GLN', 'CD'): 'CNH2', ('GLN', 'CG'): 'CH2', ('GLN', 'NE2'): 'NH2O', ('GLN', 'N'): 'Nbb', ('GLN', 'OE1'): 'ONH2', ('GLN', 'O'): 'OCbb', ('GLU', 'CA'): 'CAbb', ('GLU', 'CB'): 'CH2', ('GLU', 'C'): 'CObb', ('GLU', 'CD'): 'COO', ('GLU', 'CG'): 'CH2', ('GLU', 'N'): 'Nbb', ('GLU', 'OE1'): 'OOC', ('GLU', 'OE2'): 'OOC', ('GLU', 'O'): 'OCbb', ('GLY', 'CA'): 'CAbb', ('GLY', 'C'): 'CObb', ('GLY', 'N'): 'Nbb', ('GLY', 'O'): 'OCbb', ('HIS', 'CA'): 'CAbb', ('HIS', 'CB'): 'CH2', ('HIS', 'C'): 'CObb', ('HIS', 'CD2'): 'aroC', ('HIS', 'CE1'): 'aroC', ('HIS', 'CG'): 'aroC', ('HIS', 'ND1'): 'Nhis', ('HIS', 'NE2'): 'Ntrp', ('HIS', 'N'): 'Nbb', ('HIS', 'O'): 'OCbb', ('ILE', 'CA'): 'CAbb', ('ILE', 'CB'): 'CH1', ('ILE', 'C'): 'CObb', ('ILE', 'CD1'): 'CH3', ('ILE', 'CG1'): 'CH2', ('ILE', 'CG2'): 'CH3', ('ILE', 'N'): 'Nbb', ('ILE', 'O'): 'OCbb', ('LEU', 'CA'): 'CAbb', ('LEU', 'CB'): 'CH2', ('LEU', 'C'): 'CObb', ('LEU', 'CD1'): 'CH3', ('LEU', 'CD2'): 'CH3', ('LEU', 'CG'): 'CH1', ('LEU', 'N'): 'Nbb', ('LEU', 'O'): 'OCbb', ('LYS', 'CA'): 'CAbb', ('LYS', 'CB'): 'CH2', ('LYS', 'C'): 'CObb', ('LYS', 'CD'): 'CH2', ('LYS', 'CE'): 'CH2', ('LYS', 'CG'): 'CH2', ('LYS', 'N'): 'Nbb', ('LYS', 'NZ'): 'Nlys', ('LYS', 'O'): 'OCbb', ('MET', 'CA'): 'CAbb', ('MET', 'CB'): 'CH2', ('MET', 'C'): 'CObb', ('MET', 'CE'): 'CH3', ('MET', 'CG'): 'CH2', ('MET', 'N'): 'Nbb', ('MET', 'O'): 'OCbb', ('MET', 'SD'): 'S', ('PHE', 'CA'): 'CAbb', ('PHE', 'CB'): 'CH2', ('PHE', 'C'): 'CObb', ('PHE', 'CD1'): 'aroC', ('PHE', 'CD2'): 'aroC', ('PHE', 'CE1'): 'aroC', ('PHE', 'CE2'): 'aroC', ('PHE', 'CG'): 'aroC', ('PHE', 'CZ'): 'aroC', ('PHE', 'N'): 'Nbb', ('PHE', 'O'): 'OCbb', ('PRO', 'CA'): 'CAbb', ('PRO', 'CB'): 'CH2', ('PRO', 'C'): 'CObb', ('PRO', 'CD'): 'CH2', ('PRO', 'CG'): 'CH2', ('PRO', 'N'): 'Npro', ('PRO', 'O'): 'OCbb', ('SER', 'CA'): 'CAbb', ('SER', 'CB'): 'CH2', ('SER', 'C'): 'CObb', ('SER', 'N'): 'Nbb', ('SER', 'OG'): 'OH', ('SER', 'O'): 'OCbb', ('THR', 'CA'): 'CAbb', ('THR', 'CB'): 'CH1', ('THR', 'C'): 'CObb', ('THR', 'CG2'): 'CH3', ('THR', 'N'): 'Nbb', ('THR', 'OG1'): 'OH', ('THR', 'O'): 'OCbb', ('TRP', 'CA'): 'CAbb', ('TRP', 'CB'): 'CH2', ('TRP', 'C'): 'CObb', ('TRP', 'CD1'): 'aroC', ('TRP', 'CD2'): 'aroC', ('TRP', 'CE2'): 'aroC', ('TRP', 'CE3'): 'aroC', ('TRP', 'CG'): 'aroC', ('TRP', 'CH2'): 'aroC', ('TRP', 'CZ2'): 'aroC', ('TRP', 'CZ3'): 'aroC', ('TRP', 'NE1'): 'Ntrp', ('TRP', 'N'): 'Nbb', ('TRP', 'O'): 'OCbb', ('TYR', 'CA'): 'CAbb', ('TYR', 'CB'): 'CH2', ('TYR', 'C'): 'CObb', ('TYR', 'CD1'): 'aroC', ('TYR', 'CD2'): 'aroC', ('TYR', 'CE1'): 'aroC', ('TYR', 'CE2'): 'aroC', ('TYR', 'CG'): 'aroC', ('TYR', 'CZ'): 'aroC', ('TYR', 'N'): 'Nbb', ('TYR', 'OH'): 'OH', ('TYR', 'O'): 'OCbb', ('VAL', 'CA'): 'CAbb', ('VAL', 'CB'): 'CH1', ('VAL', 'C'): 'CObb', ('VAL', 'CG1'): 'CH3', ('VAL', 'CG2'): 'CH3', ('VAL', 'N'): 'Nbb', ('VAL', 'O'): 'OCbb'}

# Atome type to index
atype2num = {'CNH2': 0, 'Npro': 1, 'CH1': 2, 'CH3': 3, 'CObb': 4, 'aroC': 5, 'OOC': 6, 'Nhis': 7, 'Nlys': 8, 'COO': 9, 'NH2O': 10, 'S': 11, 'Narg': 12, 'OCbb': 13, 'Ntrp': 14, 'Nbb': 15, 'CH2': 16, 'CAbb': 17, 'ONH2': 18, 'OH': 19}

# Pairs of atoms connected.
bonds = {'ALA': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA')], 'ARG': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE'), ('NE', 'CZ'), ('CZ', 'NH1'), ('CZ', 'NH2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD', 'CG'), ('NE', 'CD'), ('CZ', 'NE'), ('NH1', 'CZ'), ('NH2', 'CZ')], 'ASN': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'ND2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('OD1', 'CG'), ('ND2', 'CG')], 'ASP': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('OD1', 'CG'), ('OD2', 'CG')], 'CYS': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'SG'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('SG', 'CB')], 'GLN': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'NE2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD', 'CG'), ('OE1', 'CD'), ('NE2', 'CD')], 'GLU': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD', 'CG'), ('OE1', 'CD'), ('OE2', 'CD')], 'GLY': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'N'), ('C', 'CA'), ('O', 'C')], 'HIS': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CD2', 'NE2'), ('CE1', 'NE2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('ND1', 'CG'), ('CD2', 'CG'), ('CE1', 'ND1'), ('NE2', 'CD2'), ('NE2', 'CE1')], 'ILE': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2'), ('CG2', 'CD1'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG1', 'CB'), ('CG2', 'CB'), ('CD1', 'CG2')], 'LEU': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD1', 'CG'), ('CD2', 'CG')], 'LYS': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD', 'CG'), ('CE', 'CD'), ('NZ', 'CE')], 'MET': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'SD'), ('SD', 'CE'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('SD', 'CG'), ('CE', 'SD')], 'PHE': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CD1', 'CE1'), ('CE1', 'CZ'), ('CG', 'CD2'), ('CD2', 'CE2'), ('CE2', 'CZ'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD1', 'CG'), ('CE1', 'CD1'), ('CZ', 'CE1'), ('CD2', 'CG'), ('CE2', 'CD2'), ('CZ', 'CE2')], 'PRO': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'N'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD', 'CG'), ('N', 'CD')], 'SER': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'OG'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('OG', 'CB')], 'THR': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'OG1'), ('CB', 'CG2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('OG1', 'CB'), ('CG2', 'CB')], 'TRP': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'NE1'), ('CD2', 'CE2'), ('CE2', 'NE1'), ('CD2', 'CE3'), ('CE2', 'CZ2'), ('CE3', 'CZ3'), ('CZ2', 'CH2'), ('CZ3', 'CH2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD1', 'CG'), ('CD2', 'CG'), ('NE1', 'CD1'), ('CE2', 'CD2'), ('NE1', 'CE2'), ('CE3', 'CD2'), ('CZ2', 'CE2'), ('CZ3', 'CE3'), ('CH2', 'CZ2'), ('CH2', 'CZ3')], 'TYR': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CD1', 'CE1'), ('CE1', 'CZ'), ('CG', 'CD2'), ('CD2', 'CE2'), ('CE2', 'CZ'), ('CZ', 'OH'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG', 'CB'), ('CD1', 'CG'), ('CE1', 'CD1'), ('CZ', 'CE1'), ('CD2', 'CG'), ('CE2', 'CD2'), ('CZ', 'CE2'), ('OH', 'CZ')], 'VAL': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2'), ('CA', 'N'), ('C', 'CA'), ('O', 'C'), ('CB', 'CA'), ('CG1', 'CB'), ('CG2', 'CB')]}
