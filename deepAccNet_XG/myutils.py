import numpy as np
import glob
import os

# Residue number definition
AMINOACID = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',\
             'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',\
             'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(AMINOACID[i], i) for i in range(len(AMINOACID))])
NUCLEICACID = ['ADE','CYT','GUA','THY','URA'] #nucleic acids

METAL = ['CA','ZN','MN','MG','FE','CD','CO']
ALL_AAS = ['UNK'] + AMINOACID + NUCLEICACID + METAL

N_AATYPE = len(ALL_AAS)

# Atom type to index
atype2num = {'CNH2': 0, 'Npro': 1, 'CH1': 2, 'CH3': 3, 'CObb': 4, 'aroC': 5, 'OOC': 6, 'Nhis': 7, 'Nlys': 8, 'COO': 9, 'NH2O': 10, 'S': 11, 'Narg': 12, 'OCbb': 13, 'Ntrp': 14, 'Nbb': 15, 'CH2': 16, 'CAbb': 17, 'ONH2': 18, 'OH': 19}

gentype2num = {'CS':0, 'CS1':1, 'CS2':2,'CS3':3,
               'CD':4, 'CD1':5, 'CD2':6,'CR':7, 'CT':8,
               'CSp':9,'CDp':10,'CRp':11,'CTp':12,'CST':13,'CSQ':14,
               'HO':15,'HN':16,'HS':17,
               # Nitrogen
               'Nam':18, 'Nam2':19, 'Nad':20, 'Nad3':21, 'Nin':22, 'Nim':23,
               'Ngu1':24, 'Ngu2':25, 'NG3':26, 'NG2':27, 'NG21':28,'NG22':29, 'NG1':30, 
               'Ohx':31, 'Oet':32, 'Oal':33, 'Oad':34, 'Oat':35, 'Ofu':36, 'Ont':37, 'OG2':38, 'OG3':39, 'OG31':40,
               #S/P
               'Sth':41, 'Ssl':42, 'SR':43,  'SG2':44, 'SG3':45, 'SG5':46, 'PG3':47, 'PG5':48, 
               # Halogens
               'Br':49, 'I':50, 'F':51, 'Cl':52, 'BrR':53, 'IR':54, 'FR':55, 'ClR':56,
               # Metals
               'Ca2p':57, 'Mg2p':58, 'Mn':59, 'Fe2p':60, 'Fe3p':60, 'Zn2p':61, 'Co2p':62, 'Cu2p':63, 'Cd':64}

# simplified idx
gentype2simple = {'CS':0,'CS1':0,'CS3':0,'CST':0,'CSQ':0,'CSp':0,
                  'CD':1,'CD1':1,'CD2':1,'CDp':1,
                  'CT':2,'CTp':2,
                  'CR':3,'CRp':3,
                  'HN':4,'HO':4,'HS':4,
                  'Nam':5,'Nam2':5,'NG3':5,
                  'Nad':6,'Nad3':6,'Nin':6,'Nim':6,'Ngu1':6,'Ngu2':6,'NG2':6,'NG21':6,'NG22':6,
                  'NG1':7,
                  'Ohx':8,'OG3':8,'Oet':8,'OG31':8,
                  'Oal':9, 'Oad':9, 'Oat':9, 'Ofu':9, 'Ont':9, 'OG2':9,
                  'Sth':10, 'Ssl':10, 'SR':10,  'SG2':10, 'SG3':10, 'SG5':10, 'PG3':11, 'PG5':11, 
                  'F':12, 'Cl':13, 'Br':14, 'I':15, 'FR':12, 'ClR':13, 'BrR':14, 'IR':15, 
                  'Ca2p':16, 'Mg2p':17, 'Mn':18, 'Fe2p':19, 'Fe3p':19, 'Zn2p':20, 'Co2p':21, 'Cu2p':22, 'Cd':23
                  }

def find_gentype2num(at):
    if at in gentype2num:
        return gentype2num[at]
    else:
        return 0 # is this okay?

def findAAindex(aa):
    if aa in ALL_AAS:
        return ALL_AAS.index(aa)
    else:
        return 0 #UNK

def read_params(p,as_list=False,ignore_hisH=True,aaname=None,read_mode='heavy'):
    atms = []
    qs = {}
    atypes = {}
    bnds = []
    
    is_his = False
    repsatm = 0
    nchi = 0
    for l in open(p):
        words = l[:-1].split()
        if l.startswith('AA'):
            if 'HIS' in l: is_his = True
        elif l.startswith('NAME'):
            aaname_read = l[:-1].split()[-1]
            if aaname != None and aaname_read != aaname: return False
            
        if l.startswith('ATOM') and len(words) > 3:
            atm = words[1]
            atype = words[2]
            if atype[0] == 'H':
                if read_mode == 'heavy':
                    continue
                elif atype not in ['Hpol','HNbb','HO','HS','HN']:
                    continue
                elif is_his and (atm in ['HE2','HD1']) and ignore_hisH:
                    continue
                
            if atype == 'VIRT': continue
            atms.append(atm)
            atypes[atm] = atype
            qs[atm] = float(words[4])
            
        elif l.startswith('BOND'):
            a1,a2 = words[1:3]
            if a1 not in atms or a2 not in atms: continue
            border = 1
            if len(words) >= 4:
                # 2 for conjugated/double-bond, 4 for ring aromaticity...
                border = {'1':1,'2':2,'CARBOXY':2,'DELOCALIZED':2,'ARO':4}[words[3]] 
            
            bnds.append((a1,a2,border))
            
        elif l.startswith('NBR_ATOM'):
            repsatm = atms.index(l[:-1].split()[-1])
        elif l.startswith('CHI'):
            nchi += 1
        elif l.startswith('PROTON_CHI'):
            nchi -= 1
            
    # bnds:pass as strings
            
    if as_list:
        qs = [qs[atm] for atm in atms]
        atypes = [atypes[atm] for atm in atms]
    return atms,qs,atypes,bnds,repsatm,nchi

def read_pdb(pdb,read_ligand=False,aas_allowed=[],
             aas_disallowed=[]):
    resnames = []
    reschains = []
    xyz = {}
    atms = {}
    
    for l in open(pdb):
        if not (l.startswith('ATOM') or l.startswith('HETATM')): continue
        atm = l[12:17].strip()
        aa3 = l[17:20].strip()

        if aas_allowed != [] and aa3 not in aas_allowed: continue
            
        reschain = l[21]+'.'+l[22:27].strip()

        if aa3[:2] in METAL: aa3 = aa3[:2]
        if aa3 in AMINOACID:
            if atm == 'CA':
                resnames.append(aa3)
                reschains.append(reschain)
        elif aa3 in NUCLEICACID:
            if atm == "C1'":
                resnames.append(aa3)
                reschains.append(reschain)
        elif aa3 in METAL:
            resnames.append(aa3)
            reschains.append(reschain)
        elif read_ligand and reschain not in reschains:
            resnames.append(aa3)
            reschains.append(reschain)

        if reschain not in xyz:
            xyz[reschain] = {}
            atms[reschain] = []
        #if 'LG1' in l[30:54]:
        #    l = l.replace('LG1','000')
        xyz[reschain][atm] = [float(l[30:38]),float(l[38:46]),float(l[46:54])]
        atms[reschain].append(atm)

    return resnames, reschains, xyz, atms

def defaultparams(aa,
                  datapath='/software/rosetta/latest/database/chemical/residue_type_sets/fa_standard/residue_types',
                  extrapath=''):
    # first search through Rosetta database
    p = None
    if aa in AMINOACID:
        p = '%s/l-caa/%s.params'%(datapath,aa)
    elif aa in NUCLEICACID:
        if aa == 'URA':
            p = '%s/nucleic/rna_phenix/URA_n.params'%(datapath)
        else:
            p = '%s/nucleic/dna/%s.params'%(datapath,aa)
    elif aa in METAL:
        p = '%s/metal_ions/%s.params'%(datapath,aa)
        
    if p != None: return p

    p = '%s/%s.params'%(extrapath,aa)
    if not os.path.exists(p):
        p = '%s/LG.params'%(extrapath)
    if not os.path.exists(p):
        return None
    return p

def fa2gentype(fats,aa=None):
    gts = {'Nbb':'Nad','Npro':'Nad3','NH2O':'Nad','Ntrp':'Nin','Nhis':'Nim','NtrR':'Ngu2','Narg':'Ngu1','Nlys':'Nam',
           'CAbb':'CS1','CObb':'CDp','CH1':'CS1','CH2':'CS2','CH3':'CS3','COO':'CDp','CH0':'CR','aroC':'CR','CNH2':'CDp',
           'OCbb':'Oad','OOC':'Oat','OH':'Ohx','ONH2':'Oad',
           'S':'Ssl','SH1':'Sth',
           'HNbb':'HN','HS':'HS','Hpol':'HO',
           'Phos':'PG5', 'Oet2':'OG3', 'Oet3':'OG3' #Nucleic acids
    }
    
    gents = []
    for fat in fats:
        #special exceptions... (skip deprotonated CYS, etc)
        if aa == 'GLY' and fat == 'CAbb':
            gents.append('CS2')
        elif fat in gentype2num:
            gents.append(fat)
        else:
            gents.append(gts[fat])
    return gents

def get_AAtype_properties(ignore_hisH=True,
                          extrapath='',
                          extrainfo={}):
    qs_aa = {}
    atypes_aa = {}
    atms_aa = {}
    bnds_aa = {}
    repsatm_aa = {}
    
    iaa = 0 #"UNK"
    for aa in AMINOACID+NUCLEICACID+METAL:
        iaa += 1
        p = defaultparams(aa)
        atms,q,atypes,bnds,repsatm,_ = read_params(p)
        atypes_aa[iaa] = fa2gentype([atypes[atm] for atm in atms],aa)
        qs_aa[iaa] = q
        atms_aa[iaa] = atms
        bnds_aa[iaa] = bnds
        if aa in AMINOACID:
            repsatm_aa[iaa] = atms.index('CA')
        else:
            repsatm_aa[iaa] = repsatm

    return qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa

### unused
def read_ligand_pdb(pdb,ligres='LG1',read_H=False):
    xyz = []
    atms = []
    for l in open(pdb):
        if not l.startswith('ATOM') and not l.startswith('HETATM'): continue
        atm = l[12:17].strip()
        aa3 = l[17:20].strip()
        if aa3 != ligres: continue
        if not read_H and atm[0] == 'H': continue

        xyz.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        atms.append(atm)
    xyz = np.array(xyz)
    return atms, xyz

def get_native_info(xyz_r,xyz_l,bnds_l=[],atms_l=[],contact_dist=5.0,shift_nl=True):
    nr = len(xyz_r)
    nl = len(xyz_l)

    # get list of ligand bond connectivity
    if bnds_l != []:
        bnds_l = [(i,j) for i,j in bnds_l]
        angs_l = []
        for i,b1 in enumerate(bnds_l[:-1]):
            for b2 in bnds_l[i+1:]:
                if b1[0] == b2[0]: angs_l.append((b1[1],b2[1]))
                elif b1[0] == b2[0]: angs_l.append((b1[1],b2[1]))
                elif b1[1] == b2[1]: angs_l.append((b1[0],b2[0]))
                elif b1[0] == b2[1]: angs_l.append((b1[1],b2[0]))
                elif b1[1] == b2[0]: angs_l.append((b1[0],b2[1]))
        bnds_l += angs_l
        # just for debugging
        bnds_a = [(atms_l[i],atms_l[j]) for i,j in bnds_l]
    
    dmap = np.array([[np.dot(xyz_l[i]-xyz_r[j],xyz_l[i]-xyz_r[j]) for j in range(nr)] for i in range(nl)])
    dmap = np.sqrt(dmap)
    contacts = np.where(dmap<contact_dist) #
    if shift_nl:
        contacts = [(j,contacts[1][i]+nl) for i,j in enumerate(contacts[0])]
        dco = [dmap[i,j-nl] for i,j in contacts]
    else:
        contacts = [(j,contacts[1][i]) for i,j in enumerate(contacts[0])]
        dco = [dmap[i,j] for i,j in contacts]

    # ligand portion
    dmap_l = np.array([[np.sqrt(np.dot(xyz_l[i]-xyz_l[j],xyz_l[i]-xyz_l[j])) for j in range(nl)] for i in range(nl)])
    contacts_l = np.where(dmap_l<contact_dist)
    contacts_l = [(j,contacts_l[1][i]) for i,j in enumerate(contacts_l[0]) if j<contacts_l[1][i] and ((j,contacts_l[1][i]) not in bnds_l)]
    
    dco += [dmap_l[i,j] for i,j in contacts_l]
    contacts += contacts_l

    return contacts, dco

# AA residue properties
AAprop = {'netq':[0,0,-1,0,-1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
          'nchi':[0,1, 2,2, 3,3,2,2,3,0,3,4,4,4,1,1,2,2,1,1],
          'Kappa':[( 5.000,  2.250,  2.154),
          (11.000,  6.694,  5.141),
          ( 8.000,  3.938,  3.746),
          ( 8.000,  3.938,  4.660),
          ( 6.000,  3.200,  3.428),          
          ( 9.000,  4.840,  4.639),
          ( 9.000,  4.840,  5.592),
          ( 4.000,  3.000,  2.879),
          ( 8.100,  4.000,  2.381),
          ( 8.000,  3.938,  3.841),
          ( 8.000,  3.938,  3.841),
          ( 9.000,  6.125,  5.684),
          ( 8.000,  5.143,  5.389),
          ( 9.091,  4.793,  3.213),
          ( 5.143,  2.344,  1.661),
          ( 6.000,  3.200,  2.809),
          ( 7.000,  3.061,  2.721),
          (10.516,  4.680,  2.737),          
          (10.083,  4.889,  3.324),
          ( 6.000,  1.633,  1.567)]
          }


w
