import glob
import numpy as np
import copy
import os,sys
from scipy.spatial import distance_matrix
import scipy
import myutils
import motif

def sasa_from_xyz(xyz,reschains,atmres_rec):
    #dmtrx
    D = distance_matrix(xyz,xyz)
    cbcounts = np.sum(D<12.0,axis=0)-1.0

    # convert to apprx sasa
    cbnorm = cbcounts/50.0
    sasa_byres = 1.0 - cbnorm**(2.0/3.0)
    sasa_byres = np.clip(sasa_byres,0.0,1.0)

    # by atm
    sasa = [sasa_byres[reschains.index(res)] for res,atm in atmres_rec]
    
    return sasa

def featurize_target_properties(pdb,npz,out,extrapath="",verbose=False):
    # get receptor info
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = myutils.get_AAtype_properties(extrapath=extrapath,
                                                                                   extrainfo={})
    resnames,reschains,xyz,atms = myutils.read_pdb(pdb,read_ligand=False)
    
    # read in only heavy + hpol atms as lists
    q_rec = []
    atypes_rec = []
    xyz_rec = []
    atmres_rec = []
    aas_rec = []
    bnds_rec = []
    repsatm_idx = []
    residue_idx = []
    atmnames = []

    skipres = []
    reschains_read = []

    for i,resname in enumerate(resnames):
        reschain = reschains[i]

        if resname in myutils.ALL_AAS:
            iaa = myutils.ALL_AAS.index(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa], repsatm_aa[iaa])
        else:
            if verbose: out.write("unknown residue: %s, skip\n"%resname)
            skipres.append(i)
            continue
            
        natm = len(xyz_rec)
        atms_r = []
        for iatm,atm in enumerate(atms):
            is_repsatm = (iatm == repsatm)
            
            if atm not in xyz[reschain]:
                if is_repsatm: return False
                continue

            atms_r.append(atm)
            q_rec.append(qs[atm])
            atypes_rec.append(atypes[iatm])
            aas_rec.append(iaa)
            xyz_rec.append(xyz[reschain][atm])
            atmres_rec.append((reschain,atm))
            residue_idx.append(i)
            if is_repsatm: repsatm_idx.append(natm+iatm)

        bnds = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2 in bnds_ if atm1 in atms_r and atm2 in atms_r]

        # make sure all bonds are right
        for (i1,i2) in copy.copy(bnds):
            dv = np.array(xyz_rec[i1+natm]) - np.array(xyz_rec[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d > 2.0:
                if verbose:
                    out.write("Warning, abnormal bond distance: ", pdb, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2],d+'\n')
                bnds.remove([i1,i2])
                
        bnds = np.array(bnds,dtype=int)
        atmnames += atms_r
        reschains_read.append(reschain)

        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])
            
    xyz_rec = np.array(xyz_rec)

    if len(atmnames) != len(xyz_rec):
        sys.exit('inconsistent anames <=> xyz')

    # sasa apprx from coord
    sasa = sasa_from_xyz(xyz_rec[repsatm_idx],reschains_read,atmres_rec)
    
    np.savez(npz,
             # per-atm
             aas_rec=aas_rec,
             xyz_rec=xyz_rec, #just native info
             atypes_rec=atypes_rec, #string
             charge_rec=q_rec,
             bnds_rec=bnds_rec,
             sasa_rec=sasa, #apo
             residue_idx=residue_idx,
             atmres_rec=atmres_rec,
             atmnames=atmnames, #[[],[],[],...]
                 
             # per-res (only for receptor)
             repsatm_idx=repsatm_idx,
             reschains=reschains,
             #atmnames=atmnames, #[[],[],[],...]

        )

    return xyz_rec, aas_rec, atmres_rec, atypes_rec, q_rec, bnds_rec, sasa, residue_idx, repsatm_idx, reschains, atmnames


def gridize(xyzs_rec,xyzs_lig,
            xyzs_true,xyzs_fake,cats_true,
            gridsize=2.0,
            clash=1.0,padding=4.0):

    # construct grid
    reso = gridsize*0.7
    bmin = [min(xyzs_lig[:,k]) for k in range(3)]
    bmax = [max(xyzs_lig[:,k]) for k in range(3)]

    imin = [int(bmin[k]/gridsize)-1 for k in range(3)]
    imax = [int(bmax[k]/gridsize)+1 for k in range(3)]

    grids = []
    print("detected %d grid points..."%((imax[0]-imin[0])*(imax[1]-imin[1])*(imax[2]-imin[2])))
    for ix in range(imin[0],imax[0]+1):
        for iy in range(imin[1],imax[1]+1):
            for iz in range(imin[2],imax[2]+1):
                grid = np.array([ix*gridsize,iy*gridsize,iz*gridsize])
                grids.append(grid)

    grids = np.array(grids)
    nfull = len(grids)

    # Remove clashing or far-off grids
    kd      = scipy.spatial.cKDTree(grids)
    kd_ca   = scipy.spatial.cKDTree(xyzs_rec)
    kd_lig  = scipy.spatial.cKDTree(xyzs_lig)
    
    # take ligand-neighs
    excl = np.concatenate(kd_ca.query_ball_tree(kd, clash))
    incl = np.concatenate(kd_ca.query_ball_tree(kd, padding))
    ilig = np.concatenate(kd_lig.query_ball_tree(kd, padding))

    interface = np.unique(np.array([i for i in incl if (i not in excl and i in ilig)],dtype=np.int16))
    grids = grids[interface]

    print("Search through %d grid points, of %d contact grids %d clash -> %d"%(nfull,len(incl),len(excl),len(grids)))

    #for i,grid in enumerate(grids):
    #    print("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f"%(i,grid[0],grid[1],grid[2]))

    # refresh kd
    indices_true, indices_fake = [],[]
    kd      = scipy.spatial.cKDTree(grids)
    kd_true   = scipy.spatial.cKDTree(xyzs_true)
    #kd_fake   = scipy.spatial.cKDTree(xyzs_fake)
    indices_true = np.concatenate(kd_true.query_ball_tree(kd, gridsize))
    #indices_fake = np.concatenate(kd_fake.query_ball_tree(kd, grisize))
        
    indices_true = np.array(np.unique(indices_true),dtype=np.int16)
    #indices_fake = list(np.unique(indices_fake))

    dv2xyz = np.array([[g-x for g in grids[indices_true]] for x in xyzs_true]) # grids x numTrue
    d2xyz = np.sum(dv2xyz*dv2xyz,axis=2)
    overlap = np.exp(-d2xyz/gridsize/gridsize)
        
    N = len(motif.MOTIFS)
    label = np.zeros((len(grids),N))

    tags = []
    for o,cat in zip(overlap,cats_true): # motif index
        for j,p in enumerate(o): # grid index
            if p > 0.01:
                label[indices_true[j],cat] = np.sqrt(p)
                #print(j,grids[indices_true[j]], cat,p)

    nlabeled = 0
    for i,l in enumerate(label):
        grid = grids[i]
        #print(grid, max(l))
        if max(l) > 0.01:
            imotif = np.argmax(l)
            B = np.sqrt(max(l))
            nlabeled += 1
            #print("HETATM %4d  ZN  ZN  X  %2d    %8.3f%8.3f%8.3f  1.00  %5.2f"%(i,imotif,grid[0],grid[1],grid[2],B))
        #else:
            #print("HETATM %4d  CA  CA  X   0    %8.3f%8.3f%8.3f  1.00  %5.2f"%(i,grid[0],grid[1],grid[2],0.0))
            

    return grids, label, tags, nlabeled

    # copied from featurize_usage

def read_mol2(mol2f):
    sys.path.insert(0,'/home/hpark/programs/generic_potential')
    
    from Molecule import MoleculeClass
    from BasicClasses import OptionClass
    import Types

    option = OptionClass(['','-s',mol2f])
    molecule = MoleculeClass(mol2f,option)

    option = OptionClass(['','-s',mol2f])
    molecule = MoleculeClass(mol2f,option)

    xyz_lig = np.array(molecule.xyz)
    atypes_lig = [atm.aclass for atm in molecule.atms]
    
    return xyz_lig, atypes_lig, molecule.atms_aro
    
def get_motifs_from_complex(xyz_rec, atypes_rec, ligmol2):
    xyz_lig, atypes_lig, aroatms_lig = read_mol2(ligmol2)

    # ligand -- gentype
    donorclass_gen = [21,22,23,25,27,28,31,32,34,43]
    acceptorclass_gen = [22,26,33,34,36,37,38,39,40,41,42,43,47]
    aliphaticclass_gen = [3,4] #3: CH2, 4: CH3; -> make [4] to be more strict (only CH3)

    D_lig = [i for i,at in enumerate(atypes_lig) if at in donorclass_gen]
    A_lig = [i for i,at in enumerate(atypes_lig) if at in acceptorclass_gen]
    H_lig = [i for i,at in enumerate(atypes_lig) if at in aliphaticclass_gen]
    R_lig = [i for i,at in enumerate(atypes_lig) if i in aroatms_lig]

    # receptor -- AA type
    donorclass_aa = ['OH','Nlys','NH2O','Ntrp','Narg','NtrR','Nbb']
    acceptorclass_aa = ['OH','OOC','OCbb','ONH2','Nhis']
    HRclass_aa = ['CH3','CH2','aroC','CH0','Nhis','Ntrp']
    
    D_rec = [i for i,at in enumerate(atypes_rec) if at in donorclass_aa]
    A_rec = [i for i,at in enumerate(atypes_rec) if at in acceptorclass_aa]
    HR_rec = [i for i,at in enumerate(atypes_rec) if at in HRclass_aa]

    kd_D   = scipy.spatial.cKDTree(xyz_rec[D_rec])
    kd_A   = scipy.spatial.cKDTree(xyz_rec[A_rec])
    kd_HR   = scipy.spatial.cKDTree(xyz_rec[HR_rec])
    
    kd_lig  = scipy.spatial.cKDTree(xyz_lig)

    iA = np.unique(np.concatenate(kd_lig.query_ball_tree(kd_D,3.3)))
    iD = np.unique(np.concatenate(kd_lig.query_ball_tree(kd_A,3.3)))
    iHR = np.unique(np.concatenate(kd_lig.query_ball_tree(kd_HR,5.0)))

    xyzs_m = []
    motifs = []
    for i,xyz in enumerate(xyz_lig):
        mtype = 0
        if i in A_lig and i in D_lig: mtype = 1
        elif i in A_lig and i not in D_lig: mtype = 2
        elif i not in A_lig and i in D_lig: mtype = 3
        elif i in H_lig: mtype = 4
        elif i in R_lig: mtype = 5

        if mtype > 0:
            motifs.append(mtype)
            xyzs_m.append(xyz)

    return xyz_lig, xyzs_m, motifs

def main(recpdb,
         ligmol2,
         gridsize=1.5,
         padding=4.0,
         verbose=False,
         out=sys.stdout,
         inputpath = './',
         outprefix=None,
         masksize=3,
         include_fake=True,
         skip_if_exist=True):

    if inputpath[-1] != '/': inputpath+='/'

    tag = recpdb.split('/')[-1][:-4]
    #recpdb = '%s/%s.pdb'%(inputpath,tag)
    
    # featurize target properties_
    out.write("Read native info\n")

    # read relevant motif
    aas, reschains, xyz, atms = myutils.read_pdb(recpdb) #rosetta processed
    
    # fake sites -- skip initially (append afterwards)
    xyzs_fake = []

    if outprefix == None: outprefix = tag
    npz = "%s.prop.npz"%outprefix
    args = featurize_target_properties(recpdb,npz,out)
    
    if not args:
        print("failed featurizing target properties, %s"%recpdb)
        return

    # numpy arrays (not dictionary)
    _xyzs_rec, _aas_rec, _atmres_rec, _atypes_rec, _charge_rec, _bnds_rec, _sasa_rec, _residue_idx, _repsatm_idx, reschains, atmnames = args

    xyzs_lig, xyzs_motif, cats_motif = get_motifs_from_complex(_xyzs_rec, _atypes_rec, ligmol2)

    if len(xyzs_motif) == 0: return
    
    grids, labels, tags, nlabeled  = gridize(_xyzs_rec, xyzs_lig, xyzs_motif, xyzs_fake,
                                             cats_motif, gridsize=gridsize,padding=padding)

    npz = "%s.grid.npz"%(outprefix)
    np.savez(npz,
             xyz=grids, # N x 3 
             labels=labels, # N x 14, float
             name=tags)

if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    recpdb = sys.argv[1]
    ligmol2 = sys.argv[2]
    main(recpdb,ligmol2,verbose=True)
