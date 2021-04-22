import glob
import numpy as np
import copy
import os,sys
from scipy.spatial import distance_matrix
from . import myutils 
#from pyrosetta import *

# pyrosetta
#init("-mute all -score:weights empty.wts -ignore_unrecognized_res")

def sasa_from_xyz(xyz,reschains,atmres_rec):
    #dmtrx
    D = distance_matrix(xyz,xyz)
    cbcounts = np.sum(D<12.0,axis=0)-1.0

    # convert to apprx sasa
    cbnorm = cbcounts/50.0
    sasa_byres = 1.0 - cbnorm**(2.0/3.0)
    sasa_byres = np.clip(sasa_byres,0.0,1.0)

    # by atm
    #print(sasa_byres)
    sasa = [sasa_byres[reschains.index(res)] for res,atm in atmres_rec]
    
    return sasa

def per_atm_lddt(xyz_lig,xyz_rec,dco,contact):
    xyz = np.concatenate([xyz_lig,xyz_rec])
    nco = len(dco)
    natm = len(xyz_lig)
    deltad = np.zeros(nco)
    deltad_per_atm = [[] for i in range(natm)]
    
    for i,(a1,a2) in enumerate(contact): #a1,a2 are lig,rec atmidx
        dv = xyz[a1]-xyz[a2]
        d = np.sqrt(np.dot(dv,dv))
        deltad[i] = abs(dco[i] - d)
        deltad_per_atm[a1].append(deltad[i])
        
    fnat = np.sum(deltad<0.5) + np.sum(deltad<1.0) + np.sum(deltad<2.0) + np.sum(deltad<4.0)
    fnat /= 4.0*(nco+0.001)

    lddt_per_atm = np.zeros(natm)
    for i,col in enumerate(deltad_per_atm):
        col = np.array(col)
        lddt_per_atm[i] = np.sum(col<0.5) + np.sum(col<1.0) + np.sum(col<2.0) + np.sum(col<4.0)
        lddt_per_atm[i] /= (len(col)+0.001)*4.0
    return fnat, lddt_per_atm

def featurize_target_properties(pdb,extrapath="",verbose=False):
    # get receptor info
    extra = {}
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = myutils.get_AAtype_properties(include_metal=True,extrapath=extrapath,
                                                                                 extrainfo=extra)
    resnames,reschains,xyz = myutils.read_pdb(pdb,read_ligand=True,aas_disallowed=["LG1"])
    
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
        
        if resname in extra:
            iaa = len(myutils.residues_and_metals)
            qs, atypes, atms, bnds_, repsatm = extra[resname]
        elif resname in myutils.residues_and_metals:
            iaa = myutils.residues_and_metals.index(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa], repsatm_aa[iaa])
        else:
            if verbose: print("unknown residue: %s, skip"%resname)
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
                print("Warning, abnormal bond distance: ", pdb, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2],d)
                bnds.remove([i1,i2])
                
        bnds = np.array(bnds,dtype=int)
        atmnames.append(atms_r)
        reschains_read.append(reschain)

        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])
            
    xyz_rec = np.array(xyz_rec)

    #cbcounts_,sasa_ = myutils.read_sasa('apo.sasa.txt',reschains)
    #sasa     = [sasa_[res] for res,atm in atmres_rec]
    
    # sasa apprx from coord
    sasa = sasa_from_xyz(xyz_rec[repsatm_idx],
                         reschains_read,atmres_rec)
    
    # sasa calc on-the-fly usnig pyrosetta
    '''
    t0 = time.time()
    pose = pose_from_file(pdb)
    atom_sasa = rosetta.core.id.AtomID_Map_double_t()
    rsd_sasa = rosetta.utility.vector1_double()
    rosetta.core.scoring.calc_per_atom_sasa( pose, atom_sasa, rsd_sasa, 1.4 )
    sasa2 = rosetta.core.scoring.sasa.get_sc_bb_sasa_per_res( pose, atom_sasa )

    sasa_ = {}
    for i in range(pose.size()):
        sasa_[reschains[i]] = sasa2[1][i+1]+sasa2[0][i+1]
        
    t1 = time.time()
    #sasa     = [sasa_[res] for res,atm in atmres_rec]
    '''

    return xyz_rec, aas_rec, atmres_rec, atypes_rec, q_rec, bnds_rec, sasa, residue_idx, repsatm_idx, reschains, atmnames

def defaultfinder(inputpath,pdb):
    k = int(pdb.split('/')[-1][-7:-4])
    if 'cross' in pdb.split('/')[-1]:
        p = '%s/X%02d.params'%(inputpath,k)
    else:
        p = '%s/LG.params'%(inputpath)
    return p
            
def main(tag,verbose=False,decoytypes=['rigid','flex'],
         out=sys.stdout,
         inputpath = '/net/scratch/hpark/PDBbindset/',
         outpath = '/net/scratch/hpark/PDBbindset/features',
         paramsfinder = defaultfinder,
         paramspath = '',
         store_npz=True,
         same_answer=True,
         extrapath='',
         debug=False):

    #inputpath = './'
    #outpath = './features'
    if inputpath[-1] != '/': inputpath+='/'
    if paramspath == '': paramspath = inputpath

    # featurize target properties_

    # featurize variable ligand-decoy features
    ligpdbs = []
    for t in decoytypes:
        ligpdbs += glob.glob('%s/%s*pdb'%(inputpath,t))
        
    #todo: add in GAligdock decoys here
    if verbose: print("featurize %s, %d ligands..."%(tag,len(ligpdbs)))

    # prop
    aas_rec = []
    atypes_rec = []
    charge_rec = []
    bnds_rec = []
    sasa_rec = []
    residue_idx = []
    repsatm_idx = []

    # lig
    xyz_lig = []
    xyz_rec = []
    lddt = []
    fnat = []
    tags = []
    rmsd = []
    q_lig = []
    atypes_lig = []
    bnds_lig = []
    repsatm_lig = []
    chainres = []
    
    nfail = 0

    '''
    if same_answer and os.path.exists('%s/holo.pdb'%(inputpath+tag)):
        print("Read native info")
        p = paramsfinder(paramspath,pdb)
        args = featurize_target_properties(pdb,extrapath=extrapath)
        _, _aas_rec, atmres_rec, _atypes_rec, _charge_rec, _bnds_rec, _sasa_rec, _residue_idx, _repsatm_idx, reschains, atmnames, resnames_read = args
        
        ligatms,_,_,bnds_lig0,_ = myutils.read_params('%s/LG.params'%(p,as_list=True))
        _, reschains, xyz = myutils.read_pdb('%s/holo.pdb'%(inputpath+tag))
        
        ligres = reschains[-1]
        xyz_lig0 = np.array([_xyz[ligres][atm] for atm in ligatms])
        xyz_rec0 = np.array([_xyz[res][atm] for res,atm in atmres_rec])
        bnds_lig0 = np.array([[ligatms.index(a1),ligatms.index(a2)] for a1,a2 in bnds_lig0],dtype=int)
        contacts,dco = myutils.get_native_info(xyz_rec0,xyz_lig0,bnds_lig0,ligatms,shift_nl=True)
    '''
        
    for pdb in ligpdbs:
        args = featurize_target_properties(pdb,extrapath=extrapath)
        #xyz_rec0,atmres_rec = args
        _, _aas_rec, atmres_rec, _atypes_rec, _charge_rec, _bnds_rec, _sasa_rec, _residue_idx, _repsatm_idx, reschains, atmnames = args
        p = paramsfinder(paramspath,pdb)
        pname = pdb.split('/')[-1][:-4]

        try:
            ligatms,_q_lig,_atypes_lig,_bnds_lig,_repsatm_lig = myutils.read_params(p,as_list=True)
            _, reschains, _xyz = myutils.read_pdb(pdb,read_ligand=True)
            ligres = reschains[-1]
            _xyz_lig = np.array([_xyz[ligres][atm] for atm in ligatms])
            _xyz_rec = np.array([_xyz[res][atm] for res,atm in atmres_rec])
            _chainres = [ligres for _ in _xyz_lig] + [res for res,atm in atmres_rec]
            # make sure xyz_lig has same length with reference atms
            _bnds_lig = np.array([[ligatms.index(a1),ligatms.index(a2)] for a1,a2 in _bnds_lig],dtype=int)

        except:
            print("Error occured while reading %s: skip."%pdb)
            continue
        
        if len(ligatms) != len(_xyz_lig):
            sys.exit("Different length b/w ref and decoy ligand atms! %d vs %d in %s"%(len(ligatms),len(_xyz_lig),pdb))

        if same_answer:
            _fnat,lddt_per_atm = per_atm_lddt(_xyz_lig,_xyz_rec,dco,contacts)
        else:
            _fnat = 0.0
            lddt_per_atm = np.zeros(len(_xyz_lig))

        # prop
        aas_rec.append(_aas_rec)
        atypes_rec.append(_atypes_rec)
        charge_rec.append(_charge_rec)
        bnds_rec.append(_bnds_rec)
        sasa_rec.append(_sasa_rec)
        residue_idx.append(_residue_idx)
        repsatm_idx.append(_repsatm_idx)

        # lig
        xyz_rec.append(_xyz_rec)
        xyz_lig.append(_xyz_lig)
        bnds_lig.append(_bnds_lig)
        q_lig.append(_q_lig)
        atypes_lig.append(_atypes_lig)
        repsatm_lig.append(_repsatm_lig)
        
        lddt.append(lddt_per_atm)
        fnat.append(_fnat)
        #rmsd.append(_rmsd)
        tags.append(pname)
        chainres.append(_chainres)
        
    if nfail > 0.5*len(ligpdbs):
        print("too many failed... return none for %s"%tag)
        return

    # store all decoy info into a single file
    if store_npz:
        np.savez("%s/%s.features.npz"%(outpath,tag),
                 #from prop
                 aas=aas_rec,
                 atypes_rec=atypes_rec, #string
                 charge_rec=charge_rec,
                 bnds_rec=bnds_rec,
                 sasa_rec=sasa_rec, #apo
                 residue_idx=residue_idx,
                 repsatm_idx=repsatm_idx,

                 #originaly at lig
                 xyz=xyz_lig,
                 xyz_rec=xyz_rec,
                 atypes_lig=atypes_lig, 
                 charge_lig=q_lig,
                 bnds_lig=bnds_lig,
                 repsatm_lig=repsatm_lig,
                 lddt=lddt,
                 fnat=fnat,
                 rmsd=rmsd,
                 chainres=chainres,
                 name=tags)

if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    tag = sys.argv[1]
    main(tag,verbose=True)
