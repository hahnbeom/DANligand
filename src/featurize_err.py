import numpy as np
import copy
import os,sys
sys.path.insert(0, ".")
from .utils import *

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

def read_pocket(pdb, ligname):
    xyz_rec = []
    xyz_lig = []
    resnames = []
    reschains = []
    xyz_by_reschain = {}
    for l in open(pdb):
        if not l.startswith('ATOM') and not l.startswith('HETATM'): continue
        atm = l[11:16].strip()
        resname = l[16:20].strip()
        resno = l[22:26].strip()
        chain = l[21]
        xyz = np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        if resname == ligname:
            xyz_lig.append(xyz)
        elif resname in residues_and_metals:
            xyz_rec.append(xyz)
        else:
            continue
        reschain = chain+'.'+resno
        if reschain not in xyz_by_reschain: xyz_by_reschain[reschain] = {}
        xyz_by_reschain[reschain][atm] = xyz
        
        resnames.append(resname)
        reschains.append(reschain)

    if xyz_lig == []:
        sys.exit("No ligand xyz found from pdb %s! skip."%pdb)
        
    contacts,_ = get_native_info(xyz_rec,xyz_lig,contact_dist=12.0,shift_nl=False) 
    pocket_rc = np.unique([reschains[j] for i,j in contacts])
    pocket_res = [resnames[reschains.index(rc)] for rc in pocket_rc]
    xyz_rec_pocket = {rc:xyz_by_reschain[rc] for rc in pocket_rc}
    
    return pocket_res, pocket_rc, xyz_rec_pocket

def featurize_target_properties(inputpdb, ligname='LG1',
                                outf=None,store_npz=False,extrapath=""):
    # get receptor info
    extra = {}
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = get_AAtype_properties(include_metal=True,extrapath=extrapath,
                                                                                 extrainfo=extra)
    resnames,reschains,xyz = read_pocket(inputpdb, ligname=ligname)
    
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
    resnames_read = []

    for i,resname in enumerate(resnames):
        reschain = reschains[i]
        
        if resname in extra:
            iaa = len(residues_and_metals)
            qs, atypes, atms, bnds_, repsatm = extra[resname]
        elif resname in residues_and_metals:
            iaa = residues_and_metals.index(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa], repsatm_aa[iaa])
        else:
            print("unknown residue: %s, skip"%resname)
            continue
            
        natm = len(xyz_rec)
        atms_r = []
        for iatm,atm in enumerate(atms):
            is_repsatm = (iatm == repsatm)
            
            if atm not in xyz[reschain]:
                if is_repsatm:
                    print("missing repsatm at %s %s -- terminate!"%(reschain,atm))
                    return False
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
        resnames_read.append(resname)

        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])
            
    xyz_rec = np.array(xyz_rec)

    #cbcounts_,sasa_ = read_sasa('%s/apo.sasa.txt'%inputpath,reschains)
    #cbcounts = [cbcounts_[res] for res,atm in atmres_rec]

    sasa_rec = sasa_from_xyz(xyz_rec[repsatm_idx],list(reschains),atmres_rec)
    
    #for i,rc in enumerate(reschains):
    #    print("%-7s %8.3f"%(rc,sasa_rec[i]))

    if store_npz:
        # save
        np.savez(outf,
                 aas=aas_rec,
                 xyz_rec=xyz_rec, #just native info
                 atypes_rec=atypes_rec, #string
                 charge_rec=q_rec,
                 bnds_rec=bnds_rec,
                 cbcounts_rec=cbcounts, #apo
                 sasa_rec=sasa_rec, #apo
                 residue_idx=residue_idx,
                 
                 # per-res (only for receptor)
                 repsatm_idx=repsatm_idx,
                 reschains=reschains,
                 atmnames=atmnames,
                 resnames=resnames_read,
        )
        return atmres_rec
    else:
        return atmres_rec,residue_idx,repsatm_idx,atypes_rec,aas_rec,q_rec,bnds_rec,sasa_rec

def read_options():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("inputpdb",help='input pdb file')
    parser.add_argument('--ligand_name','-l',default='LG1',help='ligand name in pdb & params file')
    parser.add_argument('--params','-p',default='LG.params',help='ligand params file')
    parser.add_argument('--output','-o',default=None,help='output npz file name')
    parser.add_argument('--extra_path',default=None)
    parser.add_argument('--debug',action='store_true',default=False)
        
    option = parser.parse_args()
    #if option.inputpath[-1] != '/': inputpath+='/'

    return option

def main(inputpdbs,p,output,option):

    xyz_lig_, xyz_rec_ = [],[]
    aas_, atypes_, charge_, bnds_, r2a_, repsatm_idx_, sasa_ = [],[],[],[],[],[],[]
    ligatms_, names_ = [],[]
    
    for pdb in inputpdbs:
        try:
            args = featurize_target_properties(pdb, ligname=option.ligname,
                                               extrapath=option.extrapath)

            atmres_rec,residue_idx,repsatm_rec,atypes_rec,aas_rec,q_rec,bnds_rec,sasa_rec = args

            ligatms,q_lig,atypes_lig,bnds_lig,repsatm_lig = read_params(p,as_list=True)
            resnames, reschains, xyz = read_pdb(pdb,read_ligand=True)
            ligres = reschains[resnames.index(option.ligname)]
        
            xyz_lig = np.array([xyz[ligres][atm] for atm in ligatms])
            xyz_rec = np.array([xyz[res][atm] for res,atm in atmres_rec])
        
            bnds_lig = np.array([[ligatms.index(a1),ligatms.index(a2)] for a1,a2 in bnds_lig],dtype=int)

            # concatenate rec & lig here
            naas = len(residues_and_metals)
            aas         = [naas-1 for _ in ligatms] + aas_rec
            atypes      = np.concatenate([atypes_lig, atypes_rec])
            charges     = np.concatenate([q_lig, q_rec])
            bnds        = np.concatenate([bnds_lig,bnds_rec+len(xyz_lig)])
            repsatm_idx = np.concatenate([np.array([repsatm_lig]),np.array(repsatm_rec,dtype=int)+len(xyz_lig)])
            
            r2a         = np.array(residue_idx,dtype=int) + 1 #add ligand as the first residue
            r2a         = np.concatenate([np.array([0 for _ in ligatms]),r2a])
            
            sasa_lig    = np.array([0.5 for _ in xyz_lig]) #neutral value
            sasa        = np.concatenate([sasa_lig,sasa_rec])

        except:
            print("Error occured while reading %s: skip."%pdb)
            if option.debug:
                args = featurize_target_properties(pdb, ligname=option.ligname,
                                                   extrapath=option.extrapath)
                atmres_rec,residue_idx,repsatm_rec,atypes_rec,aas_rec,q_rec,bnds_rec,sasa_rec = args

                ligatms,q_lig,atypes_lig,bnds_lig,repsatm_lig = read_params(p,as_list=True)
                resnames, reschains, xyz = read_pdb(pdb,read_ligand=True)
                ligres = reschains[resnames.index(option.ligname)]
        
                xyz_lig = np.array([xyz[ligres][atm] for atm in ligatms])
                xyz_rec = np.array([xyz[res][atm] for res,atm in atmres_rec])
        
                bnds_lig = np.array([[ligatms.index(a1),ligatms.index(a2)] for a1,a2 in bnds_lig],dtype=int)

                # concatenate rec & lig here
                aas         = [naas-1 for _ in ligatms] + aas_rec
                atypes      = np.concatenate([atypes_lig, atypes_rec])
                charges     = np.concatenate([q_lig, q_rec])
                bnds        = np.concatenate([bnds_lig,bnds_rec+len(xyz_lig)])
                repsatm_idx = np.concatenate([np.array([repsatm_lig]),np.array(repsatm_rec,dtype=int)+len(xyz_lig)])
            
                r2a         = np.array(residue_idx,dtype=int) + 1 #add ligand as the first residue
                r2a         = np.concatenate([np.array([0 for _ in ligatms]),r2a])
            
                sasa_lig    = np.array([0.5 for _ in xyz_lig]) #neutral value
                sasa        = np.concatenate([sasa_lig,sasa_rec])
            return
        
        # make sure xyz_lig has same length with reference atms
        if len(ligatms) != len(xyz_lig):
            sys.exit("Different length b/w ref and decoy ligand atms! %d vs %d in %s"%(len(ligatms),len(xyz_lig),pdb))
            
        # append
        xyz_rec_.append(xyz_rec)
        xyz_lig_.append(xyz_lig)
        aas_.append(aas)
        atypes_.append(atypes)
        charge_.append(charges)
        bnds_.append(bnds)
        r2a_.append(r2a)
        repsatm_idx_.append(repsatm_idx)
        sasa_.append(sasa)
        ligatms_.append(ligatms)
        names_.append(pdb.replace('.pdb',''))

    np.savez(output,
             xyz_lig=xyz_lig_,
             xyz_rec=xyz_rec_,
             aas=aas_,
             atypes=atypes_,
             charge=charge_,
             bnds=bnds_, #
             r2a=r2a_, #
             repsatm_idx=repsatm_idx_,
             sasa=sasa_,
             ligatms=ligatms_,
             name=names_)

if __name__ == "__main__":
    option = read_options()
    inputpdb = option.inputpdb
    p = option.params
    main([inputpdb],p,option.output,option)
