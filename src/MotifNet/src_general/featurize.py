## this is for v5, extended AA definition

import numpy as np
import copy
import os,sys
import myutils

def featurize_target_properties(pdb,outf,store_npz):
    # get receptor info
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = myutils.get_AAtype_properties()
    resnames,reschains,xyz,_ = myutils.read_pdb(pdb)
    
    # read in only heavy + hpol atms as lists
    q_rec = []
    atypes_rec = []
    xyz_rec = []
    atmres_rec = []
    aas_rec = []
    bnds_rec = []
    borders_rec = []
    repsatm_idx = []
    residue_idx = []
    atmnames = []
    resnames_read = []
    iaas = []

    for i,resname in enumerate(resnames):
        reschain = reschains[i]
        if resname in myutils.ALL_AAS:
            iaa = myutils.findAAindex(resname)# ALL_AAS.index(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa],
                                                bnds_aa[iaa], repsatm_aa[iaa])
        else:
            print("unknown residue: %s, skip"%resname)
            continue
            
        natm = len(xyz_rec)
        atms_r = []
        iaas.append(iaa)
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

        bnds = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2,_ in bnds_ if atm1 in atms_r and atm2 in atms_r]
        borders = [border for atm1,atm2,border in bnds_ if atm1 in atms_r and atm2 in atms_r]

        # make sure all bonds are right
        incl = []
        for ib,(i1,i2) in enumerate(bnds):
            dv = np.array(xyz_rec[i1+natm]) - np.array(xyz_rec[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d < 2.0:
                incl.append(ib)
            else:
                print("Warning, abnormal bond distance: ", pdb, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2], d)

        bnds = np.array(bnds,dtype=int)[incl]
        borders = np.array(borders,dtype=int)[incl]
        
        # add peptide bond -- skip...
        '''
        if i > 0:
            prvrc = reschains[i-1]
            if 'N' in xyz[reschain] and 'C' in xyz[prvrc]:
                xyzN = xyz[reschain]['N']
                xyzC = xyz[prvrc]['C']
                dv = xyzN-xyzC
                d = np.sqrt(np.dot(dv,dv))
        '''

        # by residue
        atmnames.append(atms_r)
        resnames_read.append(resname)

        if i == 0:
            bnds_rec = bnds
            borders_rec = borders
        elif bnds_ != []:
            bnds += natm #offset for the residue
            bnds_rec = np.concatenate([bnds_rec,bnds])
            borders_rec = np.concatenate([borders_rec,borders])
            
    xyz_rec = np.array(xyz_rec)

    
    if store_npz:
        # save
        np.savez(outf,
                 # per-atm
                 aas_rec=aas_rec,
                 xyz_rec=xyz_rec, #just native info
                 atypes_rec=atypes_rec, #string
                 charge_rec=q_rec,
                 bnds_rec=bnds_rec,
                 bndorders=borders_rec,
                 residue_idx=residue_idx,
                 reschains=reschains,
                 
                 # per-res 
                 repsatm_idx=repsatm_idx,
                 atmnames=atmnames, #[[],[],[],...]
                 resnames=resnames_read,
        )
        
def featurize_target_properties2(pdb, read_ligand=False):
    
    #print("readligand", read_ligand)
    # get receptor info
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = myutils.get_AAtype_properties()
    resnames,reschains,xyz,_ = myutils.read_pdb(pdb, read_ligand=read_ligand)
    
    # read in only heavy + hpol atms as lists
    q_rec = []
    atypes_rec = []
    xyz_rec = []
    atmres_rec = []
    aas_rec = []
    bnds_rec = []
    borders_rec = []
    repsatm_idx = []
    residue_idx = []
    atmnames = []
    resnames_read = []
    iaas = []

    for i,resname in enumerate(resnames):
        reschain = reschains[i]
        #print(resname, resname in myutils.ALL_AAS)
        if resname in myutils.ALL_AAS:
            iaa = myutils.findAAindex(resname)# ALL_AAS.index(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa],
                                                bnds_aa[iaa], repsatm_aa[iaa])
        else:
            print("unknown residue: %s, skip"%resname)
            continue
            
        natm = len(xyz_rec)
        atms_r = []
        iaas.append(iaa)
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

        bnds = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2,_ in bnds_ if atm1 in atms_r and atm2 in atms_r]
        borders = [border for atm1,atm2,border in bnds_ if atm1 in atms_r and atm2 in atms_r]

        # make sure all bonds are right
        incl = []
        for ib,(i1,i2) in enumerate(bnds):
            dv = np.array(xyz_rec[i1+natm]) - np.array(xyz_rec[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d < 2.0:
                incl.append(ib)
            else:
                pass
                #print("Warning, abnormal bond distance: ", pdb, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2], d)

        bnds = np.array(bnds,dtype=int)[incl]
        borders = np.array(borders,dtype=int)[incl]

        # by residue
        atmnames.append(atms_r)
        resnames_read.append(resname)

        if i == 0:
            bnds_rec = bnds
            borders_rec = borders
        elif bnds_ != []:
            bnds += natm #offset for the residue
            bnds_rec = np.concatenate([bnds_rec,bnds])
            borders_rec = np.concatenate([borders_rec,borders])
            
    xyz_rec = np.array(xyz_rec)

    
    return {"aas_rec":aas_rec,
            "xyz_rec":xyz_rec, #just native info
            "atypes_rec":atypes_rec, #string
            "charge_rec":q_rec,
            "bnds_rec":bnds_rec,
            "bndorders":borders_rec,
            "residue_idx":residue_idx,
            "reschains":reschains,
            "repsatm_idx":repsatm_idx,
            "atmnames":atmnames, #[[],[],[],...]
            "resnames":resnames_read}
    
def main(pdb,verbose=False,
         out=sys.stdout,
         inputpath = '/net/scratch/hpark/PDBbindset/',
         outpath = '/net/scratch/hpark/PDBbindset/features',
         store_npz=True,
         debug=False):

    # featurize target properties
    tag = pdb.split('/')[-1][:-4]
    featurize_target_properties('%s/%s'%(inputpath,pdb),
                                '%s/%s.prop.npz'%(outpath,tag),
                                store_npz)
    
if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    tag = sys.argv[1]
    main(tag,verbose=True)
