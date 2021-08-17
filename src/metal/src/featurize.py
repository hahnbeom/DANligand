import glob
import numpy as np
import copy
import os,sys
from scipy.spatial import distance_matrix
import myutils

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
    resnames,reschains,xyz,atms = myutils.read_pdb(pdb,read_ligand=False,
                                                   aas_disallowed=myutils.METAL)
    
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
             atmnames=atmnames, #[[],[],[],...]
                 
             # per-res (only for receptor)
             repsatm_idx=repsatm_idx,
             reschains=reschains,
             #atmnames=atmnames, #[[],[],[],...]

        )

    return xyz_rec, aas_rec, atmres_rec, atypes_rec, q_rec, bnds_rec, sasa, residue_idx, repsatm_idx, reschains, atmnames

def main(tag,verbose=False,
         out=sys.stdout,
         inputpath = '/home/hpark/decoyset/metalpdbs',
         outpath = './',
         truedata=None,
         fakedata=None,
         skip_if_exist=True):

    # load
    if truedata == None:
        truedata = np.load('%s/truesites_ncoord3.npz'%inputpath,allow_pickle=True)['sites'].item()[tag]
    if fakedata == None:
        fakedata = np.load('%s/decoysites_ncoord3.npz'%inputpath,allow_pickle=True)['sites'].item()[tag]
        
    if inputpath[-1] != '/': inputpath+='/'

    # featurize target properties_
    out.write("Read native info\n")
    
    npz = "%s/%s.prop.npz"%(outpath,tag)
    pdb = '%s/%s.pdb'%(inputpath,tag)
    args = featurize_target_properties(pdb,npz,out)
    
    _, _aas_rec, _atmres_rec, _atypes_rec, _charge_rec, _bnds_rec, _sasa_rec, _residue_idx, _repsatm_idx, reschains, atmnames = args
    #_, reschains, xyz, atms = myutils.read_pdb('%s.pdb'%(inputpath+tag))
        
    # store per metal; assume receptor xyz is frozen

    metalxyz = []
    metalidx = []
    tags = []
    # true metalse
    print(tag,truedata,fakedata)
    for i,(xyz,m) in enumerate(truedata):
        metalxyz.append(xyz)
        metalidx.append(myutils.METAL.index(m)+1)
        tags.append(tag+'.%dT'%i)
        
    # fake sites
    for j,(chainres,xyz,m) in enumerate(fakedata):
        metalxyz.append(xyz)
        metalidx.append(0)
        tags.append(tag+'.%dF'%(len(tags)))
            
    npz = "%s/%s.lig.npz"%(outpath,tag)
    np.savez(npz,
             xyz=metalxyz,
             metalidx=metalidx,
             name=tags)
    #return npzs

if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    tag = sys.argv[1]
    main(tag,verbose=True)
