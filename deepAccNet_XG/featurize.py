## this is for v5, extended AA definition

import glob
import numpy as np
import copy
import os,sys
import utilsXG

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

def featurize_target_properties(inputpath,outf,store_npz,extrapath=""):
    # get receptor info
    extra = {}
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = utilsXG.get_AAtype_properties(extrapath=extrapath,
                                                                                 extrainfo=extra)
    resnames,reschains,xyz,_ = utilsXG.read_pdb('%s/pocket.pdb'%(inputpath),read_ligand=True,aas_disallowed=["LG1"])
    
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
        if resname in extra: # UNK
            iaa = 0
            qs, atypes, atms, bnds_, repsatm = extra[resname]
        elif resname in utilsXG.ALL_AAS:
            iaa = utilsXG.findAAindex(resname)# ALL_AAS.index(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa], repsatm_aa[iaa])
        else:
            print("unknown residue: %s, skip"%resname)
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
                print("Warning, abnormal bond distance: ", inputpath, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2],d)
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

    cbcounts_,sasa_ = utilsXG.read_sasa('%s/apo.sasa.txt'%inputpath,reschains)
    cbcounts = [cbcounts_[res] for res,atm in atmres_rec]
    sasa     = [sasa_[res] for res,atm in atmres_rec]

    if store_npz:
        # save
        np.savez(outf,
                 # per-atm
                 aas_rec=aas_rec,
                 xyz_rec=xyz_rec, #just native info
                 atypes_rec=atypes_rec, #string
                 charge_rec=q_rec,
                 bnds_rec=bnds_rec,
                 cbcounts_rec=cbcounts, #apo
                 sasa_rec=sasa, #apo
                 residue_idx=residue_idx,
                 
                 # per-res (only for receptor)
                 repsatm_idx=repsatm_idx,
                 reschains=reschains,
                 atmnames=atmnames, #[[],[],[],...]
                 resnames=resnames_read, 
        )
    
    return xyz_rec, atmres_rec

def read_ligand_params(xyz,aas,chainres,
                       paramsfinder=None,paramskey={},extrapath=''):
    atms_lig = []
    q_lig = []
    atypes_lig = []
    repsatm_lig = []
    bnds_lig = []
    atmres_lig = []

    # concatenate
    natm = 0
    for ires,aa in enumerate(aas):
        rc = chainres[ires]
        if aa in paramskey and paramsfinder != None:
            p = paramsfinder(paramskey[aa])
        else:
            p = utilsXG.defaultparams(aa,extrapath=extrapath)
        atms_aa,qs_aa,atypes_aa,bnds_aa,repsatm_aa = utilsXG.read_params(p,as_list=True)

        # make sure all atom exists
        atms_aa_lig = []
        for atm,q,atype in zip(atms_aa,qs_aa,atypes_aa):
            if atm not in xyz[rc]: continue
            atms_aa_lig.append(atm)
            q_lig.append(q)
            atypes_lig.append(atype)
            atmres_lig.append((rc,atm))
        atms_lig += atms_aa_lig
            
        repsatm_lig.append(repsatm_aa)
        for a1,a2 in bnds_aa:
            bnds_lig.append( (atms_aa_lig.index(a1)+natm, atms_aa_lig.index(a2)+natm) )
            
        natm += len(atms_lig)
        # logic for peptide bonds -- let's just skip for now
        #if ires >

    bnds_lig = np.array(bnds_lig, dtype=int)
        
    return atms_lig, q_lig, atypes_lig, bnds_lig, repsatm_lig, atmres_lig

def get_ligpdbs(decoytypes, inputpath, verbose=False):
    # featurize variable ligand-decoy features
    ligpdbs = []

    if 'native' in decoytypes:
        ligpdbs += ['%s/ligand.pdb'%(inputpath)] #native
        decoytypes.remove('native')
        
    if 'rigid' in decoytypes:
        ligpdbs += ['%s/decoy.rigid%03d.pdb'%(inputpath,k) for k in range(30) \
                    if os.path.exists('%s/decoy.rigid%03d.pdb'%(inputpath,k))]
        decoytypes.remove('rigid')
    if 'flex' in decoytypes:
        ligpdbs += ['%s/decoy.flex%03d.pdb'%(inputpath,k) for k in range(60) \
                    if os.path.exists('%s/decoy.flex%03d.pdb'%(inputpath,k))]
        decoytypes.remove('flex')
    if 'cross' in decoytypes:
        ligpdbs += ['%s/decoy.cross%03d.pdb'%(inputpath,k) for k in range(30) \
                    if os.path.exists('%s/decoy.cross%03d.pdb'%(inputpath,k))]
        decoytypes.remove('cross')
    if 'CM' in decoytypes:
        ligpdbs += ['%s/decoy.CM%03d.pdb'%(inputpath,k) for k in range(50) \
                    if os.path.exists('%s/decoy.CM%03d.pdb'%(inputpath,k))]
        decoytypes.remove('CM')

    if decoytypes != []:
        for t in decoytypes:
            ligpdbs += glob.glob('%s/%s*pdb'%(inputpath,t))
        
    if len(ligpdbs) < 10:
        if verbose: print("featurize %d ligands... too small1"%len(ligpdbs))
        return
    
    #todo: add in GAligdock decoys here
    return ligpdbs
            
def main(tag,verbose=False,decoytypes=['rigid','flex'],
         out=sys.stdout,
         inputpath = '/net/scratch/hpark/PDBbindset/',
         outpath = '/net/scratch/hpark/PDBbindset/features',
         paramsfinder = None,
         paramspath = '',
         refligand = 'ligand.pdb',
         store_npz=True,
         same_answer=True,
         extrapath='',
         debug=False):

    if inputpath[-1] != '/': inputpath+='/'
    if paramspath == '': paramspath = inputpath+'/'+tag

    # get list of ligand pdbs
    ligpdbs = get_ligpdbs(decoytypes, inputpath+tag, verbose)
    if verbose: print("featurize %s, %d ligands..."%(tag,len(ligpdbs)))
    
    if refligand == -1:
        refligand = ligpdbs[0]
    else:
        refligand = inputpath+tag+'/'+refligand

    # featurize target properties_
    args = featurize_target_properties(inputpath+tag,
                                       '%s/%s.prop.npz'%(outpath,tag),
                                       store_npz,
                                       extrapath=extrapath)
    xyz_rec0,atmres_rec = args

    if same_answer:
        _aas, _reschains, _xyz, _atms = utilsXG.read_pdb(refligand,
                                                         read_ligand=True)
        ligchain = _reschains[-1].split('.')[0] # Take the last chain as the ligand chain
        
        _reschains_lig = [a for i,a in enumerate(_reschains) if a[0] == ligchain]
        _aas_lig = [_aas[i] for i,rc in enumerate(_reschains) if rc in _reschains_lig]

        paramskeys = {'LG1':(refligand,paramspath)}
        _ligatms,_,_,_bnds_lig,_,atmres_lig = read_ligand_params(_xyz,_aas_lig,_reschains_lig,paramsfinder,paramskeys,extrapath=paramspath)
        xyz_lig = np.array([_xyz[res][atm] for res,atm in atmres_lig])
        
        contacts,dco = utilsXG.get_native_info(xyz_rec0,xyz_lig,_bnds_lig,_ligatms)

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
    aas_lig = []
    chainres = []
    
    nfail = 0
    for pdb in ligpdbs:
        #p = paramsfinder(paramspath,pdb)
        pname = pdb.split('/')[-1][:-4]

        try:
            # New way 
            _aas, _reschains, _xyz, _atms = utilsXG.read_pdb(pdb,read_ligand=True)
            ligchain = _reschains[-1].split('.')[0] # Take the last chain as the ligand chain
            _reschains_lig = [rc for i,rc in enumerate(_reschains) if rc[0] == ligchain]
            
            # temporary; per-res
            _aas_lig = [_aas[i] for i,rc in enumerate(_reschains) if rc in _reschains_lig]

            paramskeys = {'LG1':(pdb,paramspath)}
            args = read_ligand_params(_xyz,_aas_lig,_reschains_lig,paramsfinder,paramskeys,
                                      extrapath=paramspath)
            _ligatms,_q_lig,_atypes_lig,_bnds_lig,_repsatm_lig,atmres_lig = args 

            _xyz_rec = np.array([_xyz[res][atm] for res,atm in atmres_rec])
            _xyz_lig = np.array([_xyz[res][atm] for res,atm in atmres_lig])
            _aas_ligA = [utilsXG.findAAindex(_aas[_reschains.index(res)]) for res,atm in atmres_lig] #per-atm
            
            # combine
            _chainres = [res for res,atm in atmres_lig] + [res for res,atm in atmres_rec]

        except:
            print("Error occured while reading %s: skip."%pdb)
            nfail += 1
            if debug:
                _aas, _reschains, _xyz, _atms = utilsXG.read_pdb(pdb,read_ligand=True)
                _reschains_lig = [rc for i,rc in enumerate(_reschains) if rc[0] == ligchain]
                _aas_lig = [_aas[i] for i,rc in enumerate(_reschains) if rc in _reschains_lig]
            
                paramskeys = {'LG1':(pdb,paramspath)}
                args = read_ligand_params(_xyz,_aas_lig,_reschains_lig,paramsfinder,paramskeys,
                                          extrapath=paramspath)
                _ligatms,_q_lig,_atypes_lig,_bnds_lig,_repsatm_lig,atmres_lig = args 

                _xyz_rec = np.array([_xyz[res][atm] for res,atm in atmres_rec])
                _xyz_lig = np.array([_xyz[res][atm] for res,atm in atmres_lig])
                _aas_ligA = [utilsXG.findAAindex(_aas[_reschains.index(res)]) for res,atm in atmres_lig]
            
                # combine
                _chainres = [res for res,atm in atmres_lig] + [res for res,atm in atmres_rec]
            continue
        
        if len(_ligatms) != len(_xyz_lig):
            sys.exit("Different length b/w ref and decoy ligand atms! %d vs %d in %s"%(len(_ligatms),len(_xyz_lig),pdb))

        is_nonbinder = (pname.startswith('d.') or 'cross' in pname)
        is_binder = (pname.startswith('l.'))
        if is_nonbinder:
            _fnat = 0.0
            lddt_per_atm = np.zeros(len(_xyz_lig))
            _rmsd = -1.0 #rmsd_dict[pdb.split('/')[-1]]
        elif is_binder:
            _fnat = 1.0 #caution -- this shouldn't be used for accuracy -- take is just for classification purpose
            lddt_per_atm = np.full(len(_xyz_lig),1.0)
            _rmsd = -1.0 
        else:
            _fnat,lddt_per_atm = per_atm_lddt(_xyz_lig,_xyz_rec,dco,contacts)
            _rmsd = -1.0 #rmsd_dict[pdb.split('/')[-1]]

        # make sure lddt_per_atm size equals
        if len(lddt_per_atm) != len(_xyz_lig):
            print("lddt size doesn't match with coord len!, skip", tag)
            continue

        xyz_rec.append(_xyz_rec)
        xyz_lig.append(_xyz_lig)
        aas_lig.append(_aas_ligA) #per-atm-index
        bnds_lig.append(_bnds_lig)
        q_lig.append(_q_lig)
        atypes_lig.append(_atypes_lig)
        repsatm_lig.append(_repsatm_lig)
        
        lddt.append(lddt_per_atm)
        fnat.append(_fnat)
        rmsd.append(_rmsd)
        tags.append(pname)
        chainres.append(_chainres)
        
        if verbose:
            out.write("%s/%s %8.3f %6.4f"%(tag,pdb.split('/')[-1],_rmsd,_fnat)+" %4.2f"*len(lddt_per_atm)%tuple(lddt_per_atm)+'\n')

    if nfail > 0.5*len(ligpdbs):
        print("too many failed... return none for %s"%tag)
        return

    # store all decoy info into a single file
    if store_npz:
        np.savez("%s/%s.lig.npz"%(outpath,tag),
                 aas_lig=aas_lig,
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
