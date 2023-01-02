import glob
import numpy as np
import copy
import os,sys
import utils

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

def featurize_target_properties(inputpath,outf,store_npz):
    # get receptor info
    resnames,reschains,xyz = utils.read_pdb('%s/pocket.pdb'%(inputpath))
    qs_aa, atypes_aa, atms_aa, bnds_aa = utils.get_AAtype_properties(include_metal=True)
    
    # get native ligand info -- not used for cross dock
    atms_l,q_lig,atypes_lig,bnds_lig,xyz_lig = None,None,None,None,[]
    if os.path.exists('%s/LG.params'%inputpath):
        atms_l,q_lig,atypes_lig,bnds_lig = utils.read_params('%s/LG.params'%(inputpath),as_list=True)
        _, _, _xyz_lig = utils.read_pdb('%s/ligand.pdb'%(inputpath))
        xyz_lig = np.array([_xyz_lig['X.1'][atm] for atm in atms_l])
        bnds_lig = np.array([[atms_l.index(a1),atms_l.index(a2)] for a1,a2 in bnds_lig],dtype=int)

    # read in only heavy + hpol atms as lists
    q_rec = []
    atypes_rec = []
    xyz_rec = []
    atmres_rec = []
    aas_rec = []
    res_start = []
    atmnames = []
    
    for i,resname in enumerate(resnames):
        iaa = utils.residues_and_metals.index(resname)
        reschain = reschains[i]
        
        natm = len(xyz_rec)
        res_start.append(natm)
        atms_r = []
        for iatm,atm in enumerate(atms_aa[iaa]):
            if iaa> 19:
                print("detected metal: ", iaa, reschain)
            if atm not in xyz[reschain]: continue

            atms_r.append(atm)
            q_rec.append(qs_aa[iaa][atm])
            atypes_rec.append(atypes_aa[iaa][iatm])
            aas_rec.append(iaa)
            xyz_rec.append(xyz[reschain][atm])
            atmres_rec.append((reschain,atm))

        bnds = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2 in bnds_aa[iaa] if atm1 in atms_r and atm2 in atms_r]

        # make sure all bonds are right
        for (i1,i2) in copy.copy(bnds):
            dv = np.array(xyz_rec[i1+natm]) - np.array(xyz_rec[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d > 2.0:
                print("Warning, abnormal bond distance: ", inputpath, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2],d)
                bnds.remove([i1,i2])
                
        bnds = np.array(bnds,dtype=int)
        atmnames.append(atms_r)

        if i == 0:
            bnds_rec = bnds
        elif bnds_aa[iaa] != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])
            
    xyz_rec = np.array(xyz_rec)

    contacts,dco = [],[]
    if len(xyz_lig) > 0:
        contacts,dco = utils.get_native_info(xyz_rec,xyz_lig,bnds_lig,atms_l)
    
    # make sure contacts are correct
    '''
    xyz_cmplx = np.concatenate([xyz_lig,xyz_rec])
    for i,(a1,a2) in enumerate(contacts):
        dv = xyz_cmplx[a1]-xyz_cmplx[a2]
        d2 = np.dot(dv,dv)
        if d2 > 25.0:
            print(a1,a2, np.sqrt(d2))
    '''
    
    #make sure lengths are the same
    #print(len(atypes_rec)+len(atypes_lig) == len(q_rec)+len(q_lig) == len(aas))
    naas = len(utils.residues_and_metals) + 1 #add "ligand type"
    aas = [naas-1 for _ in xyz_lig] + aas_rec
    aas1hot = np.eye(naas)[aas]

    cbcounts_,sasa_ = utils.read_sasa('%s/apo.sasa.txt'%inputpath,reschains)
    cbcounts = [cbcounts_[res] for res,atm in atmres_rec]
    sasa     = [sasa_[res] for res,atm in atmres_rec]

    if store_npz:
        # save
        np.savez(outf,
                 aas=aas1hot,
                 xyz_rec=xyz_rec, #just native info
                 atypes_rec=atypes_rec, #string
                 charge_rec=q_rec,
                 bnds_rec=bnds_rec,
                 cbcounts_rec=cbcounts, #apo
                 sasa_rec=sasa, #apo
                 
                 # per-res (only for receptor) 
                 reschains=reschains,
                 res_start=res_start,
                 atmnames=atmnames,
                 
                 # ligand props -- may be overloaded by lig.npz if differs
                 atypes_lig=atypes_lig, #string
                 charge_lig=q_lig,
                 bnds_lig=bnds_lig,
                 
        )
    
    return xyz_rec, atmres_rec, aas_rec, dco, contacts

def defaultfinder(inputpath,pdb):
    k = int(pdb.split('/')[-1][-7:-4])
    if 'cross' in pdb.split('/')[-1]:
        p = '%s/X%02d.params'%(inputpath,k)
    else:
        p = '%s/LG.params'%(inputpath+tag)
    return p
            
def main(tag,verbose=False,decoytypes=['rigid','flex'],
         out=sys.stdout,
         inputpath = '/net/scratch/hpark/PDBbindset/',
         outpath = '/net/scratch/hpark/PDBbindset/features',
         paramsfinder = defaultfinder,
         paramspath = '',
         store_npz=True):

    #inputpath = './'
    #outpath = './features'
    if inputpath[-1] != '/': inputpath+='/'
    if paramspath == '': paramspath = inputpath+'/'+tag

    # featurize target properties_
    xyz_rec0,atmres_rec,aas_rec,dco,contacts = featurize_target_properties(inputpath+tag,
                                                                          '%s/%s.prop.npz'%(outpath,tag),
                                                                          store_npz)

    # featurize variable ligand-decoy features
    ligpdbs = []

    if 'native' in decoytypes:
        ligpdbs += ['%s/ligand.pdb'%(inputpath+tag)] #native
        decoytypes.remove('native')
        
    if 'rigid' in decoytypes:
        ligpdbs += ['%s/decoy.rigid%03d.pdb'%(inputpath+tag,k) for k in range(30) \
                    if os.path.exists('%s/decoy.rigid%03d.pdb'%(inputpath+tag,k))]
        decoytypes.remove('rigid')
    if 'flex' in decoytypes:
        ligpdbs += ['%s/decoy.flex%03d.pdb'%(inputpath+tag,k) for k in range(60) \
                    if os.path.exists('%s/decoy.flex%03d.pdb'%(inputpath+tag,k))]
        decoytypes.remove('flex')
    if 'cross' in decoytypes:
        ligpdbs += ['%s/decoy.cross%03d.pdb'%(inputpath+tag,k) for k in range(30) \
                    if os.path.exists('%s/decoy.cross%03d.pdb'%(inputpath+tag,k))]
        decoytypes.remove('cross')

    if decoytypes != []:
        for t in decoytypes:
            ligpdbs += glob.glob('%s/%s*pdb'%(inputpath+tag,t))
        
    if len(ligpdbs) < 10:
        if verbose: print("featurize %d ligands... too small1"%len(ligpdbs))
        return
    
    #todo: add in GAligdock decoys here
    if verbose: print("featurize %s, %d ligands..."%(tag,len(ligpdbs)))

    xyz_lig = []
    xyz_rec = []
    lddt = []
    fnat = []
    tags = []
    rmsd = []
    q_lig = []
    atypes_lig = []
    bnds_lig = []
    aas = []
    chainres = []
    
    #if verbose:
    #    out.write("ligatms:"+" ".join(ligatms)+'\n')
    #    out.write("pdb/fnat/per_atm_lddt\n")

    '''
    rmsdf = '%s/rmsd.txt'%inputpath
    rmsd_dict = {'ligand.pdb':0.0}
    for l in open(rmsdf):
        words = l[:-1].split()
        rmsd_dict[words[1]] = float(words[2])
    '''

    nfail = 0
    for pdb in ligpdbs:
        p = paramsfinder(paramspath,pdb)

        '''
        ligatms,_q_lig,_atypes_lig,_bnds_lig = utils.read_params(p,as_list=True)
        _, reschains, _xyz = utils.read_pdb(pdb,read_ligand=True)
        ligres = reschains[-1]
        _xyz_lig = np.array([_xyz[ligres][atm] for atm in ligatms])
        try:
            _xyz_rec = np.array([_xyz[res][atm] for res,atm in atmres_rec])
        except:
            for res,atm in atmres_rec:
                if atm not in _xyz[res]:
                    print(res,atm)
                    break
        '''
        
        try:
            ligatms,_q_lig,_atypes_lig,_bnds_lig = utils.read_params(p,as_list=True)
            _, reschains, _xyz = utils.read_pdb(pdb,read_ligand=True)
            ligres = reschains[-1]
            _xyz_lig = np.array([_xyz[ligres][atm] for atm in ligatms])
            _xyz_rec = np.array([_xyz[res][atm] for res,atm in atmres_rec])
            _chainres = [ligres for _ in _xyz_lig] + [res for res,atm in atmres_rec]
        except:
            print("Error occured while reading %s: skip."%pdb)
            nfail += 1
            continue
        
        # make sure xyz_lig has same length with reference atms
        _bnds_lig = np.array([[ligatms.index(a1),ligatms.index(a2)] for a1,a2 in _bnds_lig],dtype=int)
        
        if len(ligatms) != len(_xyz_lig):
            sys.exit("Different length b/w ref and decoy ligand atms! %d vs %d in %s"%(len(ligatms),len(_xyz_lig),pdb))

        #print(pdb,len(ligatms))

        if 'cross' in pdb.split('/')[-1]:
            _fnat = 0.0
            lddt_per_atm = np.zeros(len(_xyz_lig))
            _rmsd = -1.0 #rmsd_dict[pdb.split('/')[-1]]
        else:
            _fnat,lddt_per_atm = per_atm_lddt(_xyz_lig,_xyz_rec,dco,contacts)
            _rmsd = -1.0 #rmsd_dict[pdb.split('/')[-1]]

        # make sure lddt_per_atm size equals
        if len(lddt_per_atm) != len(_xyz_lig):
            print("lddt size doesn't match with coord len!, skip", tag)
            continue

        naas = len(utils.residues_and_metals) + 1 #add "ligand type"
        aas_lig = [naas-1 for _ in _xyz_lig]
        _aas = aas_lig + aas_rec
        aas1hot = np.eye(naas)[_aas]
        assert(len(_aas) == len(_xyz_rec)+len(_xyz_lig))
        
        xyz_rec.append(_xyz_rec)
        xyz_lig.append(_xyz_lig)
        bnds_lig.append(_bnds_lig)
        q_lig.append(_q_lig)
        atypes_lig.append(_atypes_lig)
        
        lddt.append(lddt_per_atm)
        fnat.append(_fnat)
        rmsd.append(_rmsd)
        aas.append(aas1hot)
        tags.append(pdb[:-4].split('/')[-1])
        chainres.append(_chainres)
        
        if verbose:
            out.write("%s/%s %8.3f %6.4f"%(tag,pdb.split('/')[-1],_rmsd,_fnat)+" %4.2f"*len(lddt_per_atm)%tuple(lddt_per_atm)+'\n')

    if nfail > 0.5*len(ligpdbs):
        print("too many failed... return none for %s"%tag)
        return

    # store all decoy info into a single file
    if store_npz:
        np.savez("%s/%s.lig.npz"%(outpath,tag),
                 xyz=xyz_lig,
                 xyz_rec=xyz_rec,
                 atypes_lig=atypes_lig, 
                 charge_lig=q_lig,
                 bnds_lig=bnds_lig,
                 lddt=lddt,
                 fnat=fnat,
                 rmsd=rmsd,
                 chainres=chainres,
                 # per-res-info (can be squeezed?)
                 aas=aas,
                 #
                 name=tags)

if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    tag = sys.argv[1]
    main(tag,verbose=True)
