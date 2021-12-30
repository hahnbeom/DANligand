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

def detect_motif(xyz,reschain,reschains,motif_atms,masksize=3):
    # skip if any atom doesn't exist
    is_bb_motif = ('O' in motif_atms)
    try:
        motifxyzs = np.array([xyz[reschain][atm] for atm in motif_atms])
        bbxyzs    = np.array([xyz[reschain][atm] for atm in ['N','CA','C']])
        Hxyz,Oxyz = np.array([xyz[reschain]['H']]),np.array([xyz[reschain]['O']])
    except:
        return [],[],[],[]

    # mask out -mask~+mask residues
    ires = reschains.index(reschain)
    
    # funky logic to treat inserted residues
    reschains_noins = [rc[:-1] if rc[-1].isupper() else rc for rc in reschains]
    c = reschain.split('.')[0]
    r = int(reschains_noins[reschains.index(reschain)].split('.')[1])

    if masksize == 999:
        adj = [rc for i,rc in enumerate(reschains) \
               if reschains_noins[i].split('.')[0] == reschain.split('.')[0]]
    else:
        b = max(0,ires-masksize)
        e = min(len(reschains)-1,ires+masksize)
        adj = [rc for i,rc in enumerate(reschains) \
               if abs(int(reschains_noins[i].split('.')[1])-r) < masksize and reschains_noins[i].split('.')[0]==c ]

    if is_bb_motif:
        # append only if H-bond exists
        Oxyz_env = np.array([list(xyz[rc]['O']) for rc in reschains if rc not in adj and 'O' in xyz[rc]])
        Hxyz_env = np.array([list(xyz[rc]['H']) for rc in reschains if rc not in adj and 'H' in xyz[rc]])
        kd1     = scipy.spatial.cKDTree(Oxyz_env)
        kd_hb1  = scipy.spatial.cKDTree(Hxyz)
        kd2     = scipy.spatial.cKDTree(Hxyz_env)
        kd_hb2  = scipy.spatial.cKDTree(Oxyz)
        
        direct_contact = kd_hb1.query_ball_tree(kd1, 2.0)[0] +kd_hb2.query_ball_tree(kd2, 2.0)[0]
    else:
        xyz_env = np.array([list(xyz[rc].values()) for rc in reschains if rc not in adj])
        xyz_env = np.concatenate(xyz_env,axis=0)

        rc_env = np.array([[rc for a in xyz[rc]] for rc in reschains if rc not in adj])
        rc_env = np.concatenate(rc_env,axis=-1)
        
        indices = []
        kd      = scipy.spatial.cKDTree(xyz_env)
        for xyz in motifxyzs:
            kd_ca   = scipy.spatial.cKDTree(xyz[None,:])
            indices += kd_ca.query_ball_tree(kd, 3.5)[0]
        direct_contact = list(np.unique(indices))
    
    #if len(direct_contact) > 0 and is_bb_motif:
    #    print(reschain, motif_atms, direct_contact)

    return motifxyzs,bbxyzs,direct_contact,adj

def get_motifs_from_hotspot(txt,aas,xyz,reschains,masksize=3):
    motifs = []
    
    for l in open(txt):
        words = l[:-1].split()
        reschain = words[0]+'.'+words[1]
        
        Eb,Eu = float(words[2]),float(words[3])
        if abs(Eb) > 1000 or abs(Eu) > 1000: continue
        if reschain not in reschains: continue #unrecognized hetmol

        # first filter by energy
        dE = float(words[4])
        dEcut = -2.0

        aa = aas[reschains.index(reschain)]
        if aa not in motif.POSSIBLE_MOTIFS: continue
        
        if aa in ['CYS','SER']:
            dEcut = -1.5
        elif aa in ['PHE','TYR','ARG']:
            dEcut = -3.0

        if dE > dEcut: continue
        
        # then by motif contact
        for v in motif.POSSIBLE_MOTIFS[aa]:
            mtype = v[0]
            motif_atms = v[1:]
            #if motif.MOTIFS[mtype] == 'bb': continue
            motifxyzs,bbxyz,direct_contact,excl = detect_motif(xyz,reschain,reschains,
                                                               motif_atms,masksize)

            if len(motifxyzs) == 0: continue
            
            m = motif.MotifClass(motifxyzs,mtype)
            m.xyz2frame() #xyz -> q,R,T

            mxyz = m.T
            R = m.R # 3x3 np.array
            mrot = np.array(m.qs_symm) # multiple quaternions (for symm)
            
            #print(reschain,mxyz,bbxyz)
            
            #neighidx = [i for i,rc in enumerate(reschains) if rc in neighs20]
            if len(direct_contact) > 0:
                motifs.append((reschain,mtype,mxyz,bbxyz,mrot,R,excl))#,neighidx))
                
    return motifs

def main(tag,verbose=False,
         out=sys.stdout,
         inputpath = '/home/hpark/decoyset/PDBbind/2018/PPset',
         outpath = './',
         masksize=3,
         include_fake=True,
         skip_if_exist=True):

    if inputpath[-1] != '/': inputpath+='/'

    # featurize target properties_
    out.write("Read native info\n")

    # read relevant motif
    #aas, reschains, xyz, atms = myutils.read_pdb('%s/%s.ent_0001.pdb'%(inputpath,tag)) #rosetta processed
    aas, reschains, xyz, atms = myutils.read_pdb('%s/%s.pdb'%(inputpath,tag)) #rosetta processed
    motifs = get_motifs_from_hotspot('%s/%s.txt'%(inputpath,tag),aas,xyz,reschains,masksize)
    
    # find random fake sites... 
    fakes = []
    if os.path.exists('%s/fake/%s.fakesites.npz'%(inputpath,tag)) and include_fake:
        fakes = np.load('%s/fake/%s.fakesites.npz'%(inputpath,tag))['fakesites']
        
    npz = "%s/%s.prop.npz"%(outpath,tag)
    pdb = '%s/%s.pdb'%(inputpath,tag) #rosetta processed
    args = featurize_target_properties(pdb,npz,out)
    if not args:
        print("failed featurizing target properties, %s"%pdb)
        return
    _, _aas_rec, _atmres_rec, _atypes_rec, _charge_rec, _bnds_rec, _sasa_rec, _residue_idx, _repsatm_idx, reschains, atmnames = args
        
    # store per metal; assume receptor xyz is frozen
    xyzs = []
    xyzs_bb = []
    rots = []
    cats = []
    tags = []
    excls = []
    bases = []
    
    # true 
    for i,(reschain,cat,mxyz,mxyz_bb,mrot,R,excl) in enumerate(motifs):
        xyzs.append(mxyz)
        xyzs_bb.append(mxyz_bb)
        rots.append(mrot)
        bases.append(R)
        cats.append(cat) #first goes to "none" (maybe used someday)
        tags.append(tag+'.'+reschain+'.%02d'%cat)
        excls.append(excl) # delete these residue-chain when training -- retreive xyz from atmres_rec[:][0]
        
    #np.random.shuffle(fakes)
    if len(fakes) > 0:
        nsel = min(len(fakes),50) # pick max 50 fake sites
        isel = np.random.choice(len(fakes),nsel)
    
        O = np.array([[1.0,0.0,0.0,0.0]])
        I = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        for i in isel:
            mxyz = np.expand_dims(fakes[i],axis=0) #real points are 2-dim
            xyzs.append(mxyz)
            xyzs_bb.append(mxyz_bb) #is this okay?
            rots.append(O)
            cats.append(0) # None
            excls.append([]) # None
            bases.append(I)
            tags.append(tag+'.fake.00')
        
    print(tag,len(tags))
    if len(tags) == 0:
        print("skip %s as no proper motif found"%tag)
    
    npz = "%s/%s.lig.npz"%(outpath,tag)
    np.savez(npz,
             xyz=xyzs,
             xyz_bb=xyzs_bb, #list of (3atm x 3) in real coordiate (not relative to motif-center)
             rot=rots,
             cat=cats,
             bases=bases,
             exclude=excls,
             name=tags)
    #return npzs

if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    tag = sys.argv[1]
    main(tag,verbose=True)
