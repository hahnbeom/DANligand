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
            clash=1.0,contact=3.0):

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
    
    excl = np.concatenate(kd_ca.query_ball_tree(kd, clash))
    incl = np.concatenate(kd_ca.query_ball_tree(kd, contact))
    ilig = np.concatenate(kd_lig.query_ball_tree(kd, contact))
    
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
    #overlap = np.exp(-d2xyz)
        
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

def detect_motif(xyz,reschain,reschains,motif_atms,masksize=999):
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

    if masksize >= 999:
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
        
        Eb,Eu = float(words[3]),float(words[4])
        if abs(Eb) > 1000 or abs(Eu) > 1000: continue
        if reschain not in reschains: continue #unrecognized hetmol

        # first filter by energy
        dE = float(words[5])
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
            if mtype > 13: continue
            motif_atms = v[1:]
            #if motif.MOTIFS[mtype] == 'bb': continue
            motifxyzs,bbxyz,direct_contact,excl = detect_motif(xyz,reschain,reschains,
                                                               motif_atms,masksize)

            if len(motifxyzs) == 0: continue
            
            m = motif.MotifClass(motifxyzs,mtype)
            m.xyz2frame() #xyz -> q,R,T

            mxyz = m.T
            if len(direct_contact) > 0:
                motifs.append((reschain,mtype,mxyz))#,neighidx))
                
    return motifs

def main(tag,verbose=False,
         out=sys.stdout,
         inputpath = '/home/hpark/data/HmapMine/inputs/',
         outpath = './',
         masksize=3,
         include_fake=True,
         skip_if_exist=True):

    if inputpath[-1] != '/': inputpath+='/'

    # featurize target properties_
    out.write("Read native info\n")

    # read relevant motif
    aas, reschains, xyz, atms = myutils.read_pdb('%s/%s.pdb'%(inputpath,tag)) #rosetta processed
    motifs = get_motifs_from_hotspot('%s/%s.hotspot.txt'%(inputpath,tag),aas,xyz,reschains,masksize)
    
    # fake sites -- skip initially (append afterwards)
    xyzs_fake = []
    #if os.path.exists('%s/fake/%s.fakesites.npz'%(inputpath,tag)) and include_fake:
    #    xyzs_fake = np.load('%s/fake/%s.fakesites.npz'%(inputpath,tag))['fakesites']
        
    npz = "%s/%s.prop.npz"%(outpath,tag)
    pdb = '%s/%s.pdb'%(inputpath,tag) #rosetta processed
    args = featurize_target_properties(pdb,npz,out)
    
    if not args:
        print("failed featurizing target properties, %s"%pdb)
        return
    
    _, _aas_rec, _atmres_rec, _atypes_rec, _charge_rec, _bnds_rec, _sasa_rec, _residue_idx, _repsatm_idx, reschains, atmnames = args
        

    # gridize per virtual-chain & store as npz
    chains = np.unique([m[0].split('.')[0] for m in motifs])
    
    for chain in chains:
        xyzs_true = np.array([m[2] for m in motifs if m[0].split('.')[0] == chain])
        cats_true = np.array([m[1] for m in motifs if m[0].split('.')[0] == chain])
        xyzs_rec = np.concatenate([list(xyz[rc].values()) for rc in xyz if rc.split('.')[0] != chain])
        xyzs_lig = np.concatenate([list(xyz[rc].values()) for rc in xyz if rc.split('.')[0] == chain])

        if len(xyzs_true) == 0: continue
        
        grids, labels, tags, nlabeled  = gridize(xyzs_rec, xyzs_lig, xyzs_true, xyzs_fake, cats_true, gridsize=2.5)
        print(chain, nlabeled)

        npz = "%s/%s.%s.grid.npz"%(outpath,tag,chain)
        np.savez(npz,
                 xyz=grids, # N x 3 
                 labels=labels, # N x 14, float
                 name=tags)

if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    tag = sys.argv[1]
    main(tag,verbose=True)
