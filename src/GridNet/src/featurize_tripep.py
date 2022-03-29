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
        return [],[]

    # mask out -mask~+mask residues
    ires = reschains.index(reschain)
    
    # funky logic to treat inserted residues
    reschains_noins = [rc[:-1] if rc[-1].isupper() else rc for rc in reschains]
    c = reschain.split('.')[0]
    r = int(reschains_noins[reschains.index(reschain)].split('.')[1])

    adj = [rc for i,rc in enumerate(reschains) \
           if reschains_noins[i].split('.')[0] == reschain.split('.')[0]]

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
            indices += kd_ca.query_ball_tree(kd, 4.5)[0]
        direct_contact = list(np.unique(indices))

    return motifxyzs,direct_contact

def get_sites_from_hotspot(txt,aas,xyz,reschains,masksize=3):

    # first detect contigs
    motifs = {}
    
    for l in open(txt):
        words = l[:-1].split()
        reschain = words[0]+'.'+words[1]
        chain = words[0]
        res = int(words[1])
        
        Eb,Eu = float(words[3]),float(words[4])
        if abs(Eb) > 1000 or abs(Eu) > 1000: continue
        if reschain not in reschains: continue #unrecognized hetmol

        # no energy filter
        aa = aas[reschains.index(reschain)]
        if aa not in motif.POSSIBLE_MOTIFS: continue
        
        if chain not in motifs: motifs[chain] = []
        motifs[chain].append(res)

    # find connected or sandwiched cases
    contigs = {chain:[] for chain in motifs}
    n = 0
    for chain in motifs:
        reslist = motifs[chain]
        reslist.sort()
        # sandwich
        for res in copy.copy(reslist):
            if res+2 in reslist and res+1 not in reslist:
                reslist.append(res+1)

        reslist.sort()

        # contig
        regs = myutils.list2region(reslist)
        for reg in regs:
            if len(reg) < 2: continue
            while len(reg) > 2:
                if len(reg) == 2:
                    if chain+'.%d'%(reg[0]-1) in reschains:
                        reg = [reg[0]-1,reg[0],reg[1]]
                    elif chain+'.%d'%(reg[1]+1) in reschains:
                        reg = [reg[0],reg[1],reg[1]+1]
                    else: continue
                    contigs[chain].append(reg)
                else:
                    contigs[chain].append([reg[0],reg[1],reg[2]])
                    reg = reg[3:]
                n += 1

    # store per contig
    sites = []
    for chain in contigs:
        for reg in contigs[chain]:
            # construct grid around region
            rcs = [(chain+'.%d'%res) for res in reg]
            aas_reg = [aas[reschains.index(rc)] for rc in rcs]
            grids = grids_around_reg(rcs,xyz,aas_reg)

            # iter through each possible motif in contig & color "motif" atoms
            labels = np.zeros(len(grids))
            idist = np.zeros(len(grids))
            for aa,rc in zip(aas_reg,rcs):
                if aa not in motif.POSSIBLE_MOTIFS: continue
                
                for v in motif.POSSIBLE_MOTIFS[aa]:
                    mtype = v[0]
                    motif_atms = v[1:]

                    motifxyzs,direct_contact = detect_motif(xyz,rc,reschains,
                                                            motif_atms,masksize)
                    
                    if len(motifxyzs) == 0 or len(direct_contact) == 0: continue
                    
                    cenxyz = motifxyzs[0]
                    # find nearby grid
                    dv = grids-cenxyz
                    d2 = np.mean(dv*dv,axis=1)
                    closest = np.where(d2<1.0)
                    labels[closest] = mtype
                    idist[closest] = 1.0/(1.0+np.sqrt(d2[closest]))

            tag = chain+".%d-%d"%(reg[0],reg[-1])
            excl = [rc for rc in reschains if rc.split('.')[0] == chain] #self-chain
            sites.append((tag, grids, labels, idist, excl))
            
    return sites
            
def grids_around_reg(rcs,xyz,aas):
    xyz_reg = []
    for aa,rc in zip(aas,rcs):
        for atm in xyz[rc]:
            xyz_reg.append(xyz[rc][atm])

    xyz_reg = np.array(xyz_reg)

    xyz_reg = np.array(xyz_reg)
    # make 1.5 ang grids
    grid0 = (xyz_reg/1.5).astype(np.int64)

    # add all within 1 grid points (total 27 points)
    grids = grid0
    for i in range(-1,1):
        for j in range(-1,1):
            for k in range(-1,1):
                kernel = np.array([i,j,k])
                grids = np.concatenate([grids,grid0+kernel])

    grids = np.unique(grids,axis=0)
    grids = 1.5*grids
    return grids

def write_pdb(outf, grids, labels, idist):
    out = open(outf,'w')
    for i,(g,l,ids) in enumerate(zip(grids,labels,idist)):
        atm = ['C','O','N'][int(l>0)+int(l==5)]
        form = 'HETATM %4d  %1s  %4s  %4d    %8.3f%8.3f%8.3f  1.00  %5.2f\n'
        out.write(form%(i,atm,"UNK",i,g[0],g[1],g[2],ids))
    out.close()

def main(trg,verbose=False,
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
    aas, reschains, xyz, atms = myutils.read_pdb('%s/%s.pdb'%(inputpath,trg)) #rosetta processed
    sites = get_sites_from_hotspot('%s/%s.hotspot.txt'%(inputpath,trg),aas,xyz,reschains,masksize)
    #fakes = 
    
    npz = "%s/%s.prop.npz"%(outpath,trg)
    pdb = '%s/%s.pdb'%(inputpath,trg) #rosetta processed
    args = featurize_target_properties(pdb,npz,out)
    if not args:
        print("failed featurizing target properties, %s"%pdb)
        return
    _, _aas_rec, _atmres_rec, _atypes_rec, _charge_rec, _bnds_rec, _sasa_rec, _residue_idx, _repsatm_idx, reschains, atmnames = args

    # store per site; assume receptor xyz is frozen
    grids = []
    labels = []
    tags = []
    excls = []
    
    # true 
    for i,(tag, grid, label, idist, excl) in enumerate(sites):
        print(trg, tag, len(grid), sum(label>0))
        grids.append(grid)
        labels.append(label) #first goes to "none" (maybe used someday)
        tags.append(tag)
        excls.append(excl) # delete these residue-chain when training -- retreive xyz from atmres_rec[:][0]
        #write_pdb(trg+"."+tag+".pdb", grid, label, idist)
        
    if len(tags) == 0:
        print("skip %s as no proper motif found"%trg)
    
    npz = "%s/%s.lig.npz"%(outpath,trg)
    np.savez(npz,
             grid=grids,
             label=labels,
             exclude=excls,
             name=tags)

if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    tag = sys.argv[1]
    main(tag,verbose=True)
