import glob
import numpy as np
import copy
import os,sys
import scipy
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import myutils

class GridOption:
    def __init__(self,padding,gridsize,option,clash):
        self.padding = padding
        self.gridsize = gridsize
        self.option = option
        self.clash = clash
        self.shellsize=5.0 # through if no contact within this distance

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
             atmnames=atmnames, #[[],[],[],...]
                 
             # per-res (only for receptor)
             repsatm_idx=repsatm_idx,
             reschains=reschains,
             #atmnames=atmnames, #[[],[],[],...]

        )

    return xyz_rec, aas_rec, atmres_rec, atypes_rec, q_rec, bnds_rec, sasa, residue_idx, repsatm_idx, reschains, atmnames

def grid_from_xyz_old(xyzs,xyz_lig,gridsize,
                      clash=1.5,contact=4.0,
                      padding=0.0,
                      option='ligandxyz',
                      gridout=sys.stdout):

    reso = gridsize*0.7
    bmin = [min(xyz_lig[:,k]) for k in range(3)]
    bmax = [max(xyz_lig[:,k]) for k in range(3)]

    imin = [int(bmin[k]/gridsize)-1 for k in range(3)]
    imax = [int(bmax[k]/gridsize)+1 for k in range(3)]

    grids = []
    print("detected %d grid points..."%((imax[0]-imin[0])*(imax[1]-imin[1])*(imax[2]-imin[2])))
    for ix in range(imin[0],imax[0]+1):
        for iy in range(imin[1],imax[1]+1):
            for iz in range(imin[2],imax[2]+1):
                grid = np.array([ix*gridsize,iy*gridsize,iz*gridsize])
                grids.append(grid)
                #i = len(grids)
                #print("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f"%(i,grid[0],grid[1],grid[2]))
    grids = np.array(grids)
    #grids = np.array([[-2.,11.,22.],[-2.,11.,23.],[-1.,11.,22.]])
    nfull = len(grids)

    # first remove clashing grids
    kd      = scipy.spatial.cKDTree(grids)
    excl,incl,indices = [],[],[]
    for xyz in xyzs:
        kd_ca   = scipy.spatial.cKDTree(xyz[None,:])
        excl += kd_ca.query_ball_tree(kd, clash)[0]
        incl += kd_ca.query_ball_tree(kd, contact)[0]
        
    incl = list(np.unique(incl))
    excl = list(np.unique(excl))
    grids = np.array([grid for i,grid in enumerate(grids) if (i in incl and i not in excl)])
    #grids = np.array([grid for i,grid in enumerate(grids) if (i not in excl)])

    if gridout != None:
        for i,grid in enumerate(grids):
            gridout.write("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f\n"%(i,grid[0],grid[1],grid[2]))
    
    print("Search through %d grid points, of %d contact grids %d clash -> %d"%(nfull,len(incl),len(excl),len(grids)))

    if option == 'ligandxyz':
        #regen kdtree
        kd      = scipy.spatial.cKDTree(grids)
        for xyz in xyz_lig:
            kd_ca   = scipy.spatial.cKDTree(xyz[None,:])
            indices += kd_ca.query_ball_tree(kd, reso)[0]
        indices = list(np.unique(indices))
        grids = grids[indices]
        
    elif option == 'ligandcubic':
        pass
    
    #for i,grid in enumerate(grids[indices]):
    #    print("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f"%(i,grid[0],grid[1],grid[2]))
    return grids

def grid_from_xyz(xyzs_rec,xyzs_lig,
                  opt,
                  gridout=None):

    reso = opt.gridsize*0.7
    bmin = np.min(xyzs_lig[:,:]-opt.padding,axis=0)
    bmax = np.max(xyzs_lig[:,:]+opt.padding,axis=0)

    imin = [int(bmin[k]/opt.gridsize)-1 for k in range(3)]
    imax = [int(bmax[k]/opt.gridsize)+1 for k in range(3)]

    grids = []
    print("detected %d grid points..."%((imax[0]-imin[0])*(imax[1]-imin[1])*(imax[2]-imin[2])))
    for ix in range(imin[0],imax[0]+1):
        for iy in range(imin[1],imax[1]+1):
            for iz in range(imin[2],imax[2]+1):
                grid = np.array([ix*opt.gridsize,iy*opt.gridsize,iz*opt.gridsize])
                grids.append(grid)

    grids = np.array(grids)
    nfull = len(grids)

    # Remove clashing or far-off grids
    kd      = scipy.spatial.cKDTree(grids)
    kd_ca   = scipy.spatial.cKDTree(xyzs_rec)
    kd_lig  = scipy.spatial.cKDTree(xyzs_lig)
    
    # take ligand-neighs
    excl = np.concatenate(kd_ca.query_ball_tree(kd, opt.clash)) #clashing
    incl = np.unique(np.concatenate(kd_ca.query_ball_tree(kd, opt.shellsize)))
    ilig = np.unique(np.concatenate(kd_lig.query_ball_tree(kd, opt.padding)))

    interface = np.unique(np.array([i for i in incl if (i not in excl and i in ilig)],dtype=np.int16))
    grids = grids[interface]
    n1 = len(grids)
    
    # filter small segments by default
    #if  == '':
    D = scipy.spatial.distance_matrix(grids,grids)
    graph = csr_matrix((D<(opt.gridsize+0.1)).astype(int))
    n, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    ncl = [sum(labels==k) for k in range(n)]
    biggest = np.unique(np.where(labels==np.argmax(ncl)))
    
    grids = grids[biggest]

    print("Search through %d grid points, of %d contact grids %d clash -> %d, remove outlier -> %d"%(nfull,len(incl),len(excl),n1,len(grids)))

    if gridout != None:
        for i,grid in enumerate(grids):
            gridout.write("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f\n"%(i,grid[0],grid[1],grid[2]))
    return grids

def main(pdb,outprefix,
         recpdb=None,
         gridsize=1.0,
         padding=0.0,
         clash=1.8,
         ligname=None,
         ligchain=None,
         out=sys.stdout,
         gridoption='ligand',
         maskres=[],
         com=[],
         skip_if_exist=True):

    # read relevant motif
    aas, reschains, xyz, atms = myutils.read_pdb(pdb,read_ligand=True)

    gridopt = GridOption(padding,gridsize,gridoption,clash)
    
    if gridoption[:6] == 'ligand':
        if (ligname != None and ligname in aas):
            #i_lig = reschains.index(ligandname)
            reschain_lig = [reschains[aas.index(ligname)]]
        elif ligchain != None:
            reschain_lig = [rc for rc in reschains if rc[0] == ligchain]
        else:
            sys.exit("Unknown ligname or ligchain: ", ligname, ligchain)
        xyz_lig = np.concatenate([list(xyz[rc].values()) for rc in reschain_lig])
        xyz = np.concatenate(np.array([list(xyz[rc].values()) for rc in reschains if rc not in reschain_lig]))

        with open(outprefix+'.grid.pdb','w') as gridout:
            grids = grid_from_xyz(xyz,xyz_lig,gridopt,gridout=gridout)
        out.write("Found %d grid points around ligand\n"%(len(grids)))

    elif gridoption == 'global':
        xyz = [np.array(list(xyz[rc].values()),dtype=np.float32) for rc in reschains if rc not in maskres]
        xyz = np.concatenate(xyz)
        with open(outprefix+'.grid.pdb','w') as gridout:
            grids = grid_from_xyz(xyz,xyz,gridopt,gridout=gridout)
        out.write("Found %d grid points around ligand\n"%(len(grids)))
        
    elif gridoption == 'com':
        xyz = [np.array(list(xyz[rc].values()),dtype=np.float32) for rc in reschains if rc not in maskres]
        xyz = np.concatenate(xyz)

        # com should be passed through input argument
        assert(len(com) == 3)
        com = com[None,:]
        with open(outprefix+'.grid.pdb','w') as gridout:
            grids = grid_from_xyz(xyz,com,gridopt,gridout=gridout)
    else:
        sys.exit("Error, exit")
        
    # featurize target properties_
    
    recnpz = "%s.prop.npz"%(outprefix)
    if recpdb == None:
        recpdb = pdb
    out.write("Featurize receptor info from %s...\n"%recpdb)
    featurize_target_properties(recpdb,recnpz,out)
    
    xyzs = []
    tags = []
    for i,grid in enumerate(grids):
        xyzs.append(grid)
        tags.append("grid%04d"%i)
        
    gridnpz = "%s.lig.npz"%(outprefix)
    np.savez(gridnpz,
             xyz=xyzs,
             name=tags)

if __name__ == "__main__":
    #trainlist = [l[:-1] for l in open(sys.argv[1])]
    #for tag in tags:
    #    main(tag)
    tag = sys.argv[1]
    main(tag,verbose=True)
