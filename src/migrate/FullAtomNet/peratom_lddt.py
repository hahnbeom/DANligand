import argparse
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    ]

aa2num= {x:i for i,x in enumerate(num2aa)}

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # val
]

# build the "alternate" sc mapping
aa2longalt=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH2"," NH1",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD2"," OD1",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE2"," OE1",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG2"," CG1",  None,  None,  None,  None,  None,  None,  None), # val
]

# build the mapping from atoms in the full rep (Nx14) to the "alternate" rep
long2alt = torch.zeros((20,14),dtype=torch.int64)
aamask = torch.zeros((20,14),dtype=torch.bool)
for i in range(20):
    i_l, i_lalt = aa2long[i],  aa2longalt[i]
    for j,a in enumerate(i_l):
        if (a is None):
            long2alt[i,j] = j
        else:
            long2alt[i,j] = i_lalt.index(a)
            aamask[i,j]=True

# lddt
#   P/Q are Nres x 14 x 3
def lddt(P, Q, atm_mask, cutoff=15):
    startidx = 0
    lddt = torch.zeros( (P.shape[0],P.shape[1]), device=P.device )

    Pij = torch.sqrt(
        torch.sum( torch.square( P[:,:,None,:]-P[atm_mask,:][None,None,:,:] ), dim=-1 ) )
    Qij = torch.sqrt(
        torch.sum( torch.square( Q[:,:,None,:]-Q[atm_mask,:][None,None,:,:] ), dim=-1 ) )

    mask = torch.logical_and(Qij>0,Qij<cutoff)
    mask[torch.logical_not(atm_mask),:] = False # >> valid distances
    delta_PQ = torch.abs(Pij-Qij)

    for distbin in (0.5,1.0,2.0,4.0):
        lddt += 0.25 * torch.sum( (delta_PQ<=distbin)*mask, dim=2 
            ) / (torch.sum( mask, dim=2 )+1e-12)

    return lddt

def alternate_coords( xyz, seq ):
    xyz_alt = torch.zeros_like(xyz)

    # is there a better way to do this?
    xyz_alt.scatter_(1, long2alt[seq,:,None].repeat(1,1,3),xyz)

    return xyz_alt

def readpdb(filename):
    lines = open(filename,'r').readlines()
    resids = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16]==" CA "]
    resids = {r:i for i,r in enumerate(resids)}

    seq = torch.zeros( (len(resids)), dtype=torch.long )
    xyz = torch.zeros( (len(resids),14,3) )
    for l in lines:
        if (l[:4]!="ATOM"):
            continue
        resid,atom,aa = int(l[22:26]),l[12:16],l[17:20]
        idx = resids[resid]
        if (atom==" CA "):
            seq[idx] = aa2num[aa]
        for i,atmtgt in enumerate(aa2long[aa2num[aa]]):
            if (atmtgt==atom):
                xyz[idx,i,:] = torch.tensor([float(l[30:38]), float(l[38:46]), float(l[46:54])])

    return (xyz, seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # checkpointing parameters
    parser.add_argument('--native', type=str, required=True, help="Native PDB")
    parser.add_argument('--pdbs', type=str, nargs='+', required=True, help="PDBs")
    FLAGS = parser.parse_args()

    nxyz,nseq = readpdb(FLAGS.native)
    mask = aamask[nseq,:]

    for i in range(len(FLAGS.pdbs)):
        xyz,seq = readpdb(FLAGS.pdbs[i])
        assert(torch.equal(nseq,seq))

        lddt_i = lddt(xyz,nxyz,mask)

        xyzalt = alternate_coords( xyz, seq )
        lddt_i = torch.max(
            lddt_i,
            lddt(xyzalt,nxyz,mask)
        )

        for i,s in enumerate(seq):
            for j,a in enumerate(aa2longalt[s]):
                if a is not None:
                    print (i,a,lddt_i[i,j])