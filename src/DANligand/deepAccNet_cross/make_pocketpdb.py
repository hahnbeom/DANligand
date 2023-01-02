import os,sys
import utils
import glob
import random
import numpy as np

def append_if_missing(contacts_ens,contacts):
    for a in contacts:
        if a not in contacts_ens:
            contacts_ens.append(a)

def make_pocket_pdb(refpdb,outpdb,pocketres):
    out = open(outpdb,'w')
    for l in open(refpdb):
        if not l.startswith('ATOM') and not l.startswith('HETATM'): continue
        reschain = l[21]+'.'+l[22:26].strip()
        if reschain in pocketres:
            out.write(l)
    out.close()
    
def get_ligpdbs(inputpath,pdbskeys,nmax):
    ligpdbs = []
    for key in pdbskeys:
        ligpdbs += glob.glob('%s/%s*pdb'%(inputpath,key))
    if len(ligpdbs) > nmax:
        random.shuffle(ligpdbs)
        ligpdbs = ligpdbs[:nmax]
    return ligpdbs
            
def main(pdbskeys,
         nmax=1000000,
         proteinpdb='',
         inputpath='/net/scratch/hpark/PDBbindset/',
         include_native=True):
    
    resnames,reschains,xyz = utils.read_pdb('%s/%s'%(inputpath,proteinpdb))
    _, _, atms_aa, _ = utils.get_AAtype_properties(include_metal=True)

    xyz_rec = []
    reschain_idx = []
    for i,resname in enumerate(resnames):
        if resname not in utils.residues_and_metals:
            print("unknown residue: %s, skip"%resname)
            continue
        iaa = utils.residues_and_metals.index(resname)
        for atm in atms_aa[iaa]:
            if atm not in xyz[reschains[i]]: continue
            xyz_rec.append(xyz[reschains[i]][atm])
            reschain_idx.append(reschains[i])
    xyz_rec = np.array(xyz_rec)

    ligpdbs = get_ligpdbs(inputpath,pdbskeys,nmax)
    
    if len(ligpdbs) == 0:
        print("No available ligand decoys... return")
        return False
    else:
        pass
    ligpdbs.sort()
    if include_native: ligpdbs += ['%s/ligand.pdb'%inputpath]

    ligres = 'LG1'
    reschain_in_pocket = []
    for pdb in ligpdbs:
        #_,xyz_lig = utils.read_ligand_pdb(pdb,ligres=ligres,read_H=False)
        try:
            _,xyz_lig = utils.read_ligand_pdb(pdb,ligres=ligres,read_H=False)
        except:
            print("failed reading %s... skip"%pdb)
            continue
        contacts,_ = utils.get_native_info(xyz_rec,xyz_lig,contact_dist=15.0,shift_nl=False) #more permissive
        contact_res = []
        for i,j in contacts: 
            if reschain_idx[j] not in contact_res: contact_res.append(reschain_idx[j])
        contact_res.sort()
        
        append_if_missing(reschain_in_pocket,contact_res)
        #print(pdb,reschain_in_pocket)
        
    print("%s: Residue in pocket from %d decoys: "%(inputpath,len(ligpdbs)))
    make_pocket_pdb('%s/%s'%(inputpath,proteinpdb),
                    '%s/pocket.pdb'%inputpath,
                    reschain_in_pocket)

if __name__ == "__main__":
    tag = sys.argv[1]
    proteinpdb = sys.argv[2]
    main(tag,[proteinpdb[:-4]],proteinpdb,False)
