import os,sys
import utils
import glob
import numpy as np

def append_if_missing(contacts_ens,contacts):
    for a in contacts:
        if a not in contacts_ens:
            contacts_ens.append(a)

def make_pocket_pdb(refpdb,outpdb,pocketres,include_ligand=True):
    out = open(outpdb,'w')
    for l in open(refpdb):
        if not l.startswith('ATOM') and not l.startswith('HETATM'): continue
        reschain = l[21]+'.'+l[22:26].strip()
        if reschain in pocketres:
            out.write(l)
        elif l[16:20].strip() == 'LG1' and include_ligand:
            out.write(l)
    out.close()

def get_reschain_from_pdb(pdb,inputpath):
    reschain_in_pocket = []
    for l in open(inputpath+'/'+pdb):
        if not l.startswith('ATOM') and not l.startswith('HETATM'): continue
        reschain = l[21]+'.'+l[22:26].strip()
        if reschain not in reschain_in_pocket:
            reschain_in_pocket.append(reschain)
    return reschain_in_pocket
    
def get_reschain_from_contacts(refpdb):
    resnames,reschains,xyz = utils.read_pdb(refpdb)
    _, _, atms_aa, _ = utils.get_AAtype_properties(include_metal=True)

    xyz_rec = []
    reschain_idx = []
    for i,resname in enumerate(resnames):
        if resname not in utils.residues_and_metals:
            #print("unknown residue: %s, skip"%resname)
            continue
        iaa = utils.residues_and_metals.index(resname)
        for atm in atms_aa[iaa]:
            if atm not in xyz[reschains[i]]: continue
            xyz_rec.append(xyz[reschains[i]][atm])
            reschain_idx.append(reschains[i])

    xyz_rec = np.array(xyz_rec)

    reschain_in_pocket = []
    #ligpdbs = glob.glob('%s/decoy.vina*.pdb'%inputpath)
    #ligpdbs = glob.glob('%s/decoy.GA*.pdb'%inputpath)
    ligpdbs = []
    for key in pdbskeys:
        ligpdbs += glob.glob('%s/%s*pdb'%(inputpath,key))
        
    if len(ligpdbs) == 0:
        #print("No available ligand decoys... return")
        return False
        
    ligatms,_,_,_ = utils.read_params('%s/LG.params'%inputpath)
    
    try:
        # should be always GAdock result
        _, reschains, _xyz_lig = utils.read_pdb(ligpdbs[0],read_ligand=True, aas_allowed=['LG1'])
        ligres = reschains[0]
        xyz_lig = np.array([_xyz_lig[ligres][atm] for atm in ligatms])
    except:
        #print("failed to read GAdock  outputs")
        return False
    
    nlig = len(xyz_lig)
    for pdb in ligpdbs:
        try:
            _, reschains, _xyz_lig = utils.read_pdb(pdb,read_ligand=True, aas_allowed=['LG1'])
        except:
            #print("failed reading %s... skip"%pdb)
            continue
        res = reschains[0]
        xyz_lig = np.array([_xyz_lig[res][atm] for atm in ligatms])
        contacts,_ = utils.get_native_info(xyz_rec,xyz_lig,contact_dist=8.0) #more permissive
        
        contact_res = [reschain_idx[j-nlig] for i,j in contacts]
        
        append_if_missing(reschain_in_pocket,contact_res)
    return reschain_in_pocket
        
def main(tag,pdbskeys,
         proteinpdb='',
         pocketpdb='',
         inputpath = '/net/scratch/hpark/PDBbindset/%s'%tag,
         include_native=True):

    if pocketpdb != '':
        refpdb = '%s/%s'%(inputpath,proteinpdb)
        outpdb = '%s/pocket.%s.pdb'%(inputpath,proteinpdb[:-4])
        reschain_in_pocket = get_reschain_from_pdb(pocketpdb,inputpath)
        
    else:
        if proteinpdb == '':
            refpdb = '%s/%s_protein.pdb'%(inputpath,tag)
            outpdb = '%s/pocket.pdb'%inputpath
        else:
            refpdb = '%s/%s'%(inputpath,proteinpdb)
            outpdb = '%s/pocket.%s.pdb'%(inputpath,proteinpdb[:-4])
        reschain_in_pocket = get_reschain_from_contacts(pdb)
        
    #print("%s: Residue in pocket from %d decoys: "%(tag,len(ligpdbs)))
    #print(reschain_in_pocket)
    make_pocket_pdb(refpdb, outpdb, reschain_in_pocket, include_ligand=True)

if __name__ == "__main__":
    tag = sys.argv[1]
    proteinpdb = sys.argv[2]
    pocketpdb = sys.argv[3]
    main(tag,[proteinpdb[:-4]],proteinpdb,pocketpdb=pocketpdb,include_native=False)
