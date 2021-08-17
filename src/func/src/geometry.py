import numpy as np

BBMOTIF = (5,'C','O','CA')

POSSIBLE_MOTIFS = \
    {#'CYS':[('SG1','HG','CB'),BBMOTIF],
        'ASP':[(1,'CG','OD1','CB'),BBMOTIF],
        'GLU':[(1,'CD','OE1','CG'),BBMOTIF],
        'PHE':[(9,'CG','CD1','CD2'),BBMOTIF],
        'HIS':[(7,'ND1','HD1','CG'),(7,'NE2','HE2','CD2'), #donor
            (8,'ND1','CE1','CG'),(8,'NE2','CE1','CD2'), #acceptor
            BBMOTIF], # 
        'LYS':[(3,'NZ','1HZ','CE'),BBMOTIF],
        'ASN':[(6,'CG','OD1','CB'),BBMOTIF],
        'SER':[(4,'OG','HG','CB'),BBMOTIF],
        'THR':[(4,'OG1','HG1','CB'),BBMOTIF],
        'GLN':[(6,'CD','OE1','CG'),BBMOTIF],
        'TYR':[(9,'CG','CD1','CD2'),(4,'OH','HH','CZ'),BBMOTIF],
        'TRP':[(9,'CD2','CE2','CD1'),(4,'CD1','NE1','HE1'),BBMOTIF] #?
        'ARG':[(2,'CZ','NH1','NH2')],
    }


# define how to get the (y-axis,x-axis,symmetry)
MOTIFFRAME = [[], #none
              ['inv2',  'orth1', 2], #OOC
              ['inv2',  'orth1', 2], #guanidinium
              ['naive1','orth2', 3], #amine
              ['naive1','orth2', 1], #alcohol
              ['naive1','orth2', 1], #bb
              ['naive1','orth2', 1], #1-st amide
              ['naive1','orth2', 2], #ring N-H
              ['inv12', 'orth2', 2], #ring N:
              ['sum12', 'orth1', 2], #phe
] 

def xyz2axis(v1,v2,ftype):
    if ftype == 'orth1'
        e1 = v1/(np.sqrt(np.dot(v1,v1)+1e-6))
        return v2 - np.einsum('li, li -> l', e1, v2)*e1
    
    elif ftype == 'orth2':
        e1 = v1/(np.sqrt(np.dot(v1,v1)+1e-6))
        return v2 - np.einsum('li, li -> l', e1, v2)*e1
    
    elif ftype == 'inv2':
        return -u2/np.sqrt(np.dot(u2,u2))

    elif ftype == 'sum12':
        v = v1+v2
        return v/np.sqrt(np.dot(v,v))

    elif ftype == 'inv12':
        v = v1+v2
        return -v/np.sqrt(np.dot(v,v))
    return False

# calc wrt origin
def axis2xyz(q,mtype):
    xyz = np.zeros((3,3))
    if mtype == 1:
        y = q[:3]
        t = q[3]
        x = np.array(q[])
        xyz[2] = -y
    return

class MotifClass:
    def __init__(self,xyz,mtype):
        self.q = None
        self.T = None
        self.pdbid = ''
        self.chainres = ''
        self.xyz = xyz
        self.mtype = mtype

    # stolen from Justas's code :)
    def xyz2frameJ(self):
        v1 = self.xyz[0] - self.xyz[1]
        v2 = self.xyz[0] - self.xyz[2]
        e1 = v1/(np.sqrt(np.dot(v1,v1)) + 1e-6)
        u2 = v2 - np.einsum('li, li -> l', e1, v2)*e1
             
        e2 = u2/(np.sqrt(np.dot(u2,u2)) + 1e-6)
        e3 = np.cross(e1, e2)
    
        R = np.cat([e1, e2, e3], axis=-1)
        
        self.q = R2q3(R)
        self.R = R
        self.T = self.xyz[0]

    def checkJ(self,outf):
        Ri = np.transpose(self.R)
        xyz_t = np.matmul(Ri,self.xyz-self.T)
        for i,xyz in enumerate(xyz_t):
            print(i,xyz)
            
    def xyz2frame(self):
        # first get
        v1 = self.xyz[0] - self.xyz[1]
        v2 = self.xyz[0] - self.xyz[2]

        framefunc1,framefunc2,symmetry = MOTIFFRAME[self.mtype]
        
        x = xyz2axis(v1,v2,framefunc1)
        y = xyz2axis(v1,v2,framefunc2)
        z = np.cross(x, y)

        # store as y-axis & rotation
        # angle around y-axis
        #xdot = 1 - x[0] # == 1 - np.dot(x,xuniv)
        self.T = self.xyz[0]
        self.q = np.array([y[0],y[1],y[2],x[0]])

    def report(self):
        print(self.aa, self.pdbid, self.chainres, self.T, self.q[:3])

    def frame2xyz(self,mtype):
        framefunc1,framefunc2,symmetry = MOTIFFRAME[mtype]
        
        v1 = axis2xyz(self.q,framefunc1)
        v2 = axis2xyz(self.q,framefunc2)

def R2q3( R ):
    Q = np.zeros(4)
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S  = np.sqrt( 1.0 + R[0,0] - R[1,1] - R[2,2] ) * 2
        Q[0] = (0.25 * S)
        Q[1] = (R[0,1] + R[1,0]) / S
        Q[2] = (R[2,0] + R[0,2] ) / S
        Q[3] = (R[2,1] - R[1,2] ) / S
    elif R[1,1] > R[2,2]:
        S  = np.sqrt( 1.0 + R[1,1] - R[0,0] - R[2,2] ) * 2
        Q[0] = (R[1,0] + R[0,1] ) / S
        Q[1] = 0.25 * S
        Q[2] = (R[2,1] + R[1,2] ) / S
        Q[3] = (R[0,2] - R[2,0] ) / S
    else:
        S  = np.sqrt( 1.0 + R[2,2] - R[0,0] - R[1,1] ) * 2
        Q[0] = (R[0,2] + R[2,0] ) / S
        Q[1] = (R[2,1] + R[1,2] ) / S
        Q[2] = 0.25 * S
        Q[3] = (R[1,0] - R[0,1]) / S

    #q3: squeezed quaternion as unitvector*theta
    theta = 2.0*np.acos(Q[3])
    e = Q[:3]/np.sqrt(np.dot(Q[:3],Q[:3])+1.0e-6)
    q3 = theta*e
    
    return q3

def pdb2motifs(pdb):
    motifs = []
    xyzs = {}
    aas = {}
    
    # first read xyz,chainres
    for l in open(pdb):
        if not l.startswith('ATOM'): continue
        aa = l[16:20].strip()
        chainres = l[21]+'.'.+l[22:26].strip()
        if chainres not in aas:
            aas[chainres] = aa
            xyzs[chainres] = {}

        aname = l[12:16].strip()
        xyzs[chainres][aname] = [float(l[30:38]),float(l[38:46]),float(l[46:54])]

    # get motifs
    for chainres in xyzs:
        if aas[chainres] not in BASES: continue
        xyz = xyzs[chainres]
        
        for bases in POSSIBLEMOTIFS[aa]:
            for mtype,a1,a2,a3 in bases:
                if a1 not in xyz or a2 not in xyz or a3 not in xyz: continue
                
                xyz_base = np.array([xyz[a1],xyz[a2],xyz[a3]])
                motif = MotifClass(xyz_base,mtype)
                motif.aa = aa
                motif.mtype = mtype
                motif.pdbid = pdb.split('/')[-1][:4]
                motif.chainres = chainres
                motif.xyz2frame()
                
                motifs.append(motif)
                motif.report()
                
    return motifs

if __name__ == "__main__":
    pdb2motifs(sys.argv[1])
