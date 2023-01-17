import os,sys

f = sys.argv[1]

#Pcut = 0.3 #consider only those P higher than this
Pb = 0.2

ks = [1,2,3,4,5,6,7,8,9,11,12,13]
Ps_by_k = [[] for k in ks]
motifs = ['ASP','ARG','LYS','ALC','BB','AMD','NHS','NTR','PHE','PHO','PH2', 'MET']
Pcut =   [  Pb,  0.5,   0.5,  0.4,   Pb,  Pb,   Pb,   Pb,  0.15,  0.15, 0.15, 0.15] #"reference correction"

for l in open(f):
    words = l[:-1].split()
    if not l.startswith('C'): continue
    if len(words) < 15: continue
    #mtype = int(words[8])
    #prob = float(words[9])

    xyz = [float(word) for word in words[2:5]]
    Ps = [float(word) for word in l[:-1].split(':')[-1].split()[1:]]

    shift = 0
    if len(Ps) == 13:
        shift = -1

    for i,k in enumerate(ks):
        if Ps[k+shift] > Pcut[i]: Ps_by_k[i].append((Ps[k+shift],xyz))

form = "HETATM %4d  %3s UNK A %3d    %8.3f%8.3f%8.3f  1.00 %5.2f\n"
for i,k in enumerate(ks):
    outf = f.split('.')[0]+'.'+motifs[i]+'.pdb'
    out = open(outf,'w')
    for iatm,(P,xyz) in enumerate(Ps_by_k[i]):
        out.write(form%(i+1,motifs[i],i+1,xyz[0],xyz[1],xyz[2],P))
    out.close()
