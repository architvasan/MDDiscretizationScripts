import numpy as np

dtrajs = np.loadtxt('output_data/dtrajs.dat')
dtrajs = np.array([int(d) for d in dtrajs])
pcca_assign = np.loadtxt('output_data/pcca_assignments.dat')
pcca_assign = np.array([int(pd) for pd in pcca_assign])
pcca_clusters = {}
pcca_frames = {}

for pd in range(int(np.max(pcca_assign))+1):
    print(pd)
    print(pcca_assign)
    print(np.where(pcca_assign==pd)[0])
    pcca_clusters[pd]=np.where(pcca_assign==pd)[0]
    pcca_frames[pd]=[]
    for c in pcca_clusters[pd]:
        pcca_frames[pd].extend(list(np.where(dtrajs==int(c))[0]))
    np.savetxt(f'metastate{pd}_frames.dat', pcca_frames[pd], fmt='%i') 
