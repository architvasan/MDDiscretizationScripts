import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt


def show_scatter(data: np.ndarray, color: np.ndarray, output: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ff = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)
    plt.colorbar(ff)
    plt.savefig(output)
    plt.close()

dihedrals = np.load('run-4/selection_runs/selection-0/dihedrals.npy')
print(dihedrals.shape)

for i in range(dihedrals.shape[1]):
    dihedral_test = dihedrals[:,i,0]
    print(dihedral_test)

if False:
    print("\n")
    print(dihedrals.shape[1])
    for i in range(dihedrals.shape[1]):
        dihedral_test = dihedrals[:][1][i]
        print(dihedral_test)
    
    print("\n")
    for i in range(dihedrals.shape[2]):
        dihedral_test = dihedrals[:][2][i]
        print(dihedral_test)
    
    print("\n")
    for i in range(dihedrals.shape[2]):
        dihedral_test = dihedrals[:][3][i]
        print(dihedral_test)


plt.plot(dihedral_test[::100])
plt.savefig('cos_dihedral_test.png')
plt.close()

S = np.load('run-4/selection_runs/selection-0/S.npy')
plt.semilogy(S, "ro-")
plt.savefig('S.png')
plt.close()

e_rmsd = np.loadtxt('output_data/dtrajs.dat')
ZPrj4 = np.load('run-4/selection_runs/selection-0/sd4_projection.npy')
print(ZPrj4.shape)
print(len(e_rmsd))

show_scatter(ZPrj4[:, :3], e_rmsd, 'Images/ZPrj4.png')

metastates = np.zeros(len(e_rmsd))
for i in range(4):
    meta = np.loadtxt(f'output_data/metastate{i}_frames.dat')
    for p in meta:
        metastates[int(p)] = i


show_scatter(ZPrj4[:, :3], metastates, 'Images/ZPrj4_meta.png')








