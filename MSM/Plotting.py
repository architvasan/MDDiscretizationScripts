import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()

parser.add_argument('-d',
                    '--dihdata',
                    type=Path,
                    help='Input trajectory data')

parser.add_argument('-S',
                    '--Sdata',
                    type=Path,
                    help='S data')

parser.add_argument('-t',
                    '--traj',
                    type=Path,
                    help='dtrajs data')

parser.add_argument('-z',
                    '--zdata',
                    type=Path,
                    help='Z projection data')

parser.add_argument('-m',
                    '--metadata',
                    type=Path,
                    help='location of metastate frames')

parser.add_argument('-n',
                    '--nummeta',
                    type=int,
                    help='number of metastates')

parser.add_argument('-o',
                    '--output',
                    type=Path,
                    help='output directory for metadata')

args = parser.parse_args()

try:
    os.mkdir(f'{args.output}')
except:
    print("directory already exists")


def show_scatter(data: np.ndarray, color: np.ndarray, output: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ff = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)
    plt.colorbar(ff)
    plt.savefig(output, dpi=300)
    plt.close()

dihedrals = np.load(args.dihdata)

for i in range(dihedrals.shape[1]):
    dihedral_test = dihedrals[:,i,0]
    print(dihedral_test)

plt.plot(dihedral_test[::100])
plt.savefig(f'{args.output}/cos_dihedral_test.png', dpi=300)
plt.close()

S = np.load(args.Sdata)
plt.semilogy(S, "ro-")
plt.savefig(f'{args.output}/S.png', dpi=300)
plt.close()

e_rmsd = np.loadtxt(args.traj)
ZPrj4 = np.load(args.zdata)
print(ZPrj4.shape)
print(len(e_rmsd))

show_scatter(ZPrj4[:, :3], e_rmsd, f'{args.output}/ZPrj4.png')

metastates = np.zeros(len(e_rmsd))
for i in range(args.nummeta):
    meta = np.loadtxt(f'{args.metadata}/metastate{i}_frames.dat')
    for p in meta:
        metastates[int(p)] = i


show_scatter(ZPrj4[:, :3], metastates, f'{args.output}/ZPrj4_meta.png')


