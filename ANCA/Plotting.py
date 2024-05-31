import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def show_scatter(data: np.ndarray, color: np.ndarray, output: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ff = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)
    plt.colorbar(ff)
    plt.savefig(output)
    plt.close()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d',
                    '--dihdata',
                    type=str,
                    help='Dihedral Data')

parser.add_argument('-S',
                    '--Sdata',
                    type=Path,
                    help='S Data')

parser.add_argument('-e',
                    '--ermsd',
                    type=Path,
                    help='e-rmsd data',
                    )

parser.add_argument('-Z',
                    '--ZPrj4',
                    type=Path,
                    help='ZProj sd4 data',
                    )

parser.add_argument('-o',
                    '--output',
                    type=Path,
                    help='Output path for Images',
                    )

args = parser.parse_args()


dihedrals = np.load(args.dihdata)

for i in range(dihedrals.shape[1]):
    dihedral_test = dihedrals[:,i,0]

plt.plot(dihedral_test[::100])
plt.savefig(f'{args.output}/cos_dihedral_test.png')
plt.close()

S = np.load(args.Sdata)
plt.semilogy(S, "ro-")
plt.savefig(f'{args.output}/S.png')
plt.close()

e_rmsd = np.load(args.ermsd)
ZPrj4 = np.load(args.ZPrj4)

show_scatter(ZPrj4[:, :3], e_rmsd[-1], f'{args.output}/ZPrj4.png')

#z = np.load('run-4/selection_runs/selection-0/z.npy')
#show_scatter(z, e_rmsd[-1], 'z_ermsds.png')
