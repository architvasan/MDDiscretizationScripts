import numpy as np
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()

parser.add_argument('-t',
                    '--trajdata',
                    type=Path,
                    help='Input trajectory data')

parser.add_argument('-p',
                    '--pcca_assign',
                    type=Path,
                    help='PCCA Assignments')

parser.add_argument('-o',
                    '--output',
                    type=Path,
                    help='output directory for metadata')

args = parser.parse_args()

try:
    os.mkdir(f'{args.output}')
except:
    print("directory already exists")

dtrajs = np.loadtxt(f'{args.trajdata}')
dtrajs = np.array([int(d) for d in dtrajs])
pcca_assign = np.loadtxt(args.pcca_assign)
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
    np.savetxt(f'{args.output}/metastate{pd}_frames.dat', pcca_frames[pd], fmt='%i')
