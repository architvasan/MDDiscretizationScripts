import numpy as np
import argparse
from pathlib import Path
import os
import MDAnalysis as mda

parser = argparse.ArgumentParser()

parser.add_argument('-i',
                    '--inpdata',
                    type=Path,
                    help='Input data into MSM (eg z data')


parser.add_argument('-m',
                    '--metadata',
                    type=Path,
                    help='Directory with metastable state frames')

parser.add_argument('-n',
                    '--nummeta',
                    type=int,
                    help='number of metastable states')

parser.add_argument('-o',
                    '--output',
                    type=Path,
                    help='output directory for metadata')

parser.add_argument('-P',
                    '--PDBfile',
                    type=Path,
                    help='PDB file for traject')

parser.add_argument('-D',
                    '--DCDfile',
                    type=Path,
                    help='DCD file for traject')

args = parser.parse_args()

def id_centroid(metaframes, input_data, output):
    data_metait = [input_data[int(f)] for f in metaframes]#input_data[metaframes]
    centroid_val = np.array(data_metait).mean(axis=0)
    dist_array = []
    for d in data_metait:
        dist_array.append((np.sum([dit**2 for dit in d])))
    
    centroid_ind = np.argmin(dist_array)#,axis=0)
    centroid_frame = metaframes[centroid_ind]
    return int(centroid_frame)#centroid

try:
    os.mkdir(args.output)
except:
    print(f"{args.output} already exists")

# Load the universe with your topology (e.g., a PDB file) and trajectory (DCD file)
u = mda.Universe(args.PDBfile, args.DCDfile)

input_data = np.load(args.inpdata)
for m in range(args.nummeta):
    input_frames = np.loadtxt(f'{args.metadata}/metastate{m}_frames.dat')
    centroid = id_centroid(input_frames, input_data, args.output)
    print(centroid)
    # Select the specific frame you want to extract
    
    # Move to the specific frame
    u.trajectory[centroid]
    
    # Write the frame to a PDB file
    with mda.Writer(f'{args.output}/meta{m}.pdb', multiframe=False) as W:
        W.write(u.atoms)
    
