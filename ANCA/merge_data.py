import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d1',
                    '--data1',
                    type=str,
                    help='Data 1')

parser.add_argument('-d2',
                    '--data2',
                    type=Path,
                    help='S Data')

parser.add_argument('-o',
                    '--output',
                    type=Path,
                    help='Output path for Images',
                    )

args = parser.parse_args()

data1 = np.load(args.data1)
data2 = np.load(args.data2)
print(data1.shape)
print(data2.shape)
datamerge = np.hstack([data1, data2])
np.save(f'{args.output}/merged.npy', datamerge)
