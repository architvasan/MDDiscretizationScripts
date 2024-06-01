import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import mdshare  # for trajectory data
from tqdm import tqdm  # progress bar
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
torch.set_num_threads(12)
print(f"Using device {device}")
from deeptime.util.data import TrajectoryDataset, TrajectoriesDataset
from deeptime.util.validation import implied_timescales, ck_test
from deeptime.plots import plot_implied_timescales, plot_ck_test
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM
import deeptime.markov as markov
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    '--inputdata',
                    type=Path,
                    help='Input Data')

parser.add_argument('-c',
                    '--numclusts',
                    type=int,
                    help='numclusters')

parser.add_argument('-o',
                    '--output',
                    type=Path,
                    help='output directory')

args = parser.parse_args()

try:
    os.mkdir(args.output)

except:
    print("already have directory")

data = np.load(args.sd4proj)#"./run-4/selection_runs/selection-0/sd4_projection.npy")
data = [data.astype(np.float32)]
dataset = TrajectoriesDataset.from_numpy(1, data)
cluster = KMeans(args.numclusts, progress=tqdm).fit_fetch(data)#30
dtrajs = [cluster.transform(x) for x in data]
#print(dtrajs)
msm_estimator = markov.msm.MaximumLikelihoodMSM(
                reversible=True,
                stationary_distribution_constraint=None
                )

lagtimes = np.arange(1, 30, dtype=np.int32)
models = [msm_estimator.fit(dtrajs, lagtime=lag).fetch_model() for lag in tqdm(lagtimes)]

ax = plot_implied_timescales(implied_timescales(models))
ax.set_yscale('log')
ax.set_xlabel('lagtime')
ax.set_ylabel('timescale')
plt.savefig(f'{args.output}/Timescales.png', dpi=300)
plt.close()


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#print(data[0][:,0])
#print(dtrajs)
np.savetxt(f'{args.output}/dtrajs.dat', np.array(dtrajs).T, fmt='%i')
np.savetxt(f'{args.output}/data.dat', data[0])
#counts,ybins,xbins,image = np.histogram2d(projections[0][:,0], projections[0][:,1], bins=100, norm=LogNorm())
cb = ax2.hist2d(data[0][:,0], data[0][:,1], bins=20)
#plt.colorbar(cb, ax=ax2)
ax1.plot(dtrajs)
ax1.set_xlim(0,40000)
ax1.set_xlabel('t')
ax1.set_ylabel('state')
ax2.set_xlabel('SD4-1')
ax2.set_ylabel('SD4-2')
plt.savefig(f'{args.output}/SD4_anddiscr.png', dpi=300)
plt.close()

dtrajs = np.loadtxt(f'{args.output}/dtrajs.dat')
dtrajs = dtrajs.astype(np.int32)
bmsms = [BayesianMSM(lagtime=lag).fit_fetch(dtrajs) for lag in tqdm([10, 20, 25, 30, 40, 50])]#, 60, 70, 80, 90, 100])]
ck_test = bmsms[5].ck_test(bmsms, 6)
plot_ck_test(ck_test)
plt.savefig(f'{args.output}/ck_test.png', dpi=300)
plt.close()
