import numpy as np
from deeptime.markov import TransitionCountEstimator
import deeptime.markov as markov
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()

parser.add_argument('-t',
                    '--trajdata',
                    type=Path,
                    help='Input trajectory data')

parser.add_argument('-l',
                    '--lagtime',
                    type=int,
                    help='lagtime')

parser.add_argument('-o',
                    '--output',
                    type=Path,
                    help='output directory')

args = parser.parse_args()




count_estimator = TransitionCountEstimator(
    lagtime=args.lagtime,
    count_mode='sliding'
    )

msm_estimator = markov.msm.MaximumLikelihoodMSM(
                reversible=True,
                stationary_distribution_constraint=None
                )

trajectory = np.loadtxt(args.trajdata).astype(int)
msm = msm_estimator.fit(trajectory, lagtime=args.lagtime).fetch_model()

np.savetxt(f'{args.output}/T_Mat.dat', msm.transition_matrix)
np.savetxt(f'{args.output}/stationary.dat', msm.stationary_distribution)
print(f"Number of states: {msm.n_states}")

import networkx as nx
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

threshold = 1e-4
title = f"Transition matrix with connectivity threshold {threshold:.0e}"
G = nx.DiGraph()
ax.set_title(title)
for i in range(msm.n_states):
    G.add_node(i, title=f"{i+1}")
for i in range(msm.n_states):
    for j in range(msm.n_states):
        if msm.transition_matrix[i, j] > threshold:
            G.add_edge(i, j, title=f"{msm.transition_matrix[i, j]:.3e}")

edge_labels = nx.get_edge_attributes(G, 'title')
pos = nx.fruchterman_reingold_layout(G)
nx.draw_networkx_nodes(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, 'title'));
nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>',
                       connectionstyle='arc3, rad=0.3')
plt.savefig(f'{args.output}/network_view_highthresh.png')
plt.close()

pcca = msm.pcca(n_metastable_sets=6)
print(f"Memberships: {pcca.memberships.shape}")
print(pcca.coarse_grained_stationary_probability)
np.savetxt(f'{args.output}/pcca_probs.dat', pcca.coarse_grained_stationary_probability)
print("Metastable distributions shape:", pcca.metastable_distributions.shape)
print(pcca.coarse_grained_transition_matrix)

np.savetxt(f'{args.output}/pcca_assignments.dat', np.array([int(p_as) for p_as in pcca.assignments]).T)
np.savetxt(f'{args.output}/pcca_coarse_grained_TMat.dat', pcca.coarse_grained_transition_matrix)


threshold = 1e-9
title = f"PCCA Transition matrix with connectivity threshold {threshold:.0e}"
G = nx.DiGraph()
ax.set_title(title)
for i in range(4):
    G.add_node(i, title=f"{i+1}")
for i in range(4):
    for j in range(4):
        print(pcca.coarse_grained_transition_matrix[i, j])
        if pcca.coarse_grained_transition_matrix[i, j] > threshold:
            G.add_edge(i, j, title=f"{pcca.coarse_grained_transition_matrix[i, j]:.3e}")

edge_labels = nx.get_edge_attributes(G, 'title')
pos = nx.fruchterman_reingold_layout(G)
nx.draw_networkx_nodes(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, 'title'));
nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>',
                       connectionstyle='arc3, rad=0.3')

plt.savefig(f'{args.output}/pcca_network_view.png')
plt.close()
