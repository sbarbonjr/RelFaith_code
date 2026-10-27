"""Figure 2: example datasets with spectral labels (top) and extension agreement (bottom)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_circles, make_moons

OUT = Path(__file__).resolve().parent
SEED, N = 0, 240
GEN = {
    'two moons': lambda n, s: make_moons(n, noise=.08, random_state=s)[0],
    'two circles': lambda n, s: make_circles(n, noise=.05, factor=.45, random_state=s)[0],
}
probe = pd.read_csv(OUT / 'position_probe_agreement.csv')

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(3.4, 3.15))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=.45, wspace=.25)
for col, (name, gen) in enumerate(GEN.items()):
    X = gen(N, SEED)
    y = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=12,
                           random_state=SEED).fit_predict(X)
    ax = fig.add_subplot(gs[0, col])
    for c, mk, fc in ((0, 'o', 'white'), (1, 's', '0.35')):
        ax.scatter(X[y == c, 0], X[y == c, 1], marker=mk, s=9, facecolors=fc,
                   edgecolors='black', linewidths=.5)
    ax.set_title(f'({"ab"[col]}) {name}', fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[1, :])
data, cols, labs = [], [], []
for d in GEN:
    for pn, col in (('uniform box', '0.75'), ('data distribution', '0.35')):
        data.append(probe[(probe.dataset == d) & (probe.probes == pn)].agree.values)
        cols.append(col)
        labs.append(f'{d.replace("two ", "")}\n{"uniform" if pn.startswith("u") else "in-dist."}')
bp = ax.boxplot(data, patch_artist=True, widths=.6, medianprops=dict(color='k'))
for p, c in zip(bp['boxes'], cols):
    p.set_facecolor(c)
ax.set_xticks(range(1, 5), labs)
ax.set_ylabel('RF vs 1-NN agreement')
ax.set_ylim(.75, 1.005)
ax.set_title('(c) extension agreement over 20 seeds', fontsize=8)
fig.savefig(OUT / 'position_probe_agreement.pdf', bbox_inches='tight')
fig.savefig(OUT / 'position_probe_agreement.png', dpi=220, bbox_inches='tight')
