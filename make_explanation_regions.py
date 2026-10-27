"""Render regions for the explanation objects reported in Table 4."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from alibi.explainers import AnchorTabular

OUT = Path(__file__).resolve().parent
DPG_PATH = OUT / '.vendor' / 'dpg'
sys.path.insert(0, str(DPG_PATH))
from dpg import DPGExplainer

SEED = 7
X, _ = make_moons(n_samples=240, noise=.08, random_state=SEED)
labels = SpectralClustering(
    n_clusters=2, affinity='nearest_neighbors', n_neighbors=12,
    random_state=SEED).fit_predict(X)
idx = np.arange(len(X))
tr, te = train_test_split(idx, test_size=.30, random_state=SEED, stratify=labels)
Xtr, ytr = X[tr], labels[tr]

rf = RandomForestClassifier(n_estimators=30, random_state=SEED).fit(Xtr, ytr)
tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr, ytr)

grid_x, grid_y = np.meshgrid(
    np.linspace(X[:, 0].min() - .35, X[:, 0].max() + .35, 55),
    np.linspace(X[:, 1].min() - .35, X[:, 1].max() + .35, 38))
grid = np.c_[grid_x.ravel(), grid_y.ravel()]

def medoids():
    out = []
    for cluster in np.unique(labels):
        points = Xtr[ytr == cluster]
        distances = cdist(points, points)
        out.append(points[np.argmin(distances.sum(axis=1))])
    return np.asarray(out)

medoid_points = medoids()
medoid_pred = cdist(grid, medoid_points).argmin(axis=1)

# DPG retained-path adapter: only votes whose complete execution path survives
# the p=.020 graph are used; unresolved points are shown as abstention.
dpg_config = {'dpg': {'default': {'perc_var': .020, 'decimal_threshold': 4,
                                  'n_jobs': 1},
                       'graph_construction': {'mode': 'execution_trace',
                                              'context_order': 1},
                       'visualization': {}}}
dpg = DPGExplainer(rf, ['x_1', 'x_2'], ['0', '1'], dpg_config=dpg_config)
dpg.fit(Xtr)

def dpg_predict(points):
    prediction = np.zeros(len(points), dtype=float)
    covered = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        local = dpg.explain_local(point, sample_id=i)
        votes = [int(path.labels[-1].replace('Class ', ''))
                 for path in local.tree_paths
                 if path.labels and all(path.edge_exists)]
        if votes:
            counts = np.bincount(votes, minlength=2)
            if counts.max() > len(votes) / 2:
                prediction[i] = np.argmax(counts)
                covered[i] = True
    return prediction, covered

dpg_pred, dpg_covered = dpg_predict(grid)

# Anchors: collect local rules from held-out points and apply their predicates
# to the grid. Identical rules with a common consequent support a region;
# uncovered or conflicting regions are abstention.
anchor = AnchorTabular(lambda data: rf.predict(data), ['x_1', 'x_2'], seed=SEED)
anchor.fit(Xtr)
rules = []
for point in X[te][:24]:
    explanation = anchor.explain(point, threshold=.90, max_anchor_size=2,
                                  batch_size=100)
    rule = tuple(sorted(explanation.anchor))
    rules.append((rule, int(explanation.raw['prediction'][0])))

def rule_holds(rule, points):
    mask = np.ones(len(points), dtype=bool)
    for condition in rule:
        if '<=' in condition:
            feature, threshold = condition.split(' <= ')
            mask &= points[:, int(feature[-1]) - 1] <= float(threshold)
        elif '>' in condition:
            feature, threshold = condition.split(' > ')
            mask &= points[:, int(feature[-1]) - 1] > float(threshold)
    return mask

anchor_pred = np.zeros(len(grid), dtype=float)
anchor_covered = np.zeros(len(grid), dtype=bool)
for point_idx in range(len(grid)):
    matches = [(rule, cls) for rule, cls in rules
               if rule_holds(rule, grid[point_idx:point_idx + 1])[0]]
    classes = {cls for _, cls in matches}
    if len(classes) == 1:
        anchor_pred[point_idx] = next(iter(classes))
        anchor_covered[point_idx] = True

predictions = [
    (medoid_pred, np.ones(len(grid), dtype=bool), '(a) Medoid'),
    (tree.predict(grid), np.ones(len(grid), dtype=bool), '(b) Tree, depth 3'),
    (rf.predict(grid), np.ones(len(grid), dtype=bool), '(c) Random Forest'),
    (dpg_pred, dpg_covered, '(d) DPG, $p=.020$'),
    (anchor_pred, anchor_covered, '(e) Anchors'),
]

fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.0), sharex=True, sharey=True)
for ax, (prediction, covered, title) in zip(axes, predictions):
    image = prediction.reshape(grid_x.shape).astype(float)
    image[~covered.reshape(grid_x.shape)] = np.nan
    ax.contourf(grid_x, grid_y, image, levels=[-.5, .5, 1.5],
                colors=['white', '0.72'], alpha=.8)
    ax.contour(grid_x, grid_y, prediction.reshape(grid_x.shape), levels=[.5],
               colors='black', linewidths=.8)
    # Abstention (no decision) is hatched so it is distinct from cluster 0 (white).
    print(title, 'abstention fraction of plotted grid:', round(float((~covered).mean()), 3))
    if (~covered).any():
        ax.contourf(grid_x, grid_y, (~covered).reshape(grid_x.shape).astype(float),
                    levels=[.5, 1.5], colors=['white'], hatches=['////'], alpha=0)
    for cluster, marker in ((0, 'o'), (1, 's')):
        mask = labels == cluster
        ax.scatter(X[mask, 0], X[mask, 1], marker=marker, s=6,
                   facecolors='white', edgecolors='black', linewidths=.4)
    if title.startswith('(a)'):
        ax.scatter(medoid_points[:, 0], medoid_points[:, 1], marker='X',
                   s=110, facecolors='black', edgecolors='white', linewidths=.9, zorder=5)
    ax.set_title(title, fontsize=8.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].set_ylabel('$x_2$', fontsize=9)
fig.tight_layout(w_pad=.6)
fig.savefig(OUT / 'sanity_surrogate_disagreement.pdf', bbox_inches='tight')
fig.savefig(OUT / 'sanity_surrogate_disagreement.png', dpi=220, bbox_inches='tight')
plt.close(fig)
