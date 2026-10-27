"""Held-out RelFaith checks for a Random Forest and its DPG explanation.

Run with ``python3 make_dpg_check.py``.  The script uses the project-local DPG
installation in .vendor/dpg and writes sanity_dpg_scores.csv.
"""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT / '.vendor' / 'dpg'))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dpg import DPGExplainer

SEED = 7
X, _ = make_moons(n_samples=240, noise=.08, random_state=SEED)
labels = SpectralClustering(
    n_clusters=2, affinity='nearest_neighbors', n_neighbors=12,
    random_state=SEED).fit_predict(X)
idx = np.arange(len(X))
tr, te = train_test_split(idx, test_size=.30, random_state=SEED, stratify=labels)
Xt, lt = X[te], labels[te]

distances = cdist(Xt, Xt)
np.fill_diagonal(distances, np.inf)


def make_context(k):
    selected = set()
    for i in range(len(lt)):
        for same_relation in (True, False):
            candidates = np.flatnonzero((lt == lt[i]) == same_relation)
            candidates = candidates[candidates != i]
            nearest = candidates[np.argsort(distances[i, candidates])[:k]]
            for j in nearest:
                selected.add((min(i, int(j)), max(i, int(j))))
    return sorted(selected)


def relation_scores(prediction, pairs, covered):
    same_total = sum(lt[i] == lt[j] for i, j in pairs)
    contrast_total = len(pairs) - same_total
    same_correct = contrast_correct = abstained = 0
    for i, j in pairs:
        if not (covered[i] and covered[j]):
            abstained += 1
        elif lt[i] == lt[j] and prediction[i] == prediction[j]:
            same_correct += 1
        elif lt[i] != lt[j] and prediction[i] != prediction[j]:
            contrast_correct += 1
    S = same_correct / same_total
    D = contrast_correct / contrast_total
    return {'S': S, 'D': D, 'F_W': 2 * S * D / (S + D),
            'A': abstained / len(pairs), 'n_pairs': len(pairs)}


contexts = {
    'all-pairs': [(i, j) for i in range(len(lt)) for j in range(i + 1, len(lt))],
    'stratified-15-NN': make_context(15),
}


def dpg_adapter(model, perc_var):
    config = {
        'dpg': {
            'default': {'perc_var': perc_var, 'decimal_threshold': 4, 'n_jobs': 1},
            'graph_construction': {'mode': 'execution_trace', 'context_order': 1},
            'visualization': {},
        }
    }
    explainer = DPGExplainer(
        model, feature_names=['x_1', 'x_2'], target_names=['0', '1'],
        dpg_config=config)
    # ``explain_global`` additionally requests community metrics, which can be
    # undefined for a heavily pruned graph.  RelFaith needs the fitted DPG and
    # its retained execution paths, not those optional community statistics.
    explainer.fit(X[tr])
    prediction = np.zeros(len(Xt), dtype=int)
    covered = np.zeros(len(Xt), dtype=bool)
    retained_votes = []
    for point_idx, point in enumerate(Xt):
        local = explainer.explain_local(point, sample_id=point_idx)
        votes = []
        for path in local.tree_paths:
            # A vote is available only when the complete predicate path used by
            # this point survives the DPG frequency filter.
            if path.labels and all(path.edge_exists):
                votes.append(int(path.labels[-1].replace('Class ', '')))
        retained_votes.append(len(votes))
        if votes:
            counts = np.bincount(votes, minlength=2)
            if counts.max() > len(votes) / 2:
                prediction[point_idx] = int(np.argmax(counts))
                covered[point_idx] = True
    return prediction, covered, explainer._graph.number_of_nodes(), \
        explainer._graph.number_of_edges(), float(np.mean(retained_votes))


rf = RandomForestClassifier(n_estimators=30, random_state=SEED).fit(X[tr], labels[tr])
rows = []
for perc_var in (0.0, .001, .005, .01, .02):
    prediction, covered, n_nodes, n_edges, mean_votes = dpg_adapter(rf, perc_var)
    for context_name, pairs in contexts.items():
        rows.append({
            'method': 'DPG of RF', 'perc_var': perc_var,
            'n_nodes': n_nodes, 'n_edges': n_edges,
            'mean_retained_tree_votes': mean_votes, 'context': context_name,
            **relation_scores(prediction, pairs, covered),
        })

rf_prediction = rf.predict(Xt)
for context_name, pairs in contexts.items():
    rows.append({
        'method': 'RF surrogate', 'perc_var': np.nan,
        'n_nodes': np.nan, 'n_edges': np.nan,
        'mean_retained_tree_votes': 30.0, 'context': context_name,
        **relation_scores(rf_prediction, pairs, np.ones(len(Xt), dtype=bool)),
    })

frame = pd.DataFrame(rows)
frame.to_csv(OUT / 'sanity_dpg_scores.csv', index=False)
print(frame.to_string(index=False))
