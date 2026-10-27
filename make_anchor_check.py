"""Held-out RelFaith check for Anchors applied to the RF surrogate."""
import numpy as np
import pandas as pd
from alibi.explainers import AnchorTabular
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

OUT = Path(__file__).resolve().parent
SEED = 7
X, _ = make_moons(n_samples=240, noise=.08, random_state=SEED)
labels = SpectralClustering(
    n_clusters=2, affinity='nearest_neighbors', n_neighbors=12,
    random_state=SEED).fit_predict(X)
idx = np.arange(len(X))
tr, te = train_test_split(idx, test_size=.30, random_state=SEED, stratify=labels)
Xt, lt = X[te], labels[te]
rf = RandomForestClassifier(n_estimators=30, random_state=SEED).fit(X[tr], labels[tr])

distances = cdist(Xt, Xt)
np.fill_diagonal(distances, np.inf)

def make_context(k):
    selected = set()
    for i in range(len(lt)):
        for same_relation in (True, False):
            candidates = np.flatnonzero((lt == lt[i]) == same_relation)
            candidates = candidates[candidates != i]
            for j in candidates[np.argsort(distances[i, candidates])[:k]]:
                selected.add((min(i, int(j)), max(i, int(j))))
    return sorted(selected)

def relation_scores(prediction, anchor_keys, pairs):
    same_total = sum(lt[i] == lt[j] for i, j in pairs)
    contrast_total = len(pairs) - same_total
    same_correct = contrast_correct = abstained = 0
    for i, j in pairs:
        if anchor_keys[i] is None or anchor_keys[j] is None:
            abstained += 1
        elif prediction[i] != prediction[j]:
            if lt[i] != lt[j]:
                contrast_correct += 1
        elif anchor_keys[i] == anchor_keys[j]:
            if lt[i] == lt[j]:
                same_correct += 1
        else:
            abstained += 1
    S = same_correct / same_total
    D = contrast_correct / contrast_total
    return {'S': S, 'D': D, 'F_W': 2 * S * D / (S + D),
            'A': abstained / len(pairs), 'n_pairs': len(pairs)}

anchor = AnchorTabular(lambda data: rf.predict(data), ['x_1', 'x_2'], seed=SEED)
anchor.fit(X[tr])
predictions, keys = [], []
for point in Xt:
    explanation = anchor.explain(point, threshold=.90, max_anchor_size=2,
                                  batch_size=100)
    predictions.append(int(explanation.raw['prediction'][0]))
    # Identical predicate sets are treated as the same local rule.  Different
    # rules with the same consequent remain unresolved and therefore abstain.
    keys.append(tuple(sorted(explanation.anchor)))

rows = []
for context, pairs in {'all-pairs': [(i, j) for i in range(len(lt))
                                     for j in range(i + 1, len(lt))],
                       'stratified-15-NN': make_context(15)}.items():
    rows.append({'method': 'Anchors', 'threshold': .90,
                 'context': context, **relation_scores(predictions, keys, pairs)})
frame = pd.DataFrame(rows)
frame.to_csv(OUT / 'sanity_anchor_scores.csv', index=False)
print(frame.to_string(index=False))
print('unique_anchor_rules=', len(set(keys)))
