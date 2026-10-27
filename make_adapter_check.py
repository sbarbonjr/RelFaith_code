"""Adapter sensitivity: one Anchors explanation set, three declared adapters."""
from pathlib import Path
import numpy as np
import pandas as pd
from alibi.explainers import AnchorTabular
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent
SEED = 7
X, _ = make_moons(n_samples=240, noise=.08, random_state=SEED)
labels = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=12,
                            random_state=SEED).fit_predict(X)
tr, te = train_test_split(np.arange(len(X)), test_size=.30, random_state=SEED, stratify=labels)
Xt, lt = X[te], labels[te]
rf = RandomForestClassifier(n_estimators=30, random_state=SEED).fit(X[tr], labels[tr])
dist = cdist(Xt, Xt)
np.fill_diagonal(dist, np.inf)


def stratified(k):
    sel = set()
    for i in range(len(lt)):
        for same in (True, False):
            cand = np.flatnonzero((lt == lt[i]) == same)
            cand = cand[cand != i]
            for j in cand[np.argsort(dist[i, cand])[:k]]:
                sel.add((min(i, int(j)), max(i, int(j))))
    return sorted(sel)


anchor = AnchorTabular(lambda d: rf.predict(d), ['x_1', 'x_2'], seed=SEED)
anchor.fit(X[tr])
pred, keys, covered = [], [], []
for point in Xt:
    e = anchor.explain(point, threshold=.90, max_anchor_size=2, batch_size=100)
    pred.append(int(e.raw['prediction'][0]))
    keys.append(tuple(sorted(e.anchor)))
    covered.append(len(e.anchor) > 0)
pred, covered = np.array(pred), np.array(covered)


def adapter(name, i, j):
    """Return 'same', 'contrast' or 'abstain' for the pair (i, j)."""
    if not (covered[i] and covered[j]):
        return 'abstain'
    if name == 'consequent':      # every covered pair is decided by the consequents
        return 'same' if pred[i] == pred[j] else 'contrast'
    if pred[i] != pred[j]:        # strict: different consequents -> contrast
        return 'contrast'
    return 'same' if keys[i] == keys[j] else 'abstain'


def scores(name, pairs):
    same_t = sum(lt[i] == lt[j] for i, j in pairs)
    con_t = len(pairs) - same_t
    sc = cc = ab = 0
    for i, j in pairs:
        r = adapter(name, i, j)
        ab += r == 'abstain'
        sc += r == 'same' and lt[i] == lt[j]
        cc += r == 'contrast' and lt[i] != lt[j]
    S, D = sc / same_t, cc / con_t
    return dict(S=S, D=D, F_W=2 * S * D / (S + D) if S + D else np.nan, A=ab / len(pairs))


all_pairs = [(i, j) for i in range(len(lt)) for j in range(i + 1, len(lt))]
rows = []
for aname in ('strict', 'consequent'):
    for cname, pairs in (('all pairs', all_pairs), ('W_3', stratified(3)), ('W_15', stratified(15))):
        rows.append(dict(adapter=aname, context=cname, n_pairs=len(pairs), **scores(aname, pairs)))
df = pd.DataFrame(rows)
df.to_csv(OUT / 'position_adapter_sensitivity.csv', index=False)
print(df.round(3).to_string(index=False))
print('uncovered test points:', int((~covered).sum()), 'distinct predicate sets:', len(set(keys)))
