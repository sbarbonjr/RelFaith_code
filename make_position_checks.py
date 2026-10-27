"""Multi-seed illustrations for the position paper.

P2: extension ambiguity (figure: make_probe_figure.py) under uniform vs data-distribution probes, and whether
    the ranking of explainers changes with the extension.
P3: relational-context sweep F_W(k) for boundary vs interior errors, with a
    size-matched random-pair control, and rank agreement between contexts.
Outputs: position_*.csv and position_probe_agreement.pdf, position_context_sensitivity.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_circles, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).resolve().parent
N, SEEDS, N_FLIPS = 240, range(20), 8
KS = (1, 2, 3, 5, 10, 15, 20)
GENERATORS = {
    'two moons': lambda n, s: make_moons(n, noise=.08, random_state=s)[0],
    'two circles': lambda n, s: make_circles(n, noise=.05, factor=.45, random_state=s)[0],
}


def pair_scores(pred, ref, I, J, covered=None):
    same_ref = ref[I] == ref[J]
    ok = np.ones(len(I), bool) if covered is None else covered[I] & covered[J]
    same_pred = pred[I] == pred[J]
    S = (same_ref & ok & same_pred).sum() / same_ref.sum()
    D = (~same_ref & ok & ~same_pred).sum() / (~same_ref).sum()
    F = 2 * S * D / (S + D) if S + D else np.nan
    return S, D, F


def stratified(ref, dist, k):
    sel = set()
    for i in range(len(ref)):
        for same in (True, False):
            cand = np.flatnonzero((ref == ref[i]) == same)
            cand = cand[cand != i]
            for j in cand[np.argsort(dist[i, cand])[:k]]:
                sel.add((min(i, int(j)), max(i, int(j))))
    a = np.array(sorted(sel))
    return a[:, 0], a[:, 1]


def knn_graph(dist, k):
    sel = set()
    for i in range(len(dist)):
        for j in np.argsort(dist[i])[:k]:
            sel.add((min(i, int(j)), max(i, int(j))))
    a = np.array(sorted(sel))
    return a[:, 0], a[:, 1]


def matched_random(ref, I, J, all_I, all_J, rng):
    """Random pairs with the same same/contrast counts as (I, J)."""
    n_same = (ref[I] == ref[J]).sum()
    n_con = len(I) - n_same
    same_pool = np.flatnonzero(ref[all_I] == ref[all_J])
    con_pool = np.flatnonzero(ref[all_I] != ref[all_J])
    pick = np.concatenate([rng.choice(same_pool, n_same, replace=False),
                           rng.choice(con_pool, n_con, replace=False)])
    return all_I[pick], all_J[pick]


def medoid_predict(Xtr, ytr, Xt):
    meds = []
    for c in np.unique(ytr):
        pts = Xtr[ytr == c]
        meds.append(pts[cdist(pts, pts).sum(1).argmin()])
    return np.unique(ytr)[cdist(Xt, np.array(meds)).argmin(1)]


probe_rows, rank_rows, sweep_rows, rankctx_rows = [], [], [], []
for dname, gen in GENERATORS.items():
    for seed in SEEDS:
        X = gen(N, seed)
        y = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=12,
                               random_state=seed).fit_predict(X)
        if len(np.unique(y)) < 2 or np.bincount(y).min() < 20:
            continue
        tr, te = train_test_split(np.arange(N), test_size=.3, random_state=seed, stratify=y)
        rf = RandomForestClassifier(100, random_state=seed).fit(X, y)
        nn1 = KNeighborsClassifier(1).fit(X, y)
        lo, hi = X.min(0) - .4, X.max(0) + .4
        U = np.random.default_rng(seed).uniform(lo, hi, (20000, 2))
        Xd = gen(5000, 1000 + seed)
        probes = {'uniform box': U, 'data distribution': Xd}
        for pname, P in probes.items():
            probe_rows.append(dict(dataset=dname, seed=seed, probes=pname,
                                   agree=(rf.predict(P) == nn1.predict(P)).mean()))

        # candidate explainers fitted on the training split only
        Xt, lt = X[te], y[te]
        cands = {
            'tree d1': DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X[tr], y[tr]).predict,
            'tree d3': DecisionTreeClassifier(max_depth=3, random_state=seed).fit(X[tr], y[tr]).predict,
            'medoid': lambda Z: medoid_predict(X[tr], y[tr], Z),
            'kNN-40': KNeighborsClassifier(40).fit(X[tr], y[tr]).predict,
            'kNN-80': KNeighborsClassifier(80).fit(X[tr], y[tr]).predict,
        }
        # prediction-relative fidelity against each extension and probe set
        fid = {}
        for pname, P in probes.items():
            for ename, ext in (('RF', rf), ('1-NN', nn1)):
                ref = ext.predict(P)
                fid[(pname, ename)] = {c: (f(P) == ref).mean() for c, f in cands.items()}
        names = list(cands)
        for pname in probes:
            a, b = fid[(pname, 'RF')], fid[(pname, '1-NN')]
            flips = sum((a[u] - a[v]) * (b[u] - b[v]) < 0
                        for ii, u in enumerate(names) for v in names[ii + 1:])
            rank_rows.append(dict(dataset=dname, seed=seed, probes=pname,
                                  top_differs=max(a, key=a.get) != max(b, key=b.get),
                                  pair_flips=flips, tau=kendalltau(
                                      [a[c] for c in names], [b[c] for c in names])[0]))

        # extension-free held-out relational scores
        dist = cdist(Xt, Xt)
        np.fill_diagonal(dist, np.inf)
        ii, jj = np.triu_indices(len(lt), 1)
        ctx = {'all pairs': (ii, jj)}
        for k in KS:
            ctx[f'strat-{k}'] = stratified(lt, dist, k)
        rng = np.random.default_rng(seed)
        ctx3 = knn_graph(dist, 3)
        ctx['kNN graph 3'] = ctx3
        preds = {c: f(Xt) for c, f in cands.items()}
        for c in names:
            for cn, (I, J) in ctx.items():
                S, D, F = pair_scores(preds[c], lt, I, J)
                rankctx_rows.append(dict(dataset=dname, seed=seed, explainer=c,
                                         context=cn, S=S, D=D, F=F))

        # controlled errors
        opp = np.where(lt[:, None] != lt[None, :], dist, np.inf).min(1)
        order = np.argsort(opp)
        for case, idx in (('boundary', order[:N_FLIPS]), ('interior', order[-N_FLIPS:])):
            pred = lt.copy()
            pred[idx] = 1 - pred[idx]
            for cn, (I, J) in ctx.items():
                S, D, F = pair_scores(pred, lt, I, J)
                row = dict(dataset=dname, seed=seed, case=case, context=cn, S=S, D=D, F=F,
                           n_pairs=len(I))
                if cn.startswith('strat'):
                    Fr = np.nanmean([pair_scores(pred, lt, *matched_random(lt, I, J, ii, jj, rng))[2]
                                     for _ in range(10)])
                    row['F_random_matched'] = Fr
                sweep_rows.append(row)

probe = pd.DataFrame(probe_rows)
rank = pd.DataFrame(rank_rows)
sweep = pd.DataFrame(sweep_rows)
rankctx = pd.DataFrame(rankctx_rows)
probe.to_csv(OUT / 'position_probe_agreement.csv', index=False)
rank.to_csv(OUT / 'position_extension_rank.csv', index=False)
sweep.to_csv(OUT / 'position_context_sensitivity.csv', index=False)
rankctx.to_csv(OUT / 'position_context_rank.csv', index=False)

# rank agreement of explainers between contexts (natural explainers)
agree = []
for (d, s), g in rankctx.groupby(['dataset', 'seed']):
    base = g[g.context == 'all pairs'].set_index('explainer').F
    for cn in ('strat-3', 'strat-15', 'kNN graph 3'):
        loc = g[g.context == cn].set_index('explainer').F
        names = list(base.index)
        rev = sum((base[u] - base[v]) * (loc[u] - loc[v]) < 0
                  for i, u in enumerate(names) for v in names[i + 1:])
        agree.append(dict(dataset=d, seed=s, context=cn, reversals=rev,
                          tau=kendalltau(base[names], loc[names])[0]))
agree = pd.DataFrame(agree)
agree.to_csv(OUT / 'position_context_rank_agreement.csv', index=False)

pd.set_option('display.width', 200)
print('== probe agreement RF vs 1-NN (mean, min, max)')
print(probe.groupby(['dataset', 'probes']).agree.agg(['mean', 'min', 'max', 'count']))
print('== extension-induced ranking changes (5 explainers, 10 pairs)')
print(rank.groupby(['dataset', 'probes']).agg(top_differs=('top_differs', 'mean'),
      any_flip=('pair_flips', lambda v: (v > 0).mean()), mean_flips=('pair_flips', 'mean'),
      tau=('tau', 'mean')))
print('== rank agreement all-pairs vs local context (reversals of 10 pairs, tau)')
print(agree.groupby(['dataset', 'context']).agg(rev=('reversals', 'mean'),
      any_rev=('reversals', lambda v: (v > 0).mean()), tau=('tau', 'mean')))
print('== sweep means')
print(sweep.groupby(['dataset', 'case', 'context'])[['n_pairs', 'S', 'D', 'F', 'F_random_matched']]
      .mean().round(3).to_string())

# ---- figures
plt.rcParams.update({'font.size': 8})
# The probe-agreement figure is drawn by make_probe_figure.py from position_probe_agreement.csv.

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, d in zip(axes, GENERATORS):
    for case, col, mk in (('boundary', 'black', 'o'), ('interior', '0.5', 's')):
        g = sweep[(sweep.dataset == d) & (sweep.case == case) & sweep.context.str.startswith('strat')]
        g = g.assign(k=g.context.str.split('-').str[1].astype(int))
        m, s = g.groupby('k').F.mean(), g.groupby('k').F.std()
        ax.plot(m.index, m.values, color=col, marker=mk, ms=3.5, label=f'{case} errors')
        ax.fill_between(m.index, m - s, m + s, color=col, alpha=.15, lw=0)
        if case == 'boundary':
            r = g.groupby('k').F_random_matched.mean()
            ax.plot(r.index, r.values, color=col, ls=':', marker='x', ms=3.5,
                    label='boundary, size-matched random pairs')
            ap = sweep[(sweep.dataset == d) & (sweep.case == case) & (sweep.context == 'all pairs')].F.mean()
            ax.axhline(ap, color='tab:red', ls='--', lw=.9, label='all pairs (both cases)')
    ax.set_xscale('log')
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.minorticks_off()
    ax.set_xlabel('neighborhood size $k$ (log scale)')
    ax.set_title(d)
axes[0].set_ylabel(r'held-out $\mathcal{F}_{\mathcal{W}_k}$')
axes[0].legend(frameon=False, fontsize=6.5, loc='lower right')
fig.tight_layout()
fig.savefig(OUT / 'position_context_sensitivity.pdf', bbox_inches='tight')
fig.savefig(OUT / 'position_context_sensitivity.png', dpi=220, bbox_inches='tight')
plt.close(fig)
