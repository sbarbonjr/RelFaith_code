import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

OUT = Path(__file__).resolve().parent
SEED = 7
X, _ = make_moons(n_samples=240, noise=0.08, random_state=SEED)
labels = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=12, random_state=SEED).fit_predict(X)
rf = RandomForestClassifier(n_estimators=500, random_state=SEED).fit(X, labels)
knn = KNeighborsClassifier(n_neighbors=1).fit(X, labels)
idx = np.arange(len(X))
tr, te = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=labels)
expl = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X[tr], labels[tr])

mins = X.min(0) - 0.4
maxs = X.max(0) + 0.4
rng = np.random.default_rng(SEED)
probes = rng.uniform(mins, maxs, size=(20000, 2))
pE = expl.predict(probes)
pRF = rf.predict(probes)
pK = knn.predict(probes)

extension_sensitivity = []
for extension_k in (1, 3, 5, 10, 15, 20):
    extension = knn if extension_k == 1 else KNeighborsClassifier(
        n_neighbors=extension_k).fit(X, labels)
    extension_sensitivity.append({
        'knn_k': extension_k,
        'observed_label_accuracy': accuracy_score(labels, extension.predict(X)),
        'agreement_with_rf_offsample': accuracy_score(pRF, extension.predict(probes)),
        'agreement_with_tree_offsample': accuracy_score(pE, extension.predict(probes)),
    })

Xt, lt = X[te], labels[te]
pe = expl.predict(Xt)
distances = cdist(Xt, Xt)
np.fill_diagonal(distances, np.inf)

def make_context(k):
    """Union of each point's k nearest same- and different-cluster pairs."""
    selected = set()
    for i in range(len(lt)):
        for same_relation in (True, False):
            candidates = np.flatnonzero((lt == lt[i]) == same_relation)
            candidates = candidates[candidates != i]
            nearest = candidates[np.argsort(distances[i, candidates])[:k]]
            for j in nearest:
                selected.add((min(i, int(j)), max(i, int(j))))
    return sorted(selected)

def relation_scores(prediction, pairs, covered=None):
    if covered is None:
        covered = np.ones(len(lt), dtype=bool)
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
    S = same_correct / same_total if same_total else np.nan
    D = contrast_correct / contrast_total if contrast_total else np.nan
    FW = 2 * S * D / (S + D) if S + D else np.nan
    return {
        'n_pairs': len(pairs), 'n_same_pairs': same_total,
        'n_contrast_pairs': contrast_total, 'S': S, 'D': D, 'F_W': FW,
        'A': abstained / len(pairs) if pairs else np.nan,
    }

all_pairs = [(i, j) for i in range(len(lt)) for j in range(i + 1, len(lt))]
context_ks = (3, 5, 10, 15, 20)
contexts = {'all-pairs': all_pairs}
contexts.update({f'stratified-{k}-NN': make_context(k) for k in context_ks})

# Controlled errors move the same number of assignments to either side of the
# reference boundary. The reference partition itself remains fixed.
opposite_distances = np.where(lt[:, None] != lt[None, :], distances, np.inf)
nearest_opposite = opposite_distances.min(axis=1)
n_flips = 8
boundary_indices = np.argsort(nearest_opposite)[:n_flips]
interior_indices = np.argsort(nearest_opposite)[-n_flips:]
boundary_prediction = lt.copy()
boundary_prediction[boundary_indices] = 1 - boundary_prediction[boundary_indices]
interior_prediction = lt.copy()
interior_prediction[interior_indices] = 1 - interior_prediction[interior_indices]

partial_coverage = np.ones(len(lt), dtype=bool)
partial_coverage[np.argsort(Xt[:, 0])[:int(0.20 * len(lt))]] = False
validation_cases = {
    'depth-1 tree': (pe, None),
    'exact partition': (lt, None),
    'boundary errors (8 points)': (boundary_prediction, None),
    'interior errors (8 points)': (interior_prediction, None),
    'partial tree adapter': (pe, partial_coverage),
}
context_results = []
for case, (prediction, covered) in validation_cases.items():
    for context_name, pairs in contexts.items():
        context_results.append({
            'case': case, 'context': context_name,
            **relation_scores(prediction, pairs, covered),
        })

context_frame = pd.DataFrame(context_results)
context_frame.to_csv(OUT / 'sanity_context_sensitivity.csv', index=False)
pd.DataFrame(extension_sensitivity).to_csv(
    OUT / 'sanity_extension_sensitivity.csv', index=False)

# Candidate explanatory rules are fitted only on the training observations and
# evaluated on held-out observed pairs.  RF and k-NN are included as flexible
# surrogate rules; shallow trees are the interpretable explanation family.
candidate_specs = []
for depth in (1, 2, 3, 4, 5):
    candidate_specs.append((
        'decision tree', depth,
        DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X[tr], labels[tr])))
for k in (1, 3, 5, 10, 20, 40, 60, 80):
    candidate_specs.append((
        'k-NN', k, KNeighborsClassifier(n_neighbors=k).fit(X[tr], labels[tr])))
for depth in (1, 2, 3, 4, 5):
    candidate_specs.append((
        'Random Forest', depth,
        RandomForestClassifier(n_estimators=300, max_depth=depth, max_features=1,
                               random_state=SEED).fit(X[tr], labels[tr])))

candidate_results = []
for family, parameter, model in candidate_specs:
    prediction = model.predict(Xt)
    for context_name in ('all-pairs', 'stratified-15-NN'):
        candidate_results.append({
            'family': family, 'parameter': parameter, 'context': context_name,
            **relation_scores(prediction, contexts[context_name]),
        })
candidate_frame = pd.DataFrame(candidate_results)
candidate_frame.to_csv(OUT / 'sanity_candidate_scores.csv', index=False)

# Non-surrogate explanations.  A rule explanation is a central axis-aligned
# box for each cluster; ambiguous or uncovered points abstain.  A prototype
# explanation consists only of labelled exemplars plus a nearest-exemplar
# adapter, without fitting a predictive model.
def rule_box_explanation(coverage):
    boxes = {}
    lower_q = (1.0 - coverage) / 2.0
    upper_q = 1.0 - lower_q
    for cluster in np.unique(labels):
        cluster_train = X[tr][labels[tr] == cluster]
        boxes[cluster] = (
            np.quantile(cluster_train, lower_q, axis=0),
            np.quantile(cluster_train, upper_q, axis=0),
        )
    prediction = np.zeros(len(Xt), dtype=int)
    covered = np.zeros(len(Xt), dtype=bool)
    for point_idx, point in enumerate(Xt):
        matches = [cluster for cluster, (lower, upper) in boxes.items()
                   if np.all(point >= lower) and np.all(point <= upper)]
        if len(matches) == 1:
            prediction[point_idx] = matches[0]
            covered[point_idx] = True
    return prediction, covered

def select_prototypes(points, n_prototypes):
    """Deterministic farthest-point exemplar selection within one cluster."""
    within_cluster_distances = cdist(points, points)
    selected = [int(np.argmin(within_cluster_distances.sum(axis=1)))]
    while len(selected) < min(n_prototypes, len(points)):
        distances_to_selected = cdist(points, points[selected]).min(axis=1)
        selected.append(int(np.argmax(distances_to_selected)))
    return points[selected]

def prototype_explanation(n_prototypes):
    exemplar_points, exemplar_labels = [], []
    for cluster in np.unique(labels):
        cluster_train = X[tr][labels[tr] == cluster]
        selected = select_prototypes(cluster_train, n_prototypes)
        exemplar_points.extend(selected)
        exemplar_labels.extend([cluster] * len(selected))
    nearest = cdist(Xt, np.asarray(exemplar_points)).argmin(axis=1)
    return np.asarray(exemplar_labels)[nearest], np.ones(len(Xt), dtype=bool)

direct_results = []
for coverage in (0.60, 0.70, 0.80, 0.90, 1.00):
    prediction, covered = rule_box_explanation(coverage)
    for context_name in ('all-pairs', 'stratified-15-NN'):
        direct_results.append({
            'explanation': 'rule boxes', 'parameter': coverage,
            'parameter_name': 'central coverage', 'context': context_name,
            **relation_scores(prediction, contexts[context_name], covered),
        })
for n_prototypes in (1, 2, 3, 5, 10):
    prediction, covered = prototype_explanation(n_prototypes)
    for context_name in ('all-pairs', 'stratified-15-NN'):
        direct_results.append({
            'explanation': 'prototypes', 'parameter': n_prototypes,
            'parameter_name': 'prototypes per cluster', 'context': context_name,
            **relation_scores(prediction, contexts[context_name], covered),
        })
direct_frame = pd.DataFrame(direct_results)
direct_frame.to_csv(OUT / 'sanity_direct_explanations.csv', index=False)

tree_all_pairs = relation_scores(pe, all_pairs)
tree_context_15 = relation_scores(pe, contexts['stratified-15-NN'])

metrics = {
    'spectral_n': len(X),
    'train_fidelity_rf': accuracy_score(labels, rf.predict(X)),
    'train_fidelity_1nn': accuracy_score(labels, knn.predict(X)),
    'explanation_train_accuracy': accuracy_score(labels[tr], expl.predict(X[tr])),
    'explanation_test_accuracy': accuracy_score(labels[te], expl.predict(X[te])),
    'offsample_agreement_expl_rf': accuracy_score(pE, pRF),
    'offsample_agreement_expl_1nn': accuracy_score(pE, pK),
    'offsample_agreement_rf_1nn': accuracy_score(pRF, pK),
    'heldout_all_pairs_S': tree_all_pairs['S'],
    'heldout_all_pairs_D': tree_all_pairs['D'],
    'heldout_all_pairs_F_W': tree_all_pairs['F_W'],
    'heldout_stratified_15nn_pairs': tree_context_15['n_pairs'],
    'heldout_stratified_15nn_S': tree_context_15['S'],
    'heldout_stratified_15nn_D': tree_context_15['D'],
    'heldout_stratified_15nn_F_W': tree_context_15['F_W'],
}
pd.DataFrame([metrics]).to_csv(OUT / 'sanity_check_metrics.csv', index=False)

# Figure 1: observed partition and stump boundary
fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.scatter(X[:, 0], X[:, 1], c=labels, s=18)
# decision stump has one split; draw the split threshold
feature = int(expl.tree_.feature[0])
threshold = float(expl.tree_.threshold[0])
if feature == 0:
    ax.axvline(threshold, linestyle='--', linewidth=1.5)
else:
    ax.axhline(threshold, linestyle='--', linewidth=1.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Observed spectral partition and shallow explanation')
fig.tight_layout()
fig.savefig(OUT / 'sanity_observed_partition.pdf', bbox_inches='tight')
fig.savefig(OUT / 'sanity_observed_partition.png', dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 2: the two assignment regions and their disagreement
xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 220), np.linspace(mins[1], maxs[1], 180))
grid = np.c_[xx.ravel(), yy.ravel()]
pred_rf = rf.predict(grid).reshape(xx.shape)
pred_knn = knn.predict(grid).reshape(xx.shape)
disagree = (pred_rf != pred_knn)
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharex=True, sharey=True)
region_colors = ['white', '0.78']

def add_observed_points(ax):
    # Marker shape identifies the fixed spectral label without relying on color.
    for cluster, marker in ((0, 'o'), (1, 's')):
        mask = labels == cluster
        ax.scatter(X[mask, 0], X[mask, 1], marker=marker, s=19,
                   facecolors='white', edgecolors='black', linewidths=0.65,
                   zorder=3)

for ax, prediction, title in (
    (axes[0], pred_rf, '(a) Random Forest'),
    (axes[1], pred_knn, '(b) 1-nearest neighbor'),
):
    ax.contourf(xx, yy, prediction, levels=[-0.5, 0.5, 1.5],
                colors=region_colors, antialiased=True)
    ax.contour(xx, yy, prediction, levels=[0.5], colors='black', linewidths=0.7)
    add_observed_points(ax)
    ax.set_title(title)

axes[2].contourf(xx, yy, disagree.astype(float), levels=[0.5, 1.5],
                 colors=['0.82'], hatches=['////'], antialiased=True)
add_observed_points(axes[2])
axes[2].set_title('(c) Disagreement')

for ax in axes:
    ax.set_xlabel('$x_1$')
    ax.set_aspect('equal', adjustable='box')
axes[0].set_ylabel('$x_2$')
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / 'sanity_surrogate_disagreement.pdf', bbox_inches='tight')
fig.savefig(OUT / 'sanity_surrogate_disagreement.png', dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 3: held-out RelFaith scores for candidate explanatory rules.
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0), sharey=True)
families = (
    ('decision tree', '(a) Shallow tree explanations', 'Maximum depth'),
    ('k-NN', '(b) k-NN surrogate rules', '$k$'),
    ('Random Forest', '(c) Random Forest surrogate rules', 'Maximum depth'),
)
for ax, (family, title, xlabel) in zip(axes, families):
    family_scores = candidate_frame[candidate_frame['family'] == family]
    for context_name, color, marker, style in (
        ('all-pairs', 'black', 'o', '-'),
        ('stratified-15-NN', '0.45', 's', '--'),
    ):
        scores = family_scores[family_scores['context'] == context_name].sort_values('parameter')
        ax.plot(scores['parameter'], scores['F_W'], color=color, marker=marker,
                linestyle=style, label=context_name.replace('stratified-15-NN', r'neighborhood $\mathcal{W}_{15}$'))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(sorted(family_scores['parameter'].unique()))
    ax.set_ylim(0.45, 1.02)
axes[0].set_ylabel(r'Held-out $\mathcal{F}_{\mathcal{W}}$')
axes[1].set_xticks((1, 10, 20, 40, 60, 80))
axes[0].legend(frameon=False, fontsize=7.6, loc='lower right')
fig.tight_layout(w_pad=1.4)
fig.savefig(OUT / 'sanity_validation_sensitivity.pdf', bbox_inches='tight')
fig.savefig(OUT / 'sanity_validation_sensitivity.png', dpi=220, bbox_inches='tight')
plt.close(fig)

print(metrics)
print(pd.DataFrame(extension_sensitivity).to_string(index=False))
print(candidate_frame.to_string(index=False))
print(direct_frame.to_string(index=False))
print(context_frame[context_frame['case'].isin(
    ['depth-1 tree', 'boundary errors (8 points)', 'interior errors (8 points)',
     'partial tree adapter'])].to_string(index=False))
