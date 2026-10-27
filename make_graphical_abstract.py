"""Graphical abstract: the four declarations and one result for each (numbers from position_*.csv)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT = Path(__file__).resolve().parent
W, H = 13.0, 5.2
fig = plt.figure(figsize=(W, H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
INK, GRAY, ACC = '#1b1b1b', '#5a5a5a', '#b3261e'

ax.text(.35, H - .32, 'Faithful to what?', fontsize=21, fontweight='bold', color=INK, va='center')
ax.text(3.6, H - .32, 'Declare the reference object when evaluating clustering explanations',
        fontsize=13, color=GRAY, va='center')

boxes = [
    ('P1', 'Name the explanandum', 'Observed partition, or partition\nplus a deployed assignment rule',
     'Spectral, agglomerative and graph\nclusterings have no native rule'),
    ('P2', 'Declare extension', 'A surrogate or reassignment rule\nis an added model',
     'RF vs 1-NN agree on .99 of\nin-distribution queries, .92 of uniform'),
    ('P3', 'Declare context W', 'Observed same / contrast pairs;\nreport sensitivity to k',
     'All pairs: .80 vs .80. Local k=3:\n.63 (boundary) vs .89 (interior)'),
    ('P4', 'Report components', 'S, D, abstention, held-out,\ncomplexity and the adapter',
     'Same Anchors, two adapters:\nF = .46 vs 1.00'),
]
bw, gap, x0, y0, bh = 2.95, .28, .3, 1.5, 2.8
for i, (tag, head, what, res) in enumerate(boxes):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc='#f6f6f6', ec='#444444', lw=1.6))
    ax.text(x + .2, y0 + bh - .33, tag, fontsize=16, fontweight='bold', color=ACC, va='center')
    ax.text(x + .72, y0 + bh - .33, head, fontsize=11.2, fontweight='bold', color=INK, va='center')
    ax.text(x + .2, y0 + bh - 1.05, what, fontsize=10.8, color=INK, va='center', linespacing=1.35)
    ax.plot([x + .2, x + bw - .2], [y0 + 1.0, y0 + 1.0], color='#bbbbbb', lw=1)
    ax.text(x + .2, y0 + .5, res, fontsize=10.2, color=GRAY, va='center', linespacing=1.35,
            style='italic')
    if i < 3:
        ax.annotate('', xy=(x + bw + gap - .03, y0 + bh / 2), xytext=(x + bw + .03, y0 + bh / 2),
                    arrowprops=dict(arrowstyle='-|>', lw=2, color='#444444'))

ax.text(.3, .98, 'Answer: an explanation is faithful to its declared explanandum. Absent a native rule,\n'
        'the default reference is the observed relational structure of the partition.',
        fontsize=12, fontweight='bold', color=INK, va='center', linespacing=1.35)
ax.text(.3, .3, 'All-pairs, unit-weight W is classical pair counting: a reporting protocol, not a new index. '
        'Results: two synthetic benchmarks, 20 seeds.', fontsize=9.5, color=GRAY, va='center')

fig.savefig(OUT / 'graphical_abstract.png', dpi=200)
Image.open(OUT / 'graphical_abstract.png').convert('RGB').save(OUT / 'graphical_abstract.tiff', compression='tiff_lzw')
