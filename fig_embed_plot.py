"""Plot the interpretability figure from runs/embed_cache.npz."""
import os
import numpy as np, json, matplotlib
BASE = os.path.dirname(os.path.abspath(__file__)) + '/'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif', 'font.size': 6.5,
                     'axes.linewidth': 0.6})
z = np.load(BASE + 'runs/embed_cache.npz')
Z, ZP, own, fam, att, rid, tid = z['Z'], z['ZP'], z['owner'], z['fam'], z['att'], z['rid'], z['tid']
fig = plt.figure(figsize=(3.5, 4.6))
ax = fig.add_axes([0.08, 0.585, 0.89, 0.38])
mk = {0: ('o', '0.72', 'legitimate'), 1: ('s', '0.0', 'F1 URL (known)'), 2: ('^', '0.0', 'F2 tag (known)'),
      3: ('v', '0.0', 'F3 farm (known)'), 4: ('x', '0.35', 'F4 reply (new)'), 5: ('+', '0.35', 'F5 ring (new)'),
      6: ('D', '0.5', 'F6 mimic (new)')}
sub = np.random.default_rng(0)
for f, (m_, g, lab) in mk.items():
    idx = np.where(fam == f)[0]
    idx = sub.choice(idx, min(700, len(idx)), replace=False) if f == 0 else idx[:150]
    hollow = m_ in ('s', '^', 'v', 'D')
    ax.scatter(Z[idx, 0], Z[idx, 1], s=3 if f == 0 else 6, marker=m_, lw=0.45,
               facecolors='none' if hollow else g, edgecolors=g if hollow else None, c=None if hollow else g, label=lab)
ax.scatter(ZP[own == 0, 0], ZP[own == 0, 1], s=46, marker='*', facecolors='white', edgecolors='black', lw=0.7, label='self memory cell', zorder=5)
ax.scatter(ZP[own > 0, 0], ZP[own > 0, 1], s=30, marker='P', c='black', edgecolors='white', lw=0.4, label='non-self memory cell', zorder=5)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
ax.set_title('(a) embeddings at $t=7$ with memory prototypes', fontsize=6.8)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=5.5, markerscale=1.0, frameon=False,
          handletextpad=0.3, columnspacing=1.2)
ax2 = fig.add_axes([0.20, 0.10, 0.77, 0.30])
relnames = ['self', 'FOLLOW', 'REPOST', 'REPLY', 'MENTION', 'CO-URL', 'CO-TAG']
rows = []
for f in range(7):
    w = att[fam == f].mean(0); agg = np.zeros(7)
    for k in range(len(rid)): agg[rid[k]] += w[k]
    rows.append(agg)
M = np.array(rows)
ax2.imshow(M, cmap='Greys', aspect='auto', vmin=0, vmax=1)
for i in range(7):
    for j in range(7):
        ax2.text(j, i, '%.2f' % M[i, j], ha='center', va='center', fontsize=5.0, color='white' if M[i, j] > 0.55 else 'black')
ax2.set_xticks(range(7)); ax2.set_xticklabels(relnames, rotation=35, ha='right', fontsize=5.6)
ax2.set_yticks(range(7)); ax2.set_yticklabels(['legitimate', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6'], fontsize=5.8)
ax2.set_title('(b) mean pooling weight by relation', fontsize=6.8)
fig.savefig(BASE + 'figs/fig_embed.png', dpi=400)
lag = [[float(att[fam == f][:, tid == l].sum(1).mean()) for l in (0, 1, 2)] for f in range(7)]
json.dump(dict(rel=M.tolist(), lag=lag), open(BASE + 'runs/attn.json', 'w'))
print(np.round(M, 2)); print(np.round(lag, 2))
