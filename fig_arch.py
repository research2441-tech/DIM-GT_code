"""Fig. 1 — DIM-GT architecture (black and white, wide format).
Rows: top = training objective (gradients flow down, dashed), middle = forward
inference path, bottom = replay buffer and open-world adaptation (dashed updates)."""
import os
import matplotlib
BASE = os.path.dirname(os.path.abspath(__file__)) + '/'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif'})

fig, ax = plt.subplots(figsize=(7.16, 4.25), dpi=400)
ax.set_xlim(0, 100); ax.set_ylim(-1, 58); ax.axis('off')
DASH = (0, (3.2, 2.0))


def box(x, y, w, h, title, lines=(), fill='white', ls='-', tsize=6.0, lsize=5.25, gap=2.55):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.7', fc=fill, ec='black', lw=0.9, ls=ls))
    tl = title.split('\n')
    for i, t in enumerate(tl):
        ax.text(x + w / 2, y + h - 1.6 - 2.6 * i, t, ha='center', va='top', fontsize=tsize, fontweight='bold')
    y0 = y + h - 1.6 - 2.6 * len(tl) - 0.9
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y0 - gap * i, t, ha='center', va='top', fontsize=lsize)


def arrow(pts, ls='-', lw=0.9):
    """poly-line arrow through the given points; head at the last point."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color='black', lw=lw, ls=ls, solid_capstyle='butt')
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle='-|>', lw=lw, ls=ls, color='black', mutation_scale=7, shrinkA=0, shrinkB=0))


def label(x, y, t, ha='left'):
    ax.text(x, y, t, fontsize=5.1, ha=ha, va='center', style='italic', bbox=dict(fc='white', ec='none', pad=0.4))


# ------------------------------------------------ middle row: forward inference
Y, H = 18.0, 19.0
xs = [(0.8, 16.6), (20.6, 16.0), (40.6, 17.4), (62.0, 16.6), (82.6, 16.6)]
box(*xs[0][:1], Y, xs[0][1], H, 'Temporal social\ngraph',
    [r'$\mathcal{G}_{t-2},\ \mathcal{G}_{t-1},\ \mathcal{G}_{t}$', 'users, posts, URLs,', 'hashtags, mentions,', 'replies, reposts'], fill='#F2F2F2')
box(xs[1][0], Y, xs[1][1], H, 'Relation–time\ntokenizer',
    [r'$\mathbf{x}_v^{\tau}$ and $\bar{\mathbf{A}}_r^{\tau}\mathbf{X}^{\tau}$', r'$K=(1+R)W$ tokens', 'relation embedding', 'Fourier time code'])
box(xs[2][0], Y, xs[2][1], H, 'Relational graph\ntransformer',
    [r'$L$ pre-LN blocks', 'multi-head attention', 'across tokens', r'self-query pooling $\rightarrow\mathbf{z}_v$'])
box(xs[3][0], Y, xs[3][1], H, 'Immune prototype\nmemory',
    [r'self $\mathcal{M}_0$ ($M_s$ cells)', r'non-self $\mathcal{M}_1,\ldots,\mathcal{M}_C$', 'cosine affinity', 'soft-nearest logits'], fill='#F2F2F2')
box(xs[4][0], Y, xs[4][1], H, 'Conformal immune\ndecision',
    [r'Mondrian $p_c(\mathbf{z}_v)$', r'$\Gamma_\alpha=\{c:\,p_c>\alpha\}$', r'$|\Gamma_\alpha|=1$: label', r'$\emptyset$: novel; $>1$: abstain'], fill='#F2F2F2')
for (xa, wa), (xb, _) in zip(xs[:-1], xs[1:]):
    arrow([(xa + wa + 0.35, Y + H / 2), (xb - 0.35, Y + H / 2)], lw=1.05)

# ------------------------------------------------ top row: training objective
Yt, Ht = 43.0, 13.5
box(40.6, Yt, 38.0, Ht, 'Training objective (end-to-end)',
    [r'$\mathcal{L}=\mathcal{L}_{\mathrm{LA}}+\lambda_c\mathcal{L}_{\mathrm{BSC}}+\lambda_n\mathcal{L}_{\mathrm{NS}}\,[+\lambda_d\mathcal{L}_{\mathrm{KD}}]$',
     'logit adjustment, balanced contrast,', 'negative selection; KD in updates'], tsize=6.0, lsize=5.2, gap=2.8)
# memory -> objective (forward: logits and embeddings)
arrow([(75.0, Y + H + 0.35), (75.0, Yt - 0.35)], lw=0.95)
label(75.8, (Y + H + Yt) / 2, r'logits, $\mathbf{z}_v$')
# reverse-mode gradients (dashed): objective -> transformer, objective -> memory
arrow([(49.3, Yt - 0.35), (49.3, Y + H + 0.35)], ls=DASH, lw=0.95)
label(50.1, (Y + H + Yt) / 2, r'$\nabla_{\theta}\mathcal{L}$')
arrow([(66.0, Yt - 0.35), (66.0, Y + H + 0.35)], ls=DASH, lw=0.95)
label(66.8, (Y + H + Yt) / 2, r'$\nabla_{\mu}\mathcal{L}$')

# ------------------------------------------------ bottom row: replay and adaptation
Yb, Hb = 0.2, 12.5
box(40.6, Yb, 17.4, Hb, 'Memory-cell replay',
    ['typical + boundary', 'exemplars per class;', 'old logits as KD targets'], ls=DASH)
box(62.0, Yb, 37.2, Hb, 'Open-world adaptation',
    [r'self tolerance: bounded EMA of $\mathcal{M}_0$',
     r'tolerance induction: new self cells',
     r'quarantine $\rightarrow$ labels $\rightarrow$ new $\mathcal{M}_{C+1}$'], ls=DASH, gap=2.6)
# decision -> adaptation (novel / abstain / confident-self nodes)
arrow([(90.9, Y - 0.35), (90.9, Yb + Hb + 0.35)], lw=0.95)
ax.text(90.0, (Y + Yb + Hb) / 2 + 1.25, 'novel / abstain /', fontsize=5.0, ha='right', va='center', style='italic')
ax.text(90.0, (Y + Yb + Hb) / 2 - 1.25, 'confident self', fontsize=5.0, ha='right', va='center', style='italic')
# adaptation -> memory (dashed update)
arrow([(70.3, Yb + Hb + 0.35), (70.3, Y - 0.35)], ls=DASH, lw=0.95)
label(71.1, (Y + Yb + Hb) / 2, r'update $\mathcal{M}$')
# adaptation -> replay buffer (new labels)
arrow([(61.6, Yb + Hb / 2), (58.4, Yb + Hb / 2)], ls=DASH, lw=0.95)
# replay -> transformer (replayed batch re-encoded)
arrow([(49.3, Yb + Hb + 0.35), (49.3, Y - 0.35)], ls=DASH, lw=0.95)
label(50.1, (Y + Yb + Hb) / 2, 'replay batch')

# ------------------------------------------------ legend (top-left, clear of all boxes)
lx, ly = 2.0, 53.0
ax.plot([lx, lx + 5], [ly, ly], color='black', lw=1.05); ax.text(lx + 6.2, ly, 'forward inference', fontsize=5.4, va='center')
ax.plot([lx, lx + 5], [ly - 4.2, ly - 4.2], color='black', lw=0.95, ls=DASH)
ax.text(lx + 6.2, ly - 4.2, 'reverse-mode gradient / adaptation', fontsize=5.4, va='center')
ax.add_patch(FancyBboxPatch((lx, ly - 9.6), 5.0, 2.6, boxstyle='round,pad=0.1,rounding_size=0.3', fc='#F2F2F2', ec='black', lw=0.7))
ax.text(lx + 6.2, ly - 8.3, 'immune components', fontsize=5.4, va='center')
plt.savefig(BASE + 'figs/fig1_arch.png', bbox_inches='tight', pad_inches=0.03)
print('ok')
