"""Result figures (black and white). Reads runs/results.json and raw run files."""
import json, os, glob, numpy as np
import matplotlib
BASE = os.path.dirname(os.path.abspath(__file__)) + '/'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})
RUNS = BASE + 'runs'; FIG = BASE + 'figs'
R = json.load(open(os.path.join(RUNS, 'results.json')))
STY = {  # method: (label, marker, linestyle, gray)
    'dimgt': ('DIM-GT', 'o', '-', '0.0'), 'er': ('ER', 's', '--', '0.25'), 'lwf': ('LwF', '^', '-.', '0.35'),
    'ewc': ('EWC', 'v', ':', '0.45'), 'ft': ('Fine-tune', 'x', ':', '0.6'), 'tokce': ('TokCE (cumul.)', 'D', '--', '0.5'),
    'gbdt': ('GBDT-Agg (cumul.)', 'P', '-', '0.65'), 'rgcn': ('R-GCN (cumul.)', '*', '-.', '0.55'),
    'lstm': ('LSTM (cumul.)', 'h', ':', '0.3'), 'nsaga': ('NSA-GA (cumul.)', '+', '--', '0.7')}


def stream_fig(methods=('dimgt', 'er', 'lwf', 'ewc', 'ft', 'tokce', 'gbdt', 'rgcn')):
    S = R['stream']
    keys = [('macro_f1', 'Binary Macro-F1'), ('legit_fpr', 'Legitimate FPR'), ('old_family_recall', 'F1–F3 flag rate')]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    ts = np.arange(2, 10)
    for ax, (k, lab) in zip(axes, keys):
        for m in methods:
            if m not in S: continue
            l, mk, ls, g = STY[m]
            y = np.array(S[m][k + '_t']); sd = np.array(S[m][k + '_t_sd'])
            ax.plot(ts, y, ls=ls, marker=mk, ms=3.2, lw=1.4 if m == 'dimgt' else 0.9, color=g, label=l,
                    mfc='white' if m not in ('dimgt',) else g, zorder=5 if m == 'dimgt' else 3)
            if m == 'dimgt':
                ax.fill_between(ts, y - sd, y + sd, color='0.85', lw=0, zorder=1)
        for te, nm in ((3, 'F4'), (5, 'F5'), (6, 'event'), (7, 'F6'), (8, 'F7')):
            ax.axvline(te, color='0.75', lw=0.5, ls=':', zorder=0)
            ax.text(te, 1.02, nm, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5.6, color='0.3')
        ax.set_xlabel('window $t$'); ax.set_title(lab, fontsize=7.2, pad=11)
        ax.set_xticks(ts)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=6.2, bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handlelength=2.6)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.6)
    fig.savefig(os.path.join(FIG, 'fig_stream.png'), dpi=400)
    plt.close(fig)


def family_heatmap():
    S = R['stream']
    fams = [1, 2, 3, 4, 5, 6, 7]
    names = ['F1 URL', 'F2 tag', 'F3 farm', 'F4 reply', 'F5 ring', 'F6 mimic', 'F7 takeover']
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.05), sharey=True)
    for ax, m, title in zip(axes, ('dimgt', 'er'), ('DIM-GT', 'ER')):
        M = np.full((7, 8), np.nan)
        for i, f in enumerate(fams):
            for j, t in enumerate(range(2, 10)):
                v = S[m]['family_recall'].get('%d|%d' % (f, t))
                if v is not None: M[i, j] = v
        im = ax.imshow(M, cmap='Greys', vmin=0, vmax=1, aspect='auto')
        for i in range(7):
            for j in range(8):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, '%.2f' % M[i, j], ha='center', va='center', fontsize=4.4,
                            color='white' if M[i, j] > 0.6 else 'black')
        ax.set_xticks(range(8)); ax.set_xticklabels(range(2, 10), fontsize=5.5)
        ax.set_title(title, fontsize=6.8); ax.set_xlabel('window $t$', fontsize=6)
        for s in ax.spines.values(): s.set_visible(True)
    axes[0].set_yticks(range(7)); axes[0].set_yticklabels(names, fontsize=5.5)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, 'fig_family.png'), dpi=400)
    plt.close(fig)




def _load_npz(ds, m, seeds=range(5)):
    out = []
    for s in seeds:
        f = os.path.join(RUNS, 'real', '%s_%s_s%d.npz' % (ds, m, s))
        if os.path.exists(f):
            z = np.load(f); out.append(z)
    return out


def calib_fig(dss=('yelpchi', 'amazon'), methods=('gbdt', 'tokce', 'rgt', 'dimgt')):
    lab = dict(gbdt=('GBDT-Agg', 's', '--', '0.55'), tokce=('TokCE', '^', ':', '0.35'), rgt=('RGT', 'v', '-.', '0.45'),
               dimgt=('DIM-GT', 'o', '-', '0.0'))
    yall = {ds: np.load(BASE + 'data/%s_tok.npz' % ds)['y'] for ds in dss}
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.3))
    for col, ds in enumerate(dss):
        axr, axc = axes[0, col], axes[1, col]
        axr.plot([0, 1], [0, 1], color='0.75', lw=0.6, ls='-')
        for m in methods:
            Z = _load_npz(ds, m)
            if not Z: continue
            l, mk, ls, g = lab[m]
            p = np.concatenate([z['st'] for z in Z]); y = np.concatenate([yall[ds][z['te']] for z in Z])
            bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
            xs, ys = [], []
            for b in range(10):
                mm = idx == b
                if mm.sum() >= 30: xs.append(p[mm].mean()); ys.append(y[mm].mean())
            axr.plot(xs, ys, marker=mk, ls=ls, color=g, ms=2.6, lw=1.2 if m == 'dimgt' else 0.8, label=l, mfc='white' if m != 'dimgt' else g)
            # risk-coverage
            covs = np.linspace(0.3, 1.0, 36); risks = []
            for z in Z:
                yt = yall[ds][z['te']]; err = ((z['st'] >= z['thr']).astype(int) != yt).astype(float)
                o = np.argsort(-z['conf']); e = err[o]
                risks.append([e[:max(1, int(c * len(e)))].mean() for c in covs])
            axc.plot(covs, np.mean(risks, 0), ls=ls, color=g, lw=1.2 if m == 'dimgt' else 0.8, label=l)
        axr.set_title(dict(yelpchi='YelpChi', amazon='Amazon', weibo='Weibo')[ds], fontsize=7)
        axr.set_xlabel('predicted spam probability', fontsize=6); axr.set_ylabel('observed frequency' if col == 0 else '', fontsize=6)
        axc.set_xlabel('coverage', fontsize=6); axc.set_ylabel('selective risk' if col == 0 else '', fontsize=6)
        for a in (axr, axc): a.tick_params(labelsize=5.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=5.8, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=0.8)
    fig.savefig(os.path.join(FIG, 'fig_calib.png'), dpi=400)
    plt.close(fig)


def open_fig(dss=('yelpchi', 'amazon', 'weibo')):
    O = R['open']
    sc = [('msp', 'MSP', 'o'), ('energy', 'Energy', 's'), ('mahalanobis', 'Mahalanobis', '^'), ('knn', 'k-NN', 'v'),
          ('gnnsafe', 'GNNSafe', 'D'), ('gbdt_msp', 'GBDT-MSP', 'P'), ('nsa', 'NSA', 'X'), ('dimgt_msp', 'DIM-GT (softmax)', 'h'),
          ('dimgt', 'DIM-GT (conformal)', '*')]
    hatches = ['', '////', '\\\\\\\\', 'xxx', '...', '----', '++', 'oo', '']
    grays = ['0.92', 'white', 'white', 'white', 'white', 'white', 'white', '0.7', '0.15']
    fig = plt.figure(figsize=(7.1, 2.45))
    ax = fig.add_axes([0.06, 0.30, 0.50, 0.60])
    W = 0.13
    scA = [('msp', 'TokCE (softmax)', 'o'), ('gnnsafe', 'GraphSAGE (softmax)', 'D'), ('gbdt_msp', 'GBDT-Agg', 'P'), ('nsa', 'NSA', 'X'),
           ('dimgt_msp', 'DIM-GT (softmax)', 'h'), ('dimgt', 'DIM-GT (conformal $1-p_0$)', '*')]
    hatches = ['', '....', '----', '++', 'oo', '']
    grays = ['0.92', 'white', 'white', 'white', '0.7', '0.15']
    for i, (k, nm, _) in enumerate(scA):
        vals = [O.get('%s|%s' % (ds, k), {}).get('novel_vs_legit_auroc', [np.nan, 0])[0] for ds in dss]
        sds = [O.get('%s|%s' % (ds, k), {}).get('novel_vs_legit_auroc', [np.nan, 0])[1] for ds in dss]
        x = np.arange(len(dss)) + (i - 2.5) * W
        ax.bar(x, vals, W * 0.92, yerr=sds, color=grays[i], edgecolor='black', lw=0.4, hatch=hatches[i], label=nm,
               error_kw=dict(lw=0.4, capsize=1))
    ax.set_xticks(range(len(dss))); ax.set_xticklabels([dict(yelpchi='YelpChi', amazon='Amazon', weibo='Weibo')[d] for d in dss])
    ax.set_ylim(0.3, 1.02); ax.set_ylabel('AUROC'); ax.set_title('(a) unseen-strategy spam vs. legitimate (threshold-free)', fontsize=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=5.6, columnspacing=1.2, handlelength=1.6)
    # marker key for panel (b): same order as the bars
    from matplotlib.lines import Line2D
    hk = [Line2D([], [], ls='', marker=mk, mfc='black' if k.startswith('dimgt') else 'white', mec='black', ms=4.5 if k == 'dimgt' else 3.5, label=nm)
          for k, nm, mk in sc]
    ax2 = fig.add_axes([0.66, 0.30, 0.32, 0.60])
    for k, nm, mk in sc:
        xs = [O.get('%s|%s' % (ds, k), {}).get('legit_fpr', [np.nan])[0] for ds in dss]
        ys = [O.get('%s|%s' % (ds, k), {}).get('novel_recall', [np.nan])[0] for ds in dss]
        ax2.scatter(np.nanmean(xs), np.nanmean(ys), marker=mk, s=40 if k == 'dimgt' else 22,
                    c='black' if k.startswith('dimgt') else 'white', edgecolors='black', lw=0.6, zorder=4)
    ax2.set_xlabel('legitimate FPR (mean of 3 graphs)'); ax2.set_ylabel('unseen spam flagged')
    ax2.set_title('(b) operating points at the 5 % rule', fontsize=7)
    ax2.legend(handles=hk, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=5.0, columnspacing=0.8, handletextpad=0.2)
    fig.savefig(os.path.join(FIG, 'fig_open.png'), dpi=400)
    plt.close(fig)


def sens_fig():
    M = R['main']
    params = [('m_self', '$M_s$', [1, 3, 6, 10]), ('lam_ns', '$\\lambda_n$', [0.3, 1.0, 3.0]), ('m_tol', '$m_{\\mathrm{tol}}$', [0.6, 0.75, 0.9]),
              ('lam_c', '$\\lambda_c$', [0.1, 0.3, 1.0]), ('demb', '$d_e$', [16, 32, 64])]
    default = dict(m_self=6, lam_ns=1.0, m_tol=0.75, lam_c=0.3, demb=32)
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 2.9), sharey='row')
    for j, (p, lab, xs) in enumerate(params):
        for ds, mk, ls in (('amazon', 'o', '-'), ('weibo', 's', '--')):
            pr, ec = [], []
            for x in xs:
                key = 'dimgt' if x == default[p] else 'dimgt:%s=%s' % (p, str(x))
                vals = M.get('%s|%s' % (ds, key))
                pr.append(vals['pr_auc'][0] if vals else np.nan); ec.append(vals['ece'][0] if vals else np.nan)
            axes[0, j].plot(range(len(xs)), pr, marker=mk, ls=ls, color='black', ms=3, lw=0.9, mfc='white' if ds == 'weibo' else 'black',
                            label={'amazon': 'Amazon', 'weibo': 'Weibo'}[ds])
            axes[1, j].plot(range(len(xs)), ec, marker=mk, ls=ls, color='0.35', ms=3, lw=0.9, mfc='white' if ds == 'weibo' else '0.35')
        for r in (0, 1):
            axes[r, j].set_xticks(range(len(xs))); axes[r, j].set_xticklabels([('%g' % x) for x in xs], fontsize=5.6)
            axes[r, j].tick_params(labelsize=5.6)
        axes[1, j].set_xlabel(lab, fontsize=7)
    axes[0, 0].set_ylabel('PR-AUC', fontsize=6.5); axes[1, 0].set_ylabel('ECE', fontsize=6.5)
    axes[0, 0].legend(fontsize=5.6, loc='center left')
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    fig.savefig(os.path.join(FIG, 'fig_sens.png'), dpi=400)
    plt.close(fig)


def robust_fig():
    rob = R['robust']; M = R['main']
    lab = dict(gbdt=('GBDT-Agg', 's', '--', '0.55'), rgcn=('R-GCN', 'v', '-.', '0.45'), tokce=('TokCE', '^', ':', '0.3'), dimgt=('DIM-GT', 'o', '-', '0.0'))
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
    for ax, (tags, xl, xs) in zip(axes, ((('', '_ed0.3', '_ed0.6'), 'edges removed', [0, 0.3, 0.6]),
                                          (('', '_fn0.5', '_fn1.0'), 'feature noise $\\sigma$', [0, 0.5, 1.0]))):
        for m, (l, mk, ls, g) in lab.items():
            ys = []
            for tg in tags:
                vals = []
                for ds in ('amazon', 'weibo'):
                    v = rob.get(tg, {}).get('%s|%s' % (ds, m))
                    vals.append(v['pr_auc'][0] if v else np.nan)
                ys.append(np.mean(vals))
            ax.plot(xs, ys, marker=mk, ls=ls, color=g, ms=3, lw=1.2 if m == 'dimgt' else 0.8, label=l, mfc='white' if m != 'dimgt' else g)
        ax.set_xlabel(xl, fontsize=6.5); ax.set_xticks(xs); ax.tick_params(labelsize=5.8)
    axes[0].set_ylabel('PR-AUC (mean of 2 graphs)', fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=5.6, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.6)
    fig.savefig(os.path.join(FIG, 'fig_robust.png'), dpi=400)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        globals()[f]()
