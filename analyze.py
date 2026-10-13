"""Aggregate all experiment outputs into results.json (tables + statistics)."""
import json, glob, os, collections, numpy as np
from scipy import stats

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
DS = ['yelpchi', 'amazon', 'weibo', 'reddit']
METHODS = ['lr', 'rf', 'gbdt', 'nsa', 'nsaga', 'mlp', 'cnn', 'lstm', 'sage', 'rgcn', 'bwgnn', 'rgt', 'nag', 'tokce', 'dimgt']
NAMES = dict(lr='LR', rf='RF', gbdt='GBDT-Agg', nsa='NSA', nsaga='NSA-GA', mlp='MLP', cnn='1D-CNN', lstm='LSTM', sage='GraphSAGE',
             rgcn='R-GCN', bwgnn='BWGNN', rgt='RGT', nag='NAGphormer', tokce='TokCE', dimgt='DIM-GT')


def load_real(tag=''):
    R = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RUNS, 'real', '*.json')):
        b = os.path.basename(f)
        if '_smoke' in b: continue
        r = json.load(open(f))
        ds_tag = b.split('_')[0] if not tag else None
        prefix = r['ds'] + tag + '_'
        if not b.startswith(prefix): continue
        if tag == '' and (b.startswith(r['ds'] + '_ed') or b.startswith(r['ds'] + '_fn')): continue
        R[(r['ds'], r['method'])][r['seed']] = r
    return R


def wilcoxon_holm(R, metric, ref='dimgt', methods=METHODS, dss=DS):
    out = {}
    for m in methods:
        if m == ref: continue
        a, b = [], []
        for ds in dss:
            for s in range(5):
                if s in R.get((ds, ref), {}) and s in R.get((ds, m), {}):
                    a.append(R[(ds, ref)][s][metric]); b.append(R[(ds, m)][s][metric])
        a, b = np.array(a), np.array(b)
        d = a - b
        if len(d) < 5 or np.allclose(d, 0):
            out[m] = dict(n=len(d), p=1.0, rrb=0.0, mean_diff=float(d.mean()) if len(d) else 0); continue
        w = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        rk = stats.rankdata(np.abs(d[d != 0])); pos = rk[d[d != 0] > 0].sum(); neg = rk[d[d != 0] < 0].sum()
        out[m] = dict(n=len(d), p=float(w.pvalue), rrb=float((pos - neg) / (pos + neg)), mean_diff=float(d.mean()),
                      wins=int((d > 0).sum()), losses=int((d < 0).sum()))
    # Holm
    ms = sorted(out, key=lambda k: out[k]['p'])
    k = len(ms); prev = 0
    for i, m in enumerate(ms):
        adj = min(1.0, max(prev, (k - i) * out[m]['p'])); out[m]['p_holm'] = adj; prev = adj
    return out


def friedman(R, metric, methods=METHODS, dss=DS):
    blocks = []
    for ds in dss:
        for s in range(5):
            if all(s in R.get((ds, m), {}) for m in methods):
                blocks.append([R[(ds, m)][s][metric] for m in methods])
    X = np.array(blocks)
    ranks = np.array([stats.rankdata(-row) for row in X])
    st = stats.friedmanchisquare(*X.T)
    return dict(n_blocks=len(X), chi2=float(st.statistic), p=float(st.pvalue),
                avg_rank={m: float(ranks[:, i].mean()) for i, m in enumerate(methods)})


def summarize(R, keys, methods, dss):
    T = {}
    for ds in dss:
        for m in methods:
            rs = list(R.get((ds, m), {}).values())
            if not rs: continue
            T['%s|%s' % (ds, m)] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])), len(rs)]
                                    for k in keys if k in rs[0] and all(isinstance(r.get(k), (int, float)) for r in rs)}
    return T


def main():
    res = {}
    R = load_real()
    keys = ['macro_f1', 'pr_auc', 'auroc', 'mcc', 'gmean', 'bacc', 'ece', 'brier', 'nll', 'aurc', 'risk_at_80', 'risk_at_90',
            't_train', 'ms_per_1k', 'n_params', 'epochs', 'conf_cov_05', 'conf_size_05', 'conf_single_05', 'conf_sel_err_05',
            'conf_cov_c0_05', 'conf_cov_c1_05', 'conf_cov_10', 'conf_size_10', 'conf_cov_c0_10', 'conf_cov_c1_10']
    ablm = ['tokce', 'dimgt-noproto', 'dimgt-nons', 'dimgt-nobsc', 'dimgt-nola', 'dimgt-norel', 'dimgt-self', 'dimgt']
    sens = sorted(set(m for (_, m) in R if m.startswith('dimgt:')))
    res['main'] = summarize(R, keys, METHODS + ablm + sens, DS)
    done = [m for m in METHODS if all(len(R.get((ds, m), {})) == 5 for ds in DS)]
    if 'dimgt' in done:
        for met in ('pr_auc', 'macro_f1', 'mcc', 'ece', 'auroc'):
            res['wilcoxon_' + met] = wilcoxon_holm(R, met, methods=done)
        for met in ('pr_auc', 'macro_f1', 'mcc'):
            res['friedman_' + met] = friedman(R, met, methods=done)
    # ablation tests vs full model (paired over dataset x seed)
    res['abl_tests'] = {}
    for met in ('pr_auc', 'macro_f1', 'ece'):
        res['abl_tests'][met] = wilcoxon_holm(R, met, methods=[m for m in ablm if m != 'dimgt' and all(len(R.get((d, m), {})) == 5 for d in DS)] + ['dimgt'])
    # robustness
    rob = {}
    for tag in ('_ed0.3', '_ed0.6', '_fn0.5', '_fn1.0'):
        RR = collections.defaultdict(dict)
        for f in glob.glob(os.path.join(RUNS, 'real', '*%s_*.json' % tag)):
            r = json.load(open(f)); RR[(r['ds'], r['method'])][r['seed']] = r
        rob[tag] = summarize(RR, ['pr_auc', 'macro_f1', 'ece'], ['gbdt', 'rgcn', 'nag', 'tokce', 'dimgt'], ['amazon', 'yelpchi', 'weibo'])
    R3 = collections.defaultdict(dict)
    for (ds, m), d in R.items():
        for sd, r in d.items():
            if sd in (0, 1, 2): R3[(ds, m)][sd] = r
    rob[''] = summarize(R3, ['pr_auc', 'macro_f1', 'ece'], ['gbdt', 'rgcn', 'tokce', 'dimgt'], ['amazon', 'weibo'])
    res['robust'] = rob
    # open-set
    O = collections.defaultdict(list)
    for f in glob.glob(os.path.join(RUNS, 'open', '*.json')):
        r = json.load(open(f))
        for k, v in r.items():
            if isinstance(v, dict): O[(r['ds'], k)].append(v)
    res['open'] = {'%s|%s' % k: {m: [float(np.mean([x[m] for x in v])), float(np.std([x[m] for x in v])), len(v)] for m in v[0]}
                   for k, v in O.items()}
    # stream
    S = collections.defaultdict(list)
    for f in glob.glob(os.path.join(RUNS, 'stream', 'A_*.json')):
        r = json.load(open(f)); S[r['method']].append(r)
    st = {}
    for m, rs in S.items():
        def per_t(key):
            arr = []
            for r in rs:
                h = {x['t']: x for x in r['hist']}
                arr.append([h[t].get(key, np.nan) if t in h else np.nan for t in range(2, 10)])
            return np.array(arr, float)
        d = dict(n=len(rs))
        for key in ('macro_f1', 'mcc', 'pr_auc', 'legit_fpr', 'novel_recall', 'novel_isolated', 'open_auroc', 'old_family_recall',
                    'known_family_macro_recall', 'gmean', 'ece', 'abstain', 'accepted_err', 'old_family_exact'):
            a = per_t(key)
            d[key + '_t'] = np.nanmean(a, 0).tolist(); d[key + '_t_sd'] = np.nanstd(a, 0).tolist()
            d[key + '_avg'] = float(np.nanmean(a[:, 1:])) if np.isfinite(a[:, 1:]).any() else np.nan
            d[key + '_avg_seed'] = np.nanmean(a[:, 1:], 1).tolist()
        old = per_t('old_family_recall')
        forget = np.nanmax(old[:, 1:7], 1) - old[:, 7]
        d['forgetting'] = [float(np.mean(forget)), float(np.std(forget))]
        oldx = per_t('old_family_exact')
        fx = np.nanmax(oldx[:, 1:7], 1) - oldx[:, 7]
        d['forgetting_exact'] = [float(np.mean(fx)), float(np.std(fx))]
        fams = collections.defaultdict(list)
        for r in rs:
            for x in r['hist']:
                for f_, v in x.get('per_family_recall', {}).items():
                    fams[(int(f_), x['t'])].append(v)
        d['family_recall'] = {'%d|%d' % k: float(np.mean(v)) for k, v in fams.items()}
        d['t_fit'] = [float(np.mean([r['t_fit'] for r in rs])), float(np.std([r['t_fit'] for r in rs]))]
        d['seeds'] = sorted(r['seed'] for r in rs)
        st[m] = d
    res['stream'] = st
    # cross-platform
    X = collections.defaultdict(list)
    for f in glob.glob(os.path.join(RUNS, 'xplat', '*.json')):
        r = json.load(open(f)); X[r['method']].append(r)
    res['xplat'] = {m: {p: {k: [float(np.mean([r[p][k] for r in rs])), float(np.std([r[p][k] for r in rs]))] for k in rs[0][p]}
                        for p in ('A', 'B')} for m, rs in X.items()}
    json.dump(res, open(os.path.join(RUNS, 'results.json'), 'w'), indent=1, default=float)
    return res


if __name__ == '__main__':
    r = main()
    print({k: (len(v) if hasattr(v, '__len__') else v) for k, v in r.items()})
    if 'wilcoxon_pr_auc' in r:
        for m, v in r['wilcoxon_pr_auc'].items(): print(m, v)
        print(r['friedman_pr_auc'])
