"""E1/E2/E4/E6/E8 on the four public graphs.
usage: python run_real.py <dataset> <method[,method]> <seeds> [--edge_drop p] [--feat_noise s] [--tag name]"""
import sys, os, json, time, argparse, numpy as np, jax
from graphdata import load_real
from train import Model, ImmuneConformal
import ais, data_real, metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'real')
os.makedirs(OUT, exist_ok=True)

VARIANTS = {   # DIM-GT ablations (component removal)
    'dimgt': ('dimgt', {}),
    'tokce': ('tokce', {}),
    'dimgt-noproto': ('dimgt', dict(use_proto=False)),
    'dimgt-nons': ('dimgt', dict(use_ns=False)),
    'dimgt-nobsc': ('dimgt', dict(use_bsc=False)),
    'dimgt-nola': ('dimgt', dict(use_la=False)),
    'dimgt-norel': ('dimgt', dict(rel_embed=False)),
    'dimgt-self': ('dimgt', dict(self_only=True)),
}


def build(method, seed, cfg_over):
    if method in ('lr', 'rf', 'gbdt'):
        return ais.Classic(method, seed)
    if method == 'nsa':
        return ais.NSA(seed)
    if method == 'nsaga':
        return ais.NSA(seed, ga=True)
    if method in VARIANTS:
        kind, cfg = VARIANTS[method]
    elif method.startswith('dimgt:'):           # sensitivity, e.g. dimgt:m_self=2,lam_ns=0.5
        kind = 'dimgt'; cfg = {}
        for kv in method.split(':', 1)[1].split(','):
            k, v = kv.split('='); cfg[k] = type(Model('dimgt', 2).cfg[k])(float(v)) if k != 'use_la' else bool(int(v))
    else:
        kind, cfg = method, {}
    cfg = dict(cfg); cfg.update(cfg_over)
    return Model(kind, 2, cfg)


def run(ds, method, seed, edge_drop=0.0, feat_noise=0.0, tag='', cfg_over=None):
    fn = os.path.join(OUT, '%s%s_%s_s%d.json' % (ds, tag, method.replace(':', '_').replace('=', '').replace(',', '_'), seed))
    if os.path.exists(fn):
        return json.load(open(fn))
    D = load_real(ds, seed=seed, edge_drop=edge_drop, feat_noise=feat_noise)
    tr, va, ca, te = data_real.split(D.y, seed)
    m = build(method, seed, cfg_over or {})
    t0 = time.time()
    if isinstance(m, Model):
        m.fit(D, tr, va, seed=seed)
    else:
        m.fit(D, tr, va)
    t_train = time.time() - t0
    pv = m.proba(D, va)[0] if isinstance(m, Model) else m.proba(D, va)
    t1 = time.time()
    out = m.proba(D, te)
    t_inf = (time.time() - t1) / len(te) * 1000
    pt = out[0] if isinstance(m, Model) else out
    yv = (D.y[va] > 0).astype(int); yt = (D.y[te] > 0).astype(int)
    sv, st = 1 - pv[:, 0], 1 - pt[:, 0]
    thr = metrics.best_threshold(yv, sv)
    rep = metrics.binary_report(yt, st, thr)
    rep.update(ece=metrics.ece(yt, st), brier=metrics.brier(yt, st), nll=metrics.nll(yt, st))
    res = dict(ds=ds, method=method, seed=seed, edge_drop=edge_drop, feat_noise=feat_noise,
               t_train=t_train, ms_per_1k=t_inf * 1000, n_params=m.n_params(),
               epochs=getattr(m, 'epochs_run', None), **{k: float(v) for k, v in rep.items()})
    extra = {}
    if isinstance(m, Model) and m.kind == 'dimgt' and m.cfg['use_proto']:
        # conformal immune decision on the calibration split
        _, _, ec = m.proba(D, ca); _, _, et = out
        ic = ImmuneConformal(m).calibrate(ec, D.y[ca])
        for a in (0.05, 0.10):
            G, dec, Pv = ic.decide(et, a)
            cov = G[np.arange(len(te)), D.y[te]].mean()
            res['conf_cov_%02d' % int(a * 100)] = float(cov)
            res['conf_size_%02d' % int(a * 100)] = float(G.sum(1).mean())
            res['conf_single_%02d' % int(a * 100)] = float((G.sum(1) == 1).mean())
            sel = dec >= 0
            res['conf_sel_err_%02d' % int(a * 100)] = float((dec[sel] != D.y[te][sel]).mean()) if sel.any() else np.nan
            for c in (0, 1):
                mc = D.y[te] == c
                res['conf_cov_c%d_%02d' % (c, int(a * 100))] = float(G[mc, c].mean())
        extra['pv'] = Pv.astype(np.float32)
    # selective prediction: confidence = max prob
    pred = (st >= thr).astype(int)
    err = (pred != yt)
    conf = np.maximum(st, 1 - st) if not (isinstance(m, Model) and m.kind == 'dimgt' and 'pv' in extra) \
        else extra['pv'].max(1) - np.sort(extra['pv'], 1)[:, -2]
    res['aurc'] = float(metrics.aurc(err, conf))
    for cv in (0.8, 0.9):
        res['risk_at_%d' % int(cv * 100)] = float(metrics.risk_at_coverage(err, conf, cv))
    json.dump(res, open(fn, 'w'))
    np.savez_compressed(fn.replace('.json', '.npz'), te=te, st=st.astype(np.float32), thr=thr, conf=conf.astype(np.float32),
                        **extra)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ds'); ap.add_argument('methods'); ap.add_argument('seeds')
    ap.add_argument('--edge_drop', type=float, default=0.0); ap.add_argument('--feat_noise', type=float, default=0.0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    for meth in a.methods.split(';'):
        for s in seeds:
            t = time.time()
            r = run(a.ds, meth, s, a.edge_drop, a.feat_noise, a.tag)
            print('%-10s %-24s s%d  F1 %.4f  PR %.4f  AUC %.4f  MCC %.4f  ECE %.4f  %.0fs' % (
                a.ds, meth, s, r['macro_f1'], r['pr_auc'], r['auroc'], r['mcc'], r['ece'], time.time() - t), flush=True)
