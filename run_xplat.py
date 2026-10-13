"""Cross-platform transfer (E10): train on SpamSim-OW platform A (windows 0-2),
evaluate without adaptation on platform B (windows 0-2, families 0-3), whose
persona mix, activity, URL and hashtag rates differ.  Feature standardisation
statistics come from platform A only."""
import os, sys, json, time, numpy as np, jax
from sklearn.metrics import f1_score, average_precision_score
from run_stream import World, make_data, Labels
from train import Model, ImmuneConformal
import ais, metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'xplat')
os.makedirs(OUT, exist_ok=True)


def run(seed, method):
    fn = os.path.join(OUT, '%s_s%d.json' % (method, seed))
    if os.path.exists(fn): return json.load(open(fn))
    A = World(seed, 'A'); B = World(seed, 'B', stats=A.stats)
    rng = np.random.default_rng(100 + seed)
    users = rng.permutation(A.N); pool = users[:int(.6 * A.N)]
    n = len(pool); tr_u, va_u, ca_u = pool[:int(n * 2 / 3)], pool[int(n * 2 / 3):int(n * 5 / 6)], pool[int(n * 5 / 6):]
    nodes = lambda W, us, ts: (lambda o: o[W.actflat[o]])(np.concatenate([W.idx(t, us) for t in ts]))
    L = Labels()
    DA = make_data(A, np.maximum(L.cls(A.yflat), 0)); DA.ylab = L.cls(A.yflat)
    tr, va, ca = nodes(A, tr_u, range(3)), nodes(A, va_u, range(3)), nodes(A, ca_u, range(3))
    DB = make_data(B, np.maximum(L.cls(B.yflat), 0)); DB.ylab = L.cls(B.yflat)
    ub = np.random.default_rng(200 + seed).permutation(B.N)[:int(.4 * B.N)]
    res = dict(method=method, seed=seed)
    if method in ('gbdt', 'nsaga'):
        m = ais.Classic('gbdt', seed, fast=True) if method == 'gbdt' else ais.NSA(seed, ga=True)
        m.fit(DA, tr, va)
        prob = lambda D, idx: m.proba(D, idx)
    else:
        m = Model(method, 4, dict(epochs=80, patience=12)).fit(DA, tr, va, seed=seed)
        prob = lambda D, idx: m.proba(D, idx)[0]
    for name, D, ev in (('A', DA, nodes(A, users[int(.6 * A.N):], range(3))), ('B', DB, nodes(B, ub, range(3)))):
        p = prob(D, ev); y = (D.ylab[ev] > 0).astype(int)
        s = 1 - p[:, 0]; pred = (p.argmax(1) > 0).astype(int)
        res[name] = dict(macro_f1=f1_score(y, pred, average='macro'), pr_auc=average_precision_score(y, s),
                         ece=metrics.ece(y, s), legit_fpr=float(pred[y == 0].mean()), tpr=float(pred[y == 1].mean()))
        if method == 'dimgt':
            _, _, ec = m.proba(DA, ca)
            ic = ImmuneConformal(m).calibrate(ec, DA.ylab[ca])
            _, _, e = m.proba(D, ev)
            G, dec, Pv = ic.decide(e, 0.05)
            res[name]['abstain'] = float((dec == -2).mean()); res[name]['novel_flag'] = float((dec == -1).mean())
            acc = dec >= 0
            res[name]['accepted_err'] = float(((dec[acc] > 0).astype(int) != y[acc]).mean())
            res[name]['cov_legit'] = float(G[y == 0, 0].mean())
    json.dump(res, open(fn, 'w'), default=float)
    return res


if __name__ == '__main__':
    for meth in sys.argv[1].split(';'):
        for s in [int(x) for x in sys.argv[2].split(',')]:
            t = time.time(); r = run(s, meth)
            print(meth, s, 'A F1 %.3f PR %.3f | B F1 %.3f PR %.3f FPR %.3f ECE %.3f  %.0fs' % (
                r['A']['macro_f1'], r['A']['pr_auc'], r['B']['macro_f1'], r['B']['pr_auc'], r['B']['legit_fpr'], r['B']['ece'], time.time() - t), flush=True)
