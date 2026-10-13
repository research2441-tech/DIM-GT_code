"""Warm-start inference latency per 1,000 nodes (median of 5 timed passes after warm-up)."""
import time, json, os, sys, numpy as np, jax
from graphdata import load_real
from train import Model, ImmuneConformal
import ais, data_real

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'bench.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in sys.argv[1].split(','):
    D = load_real(ds, seed=0)
    tr, va, ca, te = data_real.split(D.y, 0)
    for m in ['gbdt', 'rf', 'nsaga', 'mlp', 'cnn', 'lstm', 'sage', 'rgcn', 'bwgnn', 'rgt', 'nag', 'tokce', 'dimgt']:
        key = '%s|%s' % (ds, m)
        if key in res: continue
        if m in ('gbdt', 'rf'):
            mod = ais.Classic(m, 0).fit(D, tr, va); f = lambda: mod.proba(D, te)
        elif m == 'nsaga':
            mod = ais.NSA(0, ga=True).fit(D, tr, va); f = lambda: mod.proba(D, te)
        else:
            mod = Model(m, 2, dict(epochs=1)).fit(D, tr, va, seed=0)
            if m == 'dimgt':
                _, _, ec = mod.proba(D, ca); ic = ImmuneConformal(mod).calibrate(ec, D.y[ca])
                def f():
                    _, _, e = mod.proba(D, te); return ic.decide(e, 0.05)
            else:
                f = lambda: mod.proba(D, te)
        f(); f()
        ts = []
        for _ in range(5):
            t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
        res[key] = dict(ms_per_1k=float(np.median(ts) * 1e6 / len(te)),
                        n_test=int(len(te)))
        print(key, '%.2f ms / 1k nodes' % res[key]['ms_per_1k'], flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)
