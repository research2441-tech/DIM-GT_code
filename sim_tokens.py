"""Turn SpamSim-OW windows into temporal relation tokens and cache them."""
import numpy as np, scipy.sparse as sp, os, spamsim

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sim')
W_HIST = 3


def window_tokens(windows, t, N):
    """tokens for window t: for tau in t-2..t, [x^tau, mean_r x^tau] for 6 relations."""
    toks, rel, tim = [], [], []
    for lag in range(W_HIST - 1, -1, -1):
        tau = max(0, t - lag)
        w = windows[tau]
        X = w['X']
        degs, aggs = [], []
        for r in spamsim.RELS:
            s, d = w['rels'][r]
            A = sp.coo_matrix((np.ones(len(s), np.float32), (s, d)), shape=(N, N)).tocsr()
            A = A + A.T; A.data[:] = 1
            deg = np.asarray(A.sum(1)).ravel()
            degs.append(np.log1p(deg))
            P = sp.diags(1 / np.maximum(deg, 1)) @ A
            aggs.append(P @ X)
        D = np.stack(degs, 1).astype(np.float32)
        Xa = np.concatenate([X, D], 1)
        toks.append(Xa); rel.append(0); tim.append(lag)
        for r, a in enumerate(aggs):
            toks.append(np.concatenate([a, np.zeros_like(D)], 1).astype(np.float32)); rel.append(r + 1); tim.append(lag)
    return np.stack(toks, 1), np.array(rel), np.array(tim)


def build(seed, preset='A'):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, 'sim_%s_%d.npz' % (preset, seed))
    if os.path.exists(fn):
        return fn
    W, M = spamsim.simulate(seed, preset=preset)
    N = M['N']
    T, Y, ACT = [], [], []
    adj = {}
    for t in range(len(W)):
        tk, rel, tim = window_tokens(W, t, N)
        T.append(tk.astype(np.float16)); Y.append(W[t]['y']); ACT.append(W[t]['active'])
        for r in spamsim.RELS:
            s, d = W[t]['rels'][r]
            adj['t%d_%s' % (t, r)] = np.stack([s, d]).astype(np.int32)
    np.savez_compressed(fn, T=np.stack(T), y=np.stack(Y), active=np.stack(ACT), rel_id=rel, time_id=tim,
                        takeover=M['takeover'], family=M['family'], **adj)
    return fn


if __name__ == '__main__':
    import sys, time
    preset = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]]
    for s in seeds:
        t = time.time(); print(build(s, preset), '%.0fs' % (time.time() - t), flush=True)
