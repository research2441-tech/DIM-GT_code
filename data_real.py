"""Real public abuse graphs -> relation-hop token tensors.

YelpChi (review spam, 3 relations), Amazon (fraudulent users, 3 relations),
Weibo (suspicious users, 1 relation), Reddit (banned users, 1 relation).
Each node v receives tokens [x_v, A_r x, A_r^2 x for every relation r].
"""
import numpy as np, scipy.io as sio, scipy.sparse as sp, os, ptload

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _rownorm(A):
    A = sp.csr_matrix(A, dtype=np.float32)
    A.setdiag(0); A.eliminate_zeros()
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return sp.diags(1.0 / d) @ A


def _standardise(x):
    x = np.asarray(x, dtype=np.float64)
    # heavy-tailed non-negative columns -> log1p
    for j in range(x.shape[1]):
        c = x[:, j]
        if c.min() >= 0 and np.percentile(c, 99) > 20 * (np.median(c) + 1e-9) and c.max() > 10:
            x[:, j] = np.log1p(c)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    return ((x - mu) / sd).clip(-8, 8).astype(np.float32)


def load(name, hops=2):
    if name in ('yelpchi', 'amazon'):
        m = sio.loadmat(os.path.join(DATA, 'YelpChi.mat' if name == 'yelpchi' else 'Amazon.mat'))
        rels = ['net_rur', 'net_rtr', 'net_rsr'] if name == 'yelpchi' else ['net_upu', 'net_usu', 'net_uvu']
        x = m['features'].toarray(); y = m['label'].ravel().astype(int)
        As = [m[r] for r in rels]
        homo = m['homo']
        keep = np.arange(len(y))
        if name == 'amazon':          # first 3305 users are unlabelled in the CARE-GNN release
            keep = np.arange(3305, len(y))
        relnames = [r.replace('net_', '').upper() for r in rels]
    else:
        o = ptload.load(os.path.join(DATA, name + '.pt'))
        mp = o.__dict__['_store'].__dict__['_mapping']
        x = mp['x']; y = mp['y'].astype(int); ei = mp['edge_index']
        n = len(y)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.float32)
        As = [A]; homo = A; keep = np.arange(n); relnames = ['U-U']
    xs = _standardise(x)
    toks = [xs]
    Ps = [_rownorm(A) for A in As]
    for P in Ps:
        h = xs
        for k in range(hops):
            h = P @ h
            toks.append(np.asarray(h, dtype=np.float32))
    T = np.stack(toks, 1)[keep]                         # N x (1+R*hops) x d
    # metadata for each token: relation id (0=self) and hop id
    rel_id = [0] + [r + 1 for r in range(len(Ps)) for _ in range(hops)]
    hop_id = [0] + [k + 1 for _ in range(len(Ps)) for k in range(hops)]
    # restricted adjacency for message-passing baselines
    adj = [sp.csr_matrix(A)[keep][:, keep] for A in As]
    homo = sp.csr_matrix(homo)[keep][:, keep]
    return dict(name=name, X=xs[keep], T=T, y=y[keep], rel_id=np.array(rel_id), hop_id=np.array(hop_id),
                adj=adj, homo=homo, relnames=relnames, raw=np.asarray(x)[keep])


def split(y, seed, frac=(0.4, 0.2, 0.4)):
    """Stratified 40/20/40 split; validation further halved into model-selection
    (val) and conformal calibration (cal) parts."""
    rng = np.random.default_rng(seed)
    tr, va, ca, te = [], [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); n = len(idx)
        a = int(frac[0] * n); b = int((frac[0] + frac[1]) * n); h = (a + b) // 2
        tr += list(idx[:a]); va += list(idx[a:h]); ca += list(idx[h:b]); te += list(idx[b:])
    return [np.array(sorted(s)) for s in (tr, va, ca, te)]


if __name__ == '__main__':
    import time
    for nm in ['yelpchi', 'amazon', 'weibo', 'reddit']:
        t = time.time(); D = load(nm)
        e = sum(a.nnz for a in D['adj'])
        print(nm, D['T'].shape, 'pos=%d (%.2f%%)' % (D['y'].sum(), 100 * D['y'].mean()), 'edges', e,
              D['relnames'], '%.1fs' % (time.time() - t))
        np.savez_compressed(os.path.join(DATA, nm + '_tok.npz'), T=D['T'], y=D['y'], X=D['X'], raw=D['raw'],
                            rel_id=D['rel_id'], hop_id=D['hop_id'])
        sp.save_npz(os.path.join(DATA, nm + '_homo.npz'), D['homo'].tocsr())
        for i, a in enumerate(D['adj']):
            sp.save_npz(os.path.join(DATA, nm + '_adj%d.npz' % i), a.tocsr())
