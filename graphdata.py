import numpy as np, scipy.sparse as sp, os
from train import Data
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def cap_edges(A, cap, rng):
    """A: csr (row = destination). keep at most `cap` random in-neighbours per node."""
    A = sp.csr_matrix(A); A.setdiag(0); A.eliminate_zeros()
    src, dst = [], []
    indptr, ind = A.indptr, A.indices
    for v in range(A.shape[0]):
        nb = ind[indptr[v]:indptr[v + 1]]
        if len(nb) > cap: nb = rng.choice(nb, cap, replace=False)
        src.append(nb); dst.append(np.full(len(nb), v))
    return np.concatenate(src).astype(np.int32), np.concatenate(dst).astype(np.int32)


def sym_weights(src, dst, n):
    deg = np.bincount(dst, minlength=n).astype(np.float32)
    deg_s = np.bincount(src, minlength=n).astype(np.float32)
    return (1 / np.sqrt(np.maximum(deg[dst], 1) * np.maximum(deg_s[src], 1))).astype(np.float32)


def hop_tokens(A, X, H=3):
    A = sp.csr_matrix(A, dtype=np.float32); A.setdiag(0); A.eliminate_zeros(); A.data[:] = 1
    A = ((A + A.T) > 0).astype(np.float32)
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    Dm = sp.diags(1 / np.sqrt(d)); P = Dm @ A @ Dm
    out = [X]; h = X
    for _ in range(H):
        h = P @ h; out.append(np.asarray(h, np.float32))
    return np.stack(out, 1)


def load_real(name, cap=32, seed=0, edge_drop=0.0, feat_noise=0.0):
    z = np.load(os.path.join(DATA, name + '_tok.npz'))
    T, y, X = z['T'].copy(), z['y'], z['X'].copy()
    rel_id, hop_id = z['rel_id'], z['hop_id']
    n = len(y)
    rng = np.random.default_rng(1000 + seed)
    R = len([f for f in os.listdir(DATA) if f.startswith(name + '_adj')])
    adjs = [sp.load_npz(os.path.join(DATA, '%s_adj%d.npz' % (name, r))) for r in range(R)]
    homo = sp.load_npz(os.path.join(DATA, name + '_homo.npz'))
    if edge_drop > 0 or feat_noise > 0:
        # perturb structure / features and rebuild tokens consistently for all models
        def drop(A):
            A = sp.coo_matrix(A); keep = rng.random(A.nnz) >= edge_drop
            return sp.csr_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape)
        if edge_drop > 0:
            adjs = [drop(A) for A in adjs]; homo = drop(homo)
        if feat_noise > 0:
            X = X + rng.normal(0, feat_noise, X.shape).astype(np.float32)
        toks = [X]
        hops = int(hop_id.max())
        for A in adjs:
            A = sp.csr_matrix(A, dtype=np.float32); A.setdiag(0); A.eliminate_zeros(); A.data[:] = 1
            dd = np.asarray(A.sum(1)).ravel(); dd[dd == 0] = 1
            P = sp.diags(1 / dd) @ A; h = X
            for k in range(hops):
                h = P @ h; toks.append(np.asarray(h, np.float32))
        T = np.stack(toks, 1)
    rel = [cap_edges(A, cap, rng) for A in adjs]
    hs, hd = cap_edges(homo, cap, rng)
    edges = dict(homo=(hs, hd), homo_w=sym_weights(hs, hd, n), rel=rel)
    D = Data(T, rel_id, hop_id, y, X=X, edges=edges, homo_tokens=hop_tokens(homo, X, 3))
    D.name = name; D.raw = z['raw']
    return D
