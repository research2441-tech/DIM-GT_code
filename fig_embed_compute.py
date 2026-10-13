"""Interpretability figure: (a) 2-D PCA of DIM-GT embeddings on SpamSim-OW window 7 with
immune memory prototypes; (b) mean token-pooling weights by relation and window lag per family."""
import os, json, numpy as np, jax, jax.numpy as jnp
import matplotlib
BASE = os.path.dirname(os.path.abspath(__file__)) + '/'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_stream import World, make_data, Labels
from train import Model
import nets
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif', 'font.size': 6.5,
                     'axes.linewidth': 0.6})
FIG = BASE + 'figs'
seed = 0
W = World(seed, 'A')
rng = np.random.default_rng(100 + seed)
users = rng.permutation(W.N); pool = users[:int(.6 * W.N)]
n = len(pool); tr_u, va_u = pool[:int(n * 2 / 3)], pool[int(n * 2 / 3):int(n * 5 / 6)]
nodes = lambda us, ts: (lambda o: o[W.actflat[o]])(np.concatenate([W.idx(t, us) for t in ts]))
L = Labels()
D = make_data(W, np.maximum(L.cls(W.yflat), 0)); D.ylab = L.cls(W.yflat)
tr, va = nodes(tr_u, range(3)), nodes(va_u, range(3))
m = Model('dimgt', 4, dict(epochs=80, patience=12)).fit(D, tr, va, seed=seed)
ev = W.idx(7, users[int(.6 * W.N):]); ev = ev[W.actflat[ev]]
fam = W.yflat[ev]
r, a = m._ids(D)
T = jnp.asarray(D.T[ev])
e, att = nets.token_encoder(m.params['enc'], T, r, a, None, False, 4, 0.1, True, True)
e = np.asarray(e); att = np.asarray(att)
e = e / np.linalg.norm(e, axis=1, keepdims=True)
P = np.asarray(m.params['mem']['P']); P = P / np.linalg.norm(P, axis=1, keepdims=True)
mu = e.mean(0); U, S, Vt = np.linalg.svd(e - mu, full_matrices=False)
proj = lambda x: (x - mu) @ Vt[:2].T
Z = proj(e); ZP = proj(P)
np.savez(BASE + 'runs/embed_cache.npz', Z=Z, ZP=ZP, owner=m.owner, fam=fam, att=att, rid=np.asarray(D.rel_id), tid=np.asarray(D.aux_id))

