"""Generic training loop for every learned model + the DIM-GT objective."""
import time, numpy as np, jax, jax.numpy as jnp, optax
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
import nets

jax.config.update('jax_platform_name', 'cpu')

TOKEN_MODELS = ('dimgt', 'tokce', 'mlp', 'cnn', 'lstm', 'nag')
GRAPH_MODELS = ('sage', 'rgcn', 'rgt', 'bwgnn')


class Data:
    """container. T: N x K x d tokens; rel_id/aux_id: K; X: N x d; edges: dict(homo, homo_w, rel)."""
    def __init__(self, T, rel_id, aux_id, y, X=None, edges=None, time_fourier=False, homo_tokens=None):
        self.T = np.asarray(T, np.float32); self.rel_id = np.asarray(rel_id); self.aux_id = np.asarray(aux_id)
        self.y = np.asarray(y); self.X = self.T[:, 0] if X is None else np.asarray(X, np.float32)
        self.edges = edges; self.time_fourier = time_fourier
        self.homo_tokens = homo_tokens          # NAGphormer hop tokens (N x (1+H) x d)
        self.N = len(self.y)


DEFAULT = dict(d=64, layers=2, heads=4, demb=32, drop=0.1, lr=2e-3, wd=1e-5, epochs=120, patience=20,
               batch=1024, m_self=6, m_ns=3, gamma=0.1, kappa=10.0, tau_c=0.1, lam_c=0.3, lam_ns=1.0,
               m_tol=0.75, m_cens=0.35, la_tau=1.0, warm=3,
               use_proto=True, use_bsc=True, use_ns=True, use_la=True, rel_embed=True, self_only=False,
               lam_kd=1.0, kd_T=2.0)


def _class_weights(y, C):
    cnt = np.bincount(y, minlength=C).astype(float) + 1
    w = cnt.sum() / (C * cnt)
    return jnp.asarray(w / w.mean(), jnp.float32)


class Model:
    def __init__(self, kind, n_class, cfg=None):
        self.kind = kind; self.C = n_class
        self.cfg = dict(DEFAULT); self.cfg.update(cfg or {})
        self.params = None; self.owner = None

    # ----------------------------------------------------------- init / forward
    def init(self, data, key):
        c = self.cfg; din = data.T.shape[-1]; K = data.T.shape[1]
        k1, k2 = jax.random.split(key)
        if self.kind in ('dimgt', 'tokce'):
            enc = nets.token_encoder_init(k1, din, int(data.rel_id.max()) + 1, int(data.aux_id.max()) + 1,
                                          c['d'], c['layers'], c['heads'], c['demb'], data.time_fourier,
                                          c['rel_embed'])
            if self.kind == 'dimgt' and c['use_proto']:
                mem, self.owner = nets.memory_init(k2, self.C, c['m_self'], c['m_ns'], c['demb'])
                self.params = dict(enc=enc, mem=mem)
            else:
                self.params = dict(enc=enc, head=nets.dense_init(k2, c['demb'], self.C))
        elif self.kind == 'nag':
            Kh = data.homo_tokens.shape[1]
            enc = nets.token_encoder_init(k1, din, 1, Kh, c['d'], c['layers'], c['heads'], c['demb'])
            self.params = dict(enc=enc, head=nets.dense_init(k2, c['demb'], self.C))
        elif self.kind == 'mlp':
            self.params = nets.mlp_init(k1, din * K, c['d'] * 2, self.C)
        elif self.kind == 'cnn':
            self.params = nets.cnn_init(k1, din, K, c['d'], self.C)
        elif self.kind == 'lstm':
            self.params = nets.lstm_init(k1, din, c['d'], self.C)
        elif self.kind == 'sage':
            self.params = nets.sage_init(k1, data.X.shape[1], c['d'], self.C)
        elif self.kind == 'rgcn':
            self.params = nets.rgcn_init(k1, data.X.shape[1], len(data.edges['rel']), c['d'], self.C)
        elif self.kind == 'rgt':
            p = nets.rgt_init(k1, data.X.shape[1], len(data.edges['rel']), c['d'], 4, self.C)
            self.heads = p.pop('heads'); self.params = p
        elif self.kind == 'bwgnn':
            p = nets.bwgnn_init(k1, data.X.shape[1], c['d'], 2, self.C)
            p.pop('C'); self.params = p
        return self.params

    def _apply_tokens(self, params, Tb, rel_id, aux_id, key, train, tf):
        c = self.cfg
        if self.kind in ('dimgt', 'tokce', 'nag'):
            e, att = nets.token_encoder(params['enc'], Tb, rel_id, aux_id, key, train, c['heads'], c['drop'],
                                        tf if self.kind != 'nag' else False,
                                        c['rel_embed'] if self.kind != 'nag' else False)
            if 'mem' in params:
                logits, cos = nets.memory_logits(params['mem'], e, self.owner, self.C, c['gamma'], c['kappa'])
            else:
                logits = nets.dense(params['head'], e)
            return logits, e
        if self.kind == 'mlp':
            return nets.mlp(params, Tb.reshape(Tb.shape[0], -1), key, train, c['drop'])
        if self.kind == 'cnn':
            return nets.cnn(params, Tb, key, train, c['drop'])
        if self.kind == 'lstm':
            return nets.lstm(params, Tb, key, train, c['drop'])

    def _apply_graph(self, params, X, edges, key, train):
        f = dict(sage=nets.sage, rgcn=nets.rgcn, bwgnn=nets.bwgnn)
        if self.kind == 'rgt':
            p = dict(params); p['heads'] = self.heads
            return nets.rgt(p, X, edges, key, train, self.cfg['drop'])
        if self.kind == 'bwgnn':
            p = dict(params); p['C'] = 2
            return nets.bwgnn(p, X, edges, key, train, self.cfg['drop'])
        return f[self.kind](params, X, edges, key, train, self.cfg['drop'])

    def _tokens(self, data, idx):
        T = data.homo_tokens if self.kind == 'nag' else data.T
        T = T[idx]
        if self.cfg['self_only'] and self.kind in ('dimgt', 'tokce'):
            T = T[:, :1]
        return T

    def _ids(self, data):
        if self.kind == 'nag':
            K = data.homo_tokens.shape[1]
            return jnp.zeros(K, jnp.int32), jnp.arange(K)
        r, a = jnp.asarray(data.rel_id), jnp.asarray(data.aux_id)
        if self.cfg['self_only'] and self.kind in ('dimgt', 'tokce'):
            r, a = r[:1], a[:1]
        return r, a

    # ----------------------------------------------------------- loss
    def _loss(self, params, logits, e, y, cw, prior, old=None):
        c = self.cfg
        if self.kind == 'dimgt':
            adj = logits + c['la_tau'] * jnp.log(prior)[None] if c['use_la'] else logits
            ce = optax.softmax_cross_entropy_with_integer_labels(adj, y)
            loss = ce.mean() if c['use_la'] else (ce * cw[y]).mean()
            if 'mem' in params:
                if c['use_bsc']:
                    loss = loss + c['lam_c'] * nets.balanced_supcon(e, y, params['mem']['P'], self.owner, self.C, c['tau_c'])
                if c['use_ns']:
                    loss = loss + c['lam_ns'] * nets.negative_selection_loss(e, y, params['mem'], self.owner,
                                                                              c['m_tol'], c['m_cens'])
            elif c['use_bsc']:
                loss = loss + c['lam_c'] * nets.balanced_supcon(e, y, jnp.zeros((0, e.shape[1])), np.zeros(0, int), self.C, c['tau_c'])
        else:
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            loss = (ce * cw[y]).mean()
        if old is not None:      # knowledge distillation on the old classes
            ol, mask = old
            Cold = ol.shape[1]
            T = c['kd_T']
            p_old = jax.nn.softmax(ol / T, -1)
            lp_new = jax.nn.log_softmax(logits[:, :Cold] / T, -1)
            kd = -(p_old * lp_new).sum(-1)
            loss = loss + c['lam_kd'] * (kd * mask).sum() / jnp.maximum(mask.sum(), 1) * T * T
        return loss

    # ----------------------------------------------------------- forward on index set
    def forward(self, data, idx, params=None, bs=4096):
        params = self.params if params is None else params
        if self.kind in GRAPH_MODELS:
            if not hasattr(data, '_jedges'):
                data._jedges = jax.tree_util.tree_map(jnp.asarray, data.edges)
            if not hasattr(self, '_gfwd'):
                self._gfwd = jax.jit(lambda p, X, E: self._apply_graph(p, X, E, None, False))
            lg, e = self._gfwd(params, jnp.asarray(data.X), data._jedges)
            return np.asarray(lg)[idx], np.asarray(e)[idx]
        if not hasattr(self, '_tfwd'):
            r, a = self._ids(data)
            tf = data.time_fourier
            self._tfwd = jax.jit(lambda p, Tb: self._apply_tokens(p, Tb, r, a, None, False, tf))
        L, E = [], []
        for s in range(0, len(idx), bs):
            Tb = jnp.asarray(self._tokens(data, idx[s:s + bs]))
            lg, e = self._tfwd(params, Tb)
            L.append(np.asarray(lg)); E.append(np.asarray(e))
        return np.concatenate(L), np.concatenate(E)

    # ----------------------------------------------------------- prototype seeding (clonal initialisation)
    def seed_prototypes(self, data, idx, y, classes=None, params=None):
        params = self.params if params is None else params
        _, e = self.forward(data, idx, params)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        P = np.array(params['mem']['P'])
        for cl in (range(self.C) if classes is None else classes):
            rows = np.where(self.owner == cl)[0]
            ec = e[y == cl]
            if len(ec) == 0: continue
            k = min(len(rows), len(ec))
            km = KMeans(k, n_init=3, random_state=0).fit(ec)
            cen = km.cluster_centers_
            cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-8)
            P[rows[:k]] = cen
        params['mem']['P'] = jnp.asarray(P)
        return params

    # ----------------------------------------------------------- fit
    def fit(self, data, tr, va, seed=0, epochs=None, verbose=False, old_model=None, prior=None,
            init=True, select='prauc', reseed=True, ewc=None, lr=None, enc_scale=None):
        c = self.cfg
        epochs = epochs or c['epochs']
        key = jax.random.PRNGKey(seed)
        if init or self.params is None:
            self.init(data, key)
        y_all = data.y
        ytr = y_all[tr]
        cw = _class_weights(ytr, self.C)
        pri = np.bincount(ytr, minlength=self.C).astype(float) + 1
        prior = jnp.asarray(pri / pri.sum(), jnp.float32) if prior is None else jnp.asarray(prior)
        base_lr = lr or c['lr']
        if enc_scale is not None and isinstance(self.params, dict) and 'enc' in self.params:
            labels = {k: ('enc' if k == 'enc' else 'rest') for k in self.params}
            enc_opt = optax.set_to_zero() if enc_scale == 0 else optax.adamw(base_lr * enc_scale, weight_decay=c['wd'])
            opt = optax.multi_transform({'enc': enc_opt, 'rest': optax.adamw(base_lr, weight_decay=c['wd'])}, labels)
        else:
            opt = optax.adamw(base_lr, weight_decay=c['wd'])
        def ewc_pen(p):
            if ewc is None: return 0.0
            F, old, lam = ewc
            return lam * sum(jnp.sum(f * (a - b) ** 2) for f, a, b in zip(jax.tree_util.tree_leaves(F),
                             jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(old)))
        if self.kind in GRAPH_MODELS:
            data._jedges = jax.tree_util.tree_map(jnp.asarray, data.edges)
        params = self.params
        state = opt.init(params)
        # old-model logits for distillation (computed once)
        old_logits = None
        if old_model is not None:
            ol, _ = old_model.forward(data, tr)
            old_logits = jnp.asarray(ol)
        r, a = self._ids(data) if self.kind in TOKEN_MODELS else (None, None)
        tf = data.time_fourier

        if self.kind in TOKEN_MODELS:
            def step(params, state, Tb, yb, key, olb, omask):
                def lf(p):
                    lg, e = self._apply_tokens(p, Tb, r, a, key, True, tf)
                    return self._loss(p, lg, e, yb, cw, prior, None if olb is None else (olb, omask)) + ewc_pen(p)
                l, g = jax.value_and_grad(lf)(params)
                upd, state = opt.update(g, state, params)
                return optax.apply_updates(params, upd), state, l
            step = jax.jit(step)
        else:
            Xj = jnp.asarray(data.X); E = data._jedges
            trj = jnp.asarray(tr); ytrj = jnp.asarray(ytr)
            def gstep(params, state, key, olb, omask, Xj, E, trj, ytrj):
                def lf(p):
                    lg, e = self._apply_graph(p, Xj, E, key, True)
                    return self._loss(p, lg[trj], e[trj], ytrj, cw, prior, None if olb is None else (olb, omask)) + ewc_pen(p)
                l, g = jax.value_and_grad(lf)(params)
                upd, state = opt.update(g, state, params)
                return optax.apply_updates(params, upd), state, l
            gstep = jax.jit(gstep)

        best, best_p, bad = -np.inf, params, 0
        rng = np.random.default_rng(seed)
        B = c['batch']
        hist = []
        for ep in range(epochs):
            key, sk = jax.random.split(key)
            if self.kind in TOKEN_MODELS:
                perm = rng.permutation(len(tr))
                if len(perm) < B:
                    perm = np.r_[perm, rng.choice(len(tr), B - len(perm))]
                nb = max(1, len(perm) // B)
                for bi in range(nb):
                    sel = perm[bi * B:(bi + 1) * B]
                    sk, kk = jax.random.split(sk)
                    Tb = jnp.asarray(self._tokens(data, tr[sel]))
                    yb = jnp.asarray(ytr[sel])
                    if old_logits is not None:
                        olb = old_logits[sel]; om = jnp.ones(len(sel))
                    else:
                        olb, om = None, None
                    params, state, l = step(params, state, Tb, yb, kk, olb, om)
            else:
                om = None if old_logits is None else jnp.ones(len(tr))
                params, state, l = gstep(params, state, sk, old_logits, om, Xj, E, trj, ytrj)
            if self.kind == 'dimgt' and 'mem' in params and ep == c['warm'] - 1 and reseed and init:
                params = self.seed_prototypes(data, tr, ytr, params=params)
                state = opt.init(params)
            # validation
            if va is not None and len(va):
                lg, _ = self.forward(data, va, params)
                p = np.asarray(jax.nn.softmax(lg, -1))
                yv = (y_all[va] > 0).astype(int)
                s = 1 - p[:, 0]
                score = average_precision_score(yv, s) if 0 < yv.mean() < 1 else -float(l)
                hist.append(score)
                if score > best + 1e-4:
                    best, best_p, bad = score, jax.tree_util.tree_map(lambda x: x, params), 0
                else:
                    bad += 1
                if verbose and ep % 10 == 0:
                    print(ep, float(l), score, flush=True)
                if bad >= c['patience'] and ep >= c['warm'] + 5:
                    break
            else:
                best_p = params
        self.params = best_p
        self.best_val = best; self.epochs_run = ep + 1
        if hasattr(self, '_tfwd'): del self._tfwd
        if hasattr(self, '_gfwd'): del self._gfwd
        return self

    def fisher(self, data, idx, n_batches=16, bs=32, seed=0):
        """empirical diagonal Fisher on mini-batches (token models)."""
        r, a = self._ids(data); tf = data.time_fourier
        def ll(p, Tb, yb):
            lg, _ = self._apply_tokens(p, Tb, r, a, None, False, tf)
            return optax.softmax_cross_entropy_with_integer_labels(lg, yb).mean()
        g = jax.jit(jax.grad(ll))
        rng = np.random.default_rng(seed)
        F = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        for _ in range(n_batches):
            sel = rng.choice(idx, min(bs, len(idx)), replace=False)
            gr = g(self.params, jnp.asarray(self._tokens(data, sel)), jnp.asarray(data.y[sel]))
            F = jax.tree_util.tree_map(lambda f, x: f + x ** 2 / n_batches, F, gr)
        return F

    def proba(self, data, idx):
        lg, e = self.forward(data, idx)
        return np.asarray(jax.nn.softmax(lg, -1)), lg, e

    def n_params(self):
        return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(self.params)))


# =============================================================== immune conformal decision layer
class RawSelf:
    """input-space self tolerance: mean distance to the k nearest legitimate training
    accounts in a PCA projection of the flattened token set (classical negative-selection view)."""
    def __init__(self, data, idx_self, k=10, dim=16, max_ref=5000, seed=0):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.default_rng(seed)
        if len(idx_self) > max_ref: idx_self = rng.choice(idx_self, max_ref, replace=False)
        Z = data.T[idx_self].reshape(len(idx_self), -1)
        self.pca = PCA(min(dim, Z.shape[1]), random_state=seed).fit(Z)
        self.nn = NearestNeighbors(n_neighbors=k).fit(self.pca.transform(Z))
        self.data = data

    def score(self, idx, bs=4096):
        out = []
        for s in range(0, len(idx), bs):
            Z = self.pca.transform(self.data.T[idx[s:s + bs]].reshape(len(idx[s:s + bs]), -1))
            out.append(self.nn.kneighbors(Z)[0].mean(1))
        return np.concatenate(out)


class ImmuneConformal:
    """class-conditional (Mondrian) conformal p-values from memory affinities.
    nonconformity s_c(e) = 1 - max_j cos(e, mu_cj); for self, optionally the larger of the
    calibrated ranks of the embedding distance and the input-space self distance."""
    def __init__(self, model, raw=None):
        self.m = model; self.raw = raw

    def scores(self, e):
        P = np.asarray(self.m.params['mem']['P'])
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        cos = e @ P.T
        S = np.stack([1 - cos[:, self.m.owner == c].max(1) for c in range(self.m.C)], 1)
        return S

    def _self_score(self, S0, r0):
        if r0 is None or not hasattr(self, 'ref_emb'):
            return S0
        fe = np.searchsorted(self.ref_emb, S0, side='right') / len(self.ref_emb)
        fr = np.searchsorted(self.ref_raw, r0, side='right') / len(self.ref_raw)
        return np.maximum(fe, fr)

    def calibrate(self, e_cal, y_cal, raw_cal=None):
        """two-level Mondrian taxonomy: self (class 0) is calibrated on its own scores; all
        confirmed non-self classes share one pooled list of own-class scores, so newly added
        families with few calibration nodes still receive informative p-values."""
        S = self.scores(e_cal)
        if raw_cal is not None:
            self.ref_emb = np.sort(S[y_cal == 0, 0]); self.ref_raw = np.sort(raw_cal[y_cal == 0])
            self.cal = {0: np.sort(self._self_score(S[y_cal == 0, 0], raw_cal[y_cal == 0]))}
        else:
            self.cal = {0: np.sort(S[y_cal == 0, 0])}
        ns = y_cal > 0
        self.cal_ns = np.sort(S[ns, y_cal[ns]]) if ns.any() else np.zeros(0)
        self.known = set(int(c) for c in np.unique(y_cal))
        return self

    def pvalues(self, e, raw=None):
        S = self.scores(e)
        if raw is not None and hasattr(self, 'ref_emb'):
            S[:, 0] = self._self_score(S[:, 0], raw)
        Pv = np.zeros_like(S)
        for c in range(self.m.C):
            cs = self.cal[0] if c == 0 else self.cal_ns
            if len(cs) == 0 or (c > 0 and c not in self.known and len(self.known) > 1 and False):
                Pv[:, c] = 0; continue
            n = len(cs)
            # p = (1 + #{cal >= s}) / (n + 1)
            Pv[:, c] = (1 + n - np.searchsorted(cs, S[:, c], side='left')) / (n + 1)
        return Pv

    def decide(self, e, alpha=0.05, raw=None):
        """returns (prediction set mask, decision) where decision: class id, -1 novel, -2 abstain."""
        Pv = self.pvalues(e, raw)
        G = Pv > alpha
        k = G.sum(1)
        dec = np.where(k == 0, -1, np.where(k == 1, G.argmax(1), -2))
        return G, dec, Pv
