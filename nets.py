"""JAX building blocks: DIM-GT (relational-temporal token transformer + immune
prototype memory) and the learned baselines (MLP, 1D-CNN, LSTM, NAGphormer-style
hop transformer, GraphSAGE, RGCN, RGT-style relational attention, BWGNN)."""
import jax, jax.numpy as jnp, numpy as np, optax
from functools import partial

# ------------------------------------------------------------------ helpers

def glorot(key, shape):
    lim = np.sqrt(6.0 / (shape[0] + shape[-1]))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def dense_init(key, din, dout):
    return dict(w=glorot(key, (din, dout)), b=jnp.zeros(dout))


def dense(p, x):
    return x @ p['w'] + p['b']


def ln_init(d):
    return dict(g=jnp.ones(d), b=jnp.zeros(d))


def ln(p, x, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * p['g'] + p['b']


def dropout(key, x, rate, train):
    if not train or rate == 0 or key is None:
        return x
    keep = jax.random.bernoulli(key, 1 - rate, x.shape)
    return jnp.where(keep, x / (1 - rate), 0)


def l2n(x, eps=1e-8):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

# ------------------------------------------------------------------ token transformer encoder

def tf_block_init(key, d, h, f):
    k = jax.random.split(key, 6)
    return dict(ln1=ln_init(d), ln2=ln_init(d), q=dense_init(k[0], d, d), k=dense_init(k[1], d, d),
                v=dense_init(k[2], d, d), o=dense_init(k[3], d, d), f1=dense_init(k[4], d, f), f2=dense_init(k[5], f, d))


def tf_block(p, z, heads, key, train, drop, bias=None):
    B, K, d = z.shape
    hd = d // heads
    y = ln(p['ln1'], z)
    q = dense(p['q'], y).reshape(B, K, heads, hd)
    k = dense(p['k'], y).reshape(B, K, heads, hd)
    v = dense(p['v'], y).reshape(B, K, heads, hd)
    s = jnp.einsum('bqhd,bkhd->bhqk', q, k) / np.sqrt(hd)
    if bias is not None:
        s = s + bias
    a = jax.nn.softmax(s, -1)
    o = jnp.einsum('bhqk,bkhd->bqhd', a, v).reshape(B, K, d)
    k1, k2 = (jax.random.split(key) if key is not None else (None, None))
    z = z + dropout(k1, dense(p['o'], o), drop, train)
    y = ln(p['ln2'], z)
    z = z + dropout(k2, dense(p['f2'], jax.nn.gelu(dense(p['f1'], y))), drop, train)
    return z, a


def token_encoder_init(key, din, n_rel, n_aux, d=64, layers=2, heads=4, demb=32, time_fourier=False,
                       rel_embed=True):
    ks = jax.random.split(key, 8 + layers)
    p = dict(inp=dense_init(ks[0], din, d), rel=0.02 * jax.random.normal(ks[1], (n_rel, d)),
             aux=0.02 * jax.random.normal(ks[2], (n_aux, d)),
             blocks=[tf_block_init(ks[8 + i], d, heads, 2 * d) for i in range(layers)],
             lnf=ln_init(d), pool_q=0.02 * jax.random.normal(ks[3], (d,)),
             out1=dense_init(ks[4], 2 * d, d), out2=dense_init(ks[5], d, demb))
    if time_fourier:
        p['omega'] = jnp.linspace(0.2, 3.0, d // 2)
        p['tproj'] = dense_init(ks[6], d, d)
    if not rel_embed:
        p['rel'] = jnp.zeros((n_rel, d))
    return p


def token_encoder(p, T, rel_id, aux_id, key, train, heads=4, drop=0.1, time_fourier=False, rel_embed=True):
    """T: B x K x din  ->  (embedding B x demb, attention of the pooled read-out B x K)."""
    z = dense(p['inp'], T)
    if rel_embed:
        z = z + p['rel'][rel_id][None]
    if time_fourier:      # Fourier temporal encoding of the window lag
        ang = aux_id[:, None].astype(jnp.float32) * p['omega'][None]
        phi = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], -1)
        z = z + dense(p['tproj'], phi)[None]
    else:
        z = z + p['aux'][aux_id][None]
    keys = jax.random.split(key, len(p['blocks']) + 1) if key is not None else [None] * (len(p['blocks']) + 1)
    for i, b in enumerate(p['blocks']):
        z, _ = tf_block(b, z, heads, keys[i], train, drop)
    z = ln(p['lnf'], z)
    self_tok = z[:, 0]
    s = jnp.einsum('bkd,d->bk', z, p['pool_q'] + 0.0) / np.sqrt(z.shape[-1]) + \
        jnp.einsum('bkd,bd->bk', z, self_tok) / np.sqrt(z.shape[-1])
    att = jax.nn.softmax(s, -1)
    pooled = jnp.einsum('bk,bkd->bd', att, z)
    h = jax.nn.gelu(dense(p['out1'], jnp.concatenate([self_tok, pooled], -1)))
    h = dropout(keys[-1], h, drop, train)
    return dense(p['out2'], h), att

# ------------------------------------------------------------------ immune prototype memory head

def memory_init(key, n_class, m_self, m_ns, demb):
    """prototype bank: class 0 (self) gets m_self prototypes, every non-self class m_ns."""
    counts = [m_self] + [m_ns] * (n_class - 1)
    owner = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    P = l2n(jax.random.normal(key, (len(owner), demb)))
    return dict(P=P), owner


def memory_logits(mem, e, owner, n_class, gamma=0.1, kappa=10.0):
    """class logit = kappa * gamma * logsumexp_j(cos(e, mu_cj)/gamma)  (soft nearest memory)."""
    cos = l2n(e) @ l2n(mem['P']).T                                  # B x M
    z = cos / gamma
    mask = (owner[None, :] == jnp.arange(n_class)[:, None])         # C x M
    zc = jnp.where(mask[None], z[:, None, :], -1e9)                 # B x C x M
    lse = jax.scipy.special.logsumexp(zc, -1)
    return kappa * gamma * lse, cos


def balanced_supcon(e, y, protos, owner, n_class, tau=0.1):
    """class-balanced supervised contrastive loss with memory prototypes as extra anchors.
    Denominator terms are averaged within each class so minority classes are not swamped."""
    e = l2n(e); P = l2n(protos)
    feats = jnp.concatenate([e, P], 0)
    lab = jnp.concatenate([y, jnp.asarray(owner)], 0)
    sim = e @ feats.T / tau                                         # B x (B+M)
    B = e.shape[0]
    self_mask = jnp.concatenate([jnp.eye(B, dtype=bool), jnp.zeros((B, P.shape[0]), bool)], 1)
    onehot = jax.nn.one_hot(lab, n_class)                           # (B+M) x C
    cnt = onehot.sum(0)                                             # per-class count incl. prototypes
    ex = jnp.where(self_mask, 0.0, jnp.exp(sim - sim.max(1, keepdims=True)))
    per_class = ex @ onehot / jnp.maximum(cnt - 0, 1)[None]         # mean similarity mass per class
    denom = per_class.sum(1, keepdims=True)
    pos = (lab[None, :] == y[:, None]) & ~self_mask
    logp = (sim - sim.max(1, keepdims=True)) - jnp.log(denom + 1e-12)
    return -(jnp.where(pos, logp, 0).sum(1) / jnp.maximum(pos.sum(1), 1)).mean()


def negative_selection_loss(e, y, mem, owner, m_tol=0.75, m_cens=0.35):
    """(i) self tolerance: self embeddings must sit inside the tolerance cone (cos>=m_tol)
    of their nearest self memory; (ii) censoring: non-self memories may not enter the
    self repertoire (cos(mu_ns, mu_self) <= m_cens) and must not recognise self samples."""
    P = l2n(mem['P']); e = l2n(e)
    own = jnp.asarray(owner)
    cos = e @ P.T
    is_self_p = own == 0
    best_self = jnp.max(jnp.where(is_self_p[None], cos, -2), 1)
    best_ns = jnp.max(jnp.where(~is_self_p[None], cos, -2), 1)
    s = (y == 0).astype(jnp.float32)
    tol = (s * jax.nn.relu(m_tol - best_self) ** 2).sum() / jnp.maximum(s.sum(), 1)
    rec = (s * jax.nn.relu(best_ns - m_cens) ** 2).sum() / jnp.maximum(s.sum(), 1)
    pp = P @ P.T
    cens_mask = (~is_self_p)[:, None] & is_self_p[None, :]
    cens = jnp.where(cens_mask, jax.nn.relu(pp - m_cens) ** 2, 0).sum() / jnp.maximum(cens_mask.sum(), 1)
    return tol + rec + cens

# ------------------------------------------------------------------ simple token baselines

def mlp_init(key, din, hid, dout, layers=2):
    ks = jax.random.split(key, layers + 1)
    dims = [din] + [hid] * layers
    return dict(h=[dense_init(ks[i], dims[i], dims[i + 1]) for i in range(layers)], o=dense_init(ks[-1], hid, dout))


def mlp(p, x, key, train, drop=0.1):
    keys = jax.random.split(key, len(p['h'])) if key is not None else [None] * len(p['h'])
    for i, l in enumerate(p['h']):
        x = dropout(keys[i], jax.nn.relu(dense(l, x)), drop, train)
    return dense(p['o'], x), x


def cnn_init(key, din, K, ch=64, dout=2):
    ks = jax.random.split(key, 4)
    return dict(c1=dict(w=glorot(ks[0], (3, din, ch)), b=jnp.zeros(ch)),
                c2=dict(w=glorot(ks[1], (3, ch, ch)), b=jnp.zeros(ch)),
                f=dense_init(ks[2], 2 * ch, ch), o=dense_init(ks[3], ch, dout))


def _conv1d(p, x):   # x: B x K x C, same padding, kernel 3
    xp = jnp.pad(x, ((0, 0), (1, 1), (0, 0)))
    y = sum(jnp.einsum('bkc,cd->bkd', xp[:, i:i + x.shape[1]], p['w'][i]) for i in range(3))
    return y + p['b']


def cnn(p, T, key, train, drop=0.1):
    h = jax.nn.relu(_conv1d(p['c1'], T))
    h = jax.nn.relu(_conv1d(p['c2'], h))
    h = jnp.concatenate([h.max(1), h.mean(1)], -1)
    h = dropout(key, jax.nn.relu(dense(p['f'], h)), drop, train)
    return dense(p['o'], h), h


def lstm_init(key, din, hid=64, dout=2):
    ks = jax.random.split(key, 3)
    return dict(wx=glorot(ks[0], (din, 4 * hid)), wh=glorot(ks[1], (hid, 4 * hid)), b=jnp.zeros(4 * hid),
                o=dense_init(ks[2], hid, dout))


def lstm(p, T, key, train, drop=0.1):
    B = T.shape[0]; hid = p['wh'].shape[0]
    def step(carry, x):
        h, c = carry
        g = x @ p['wx'] + h @ p['wh'] + p['b']
        i, f, o, u = jnp.split(g, 4, -1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(u)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), None
    (h, c), _ = jax.lax.scan(step, (jnp.zeros((B, hid)), jnp.zeros((B, hid))), jnp.swapaxes(T, 0, 1))
    h = dropout(key, h, drop, train)
    return dense(p['o'], h), h

# ------------------------------------------------------------------ message-passing baselines (full batch)

def seg_mean(msg, dst, n):
    s = jax.ops.segment_sum(msg, dst, n)
    c = jax.ops.segment_sum(jnp.ones((msg.shape[0], 1)), dst, n)
    return s / jnp.maximum(c, 1)


def sage_init(key, din, hid=64, dout=2):
    ks = jax.random.split(key, 5)
    return dict(s1=dense_init(ks[0], din, hid), n1=dense_init(ks[1], din, hid),
                s2=dense_init(ks[2], hid, hid), n2=dense_init(ks[3], hid, hid), o=dense_init(ks[4], hid, dout))


def sage(p, X, edges, key, train, drop=0.1):
    src, dst = edges['homo']; n = X.shape[0]
    h = jax.nn.relu(dense(p['s1'], X) + dense(p['n1'], seg_mean(X[src], dst, n)))
    h = dropout(key, h, drop, train)
    h = jax.nn.relu(dense(p['s2'], h) + dense(p['n2'], seg_mean(h[src], dst, n)))
    return dense(p['o'], h), h


def rgcn_init(key, din, R, hid=64, dout=2):
    ks = jax.random.split(key, 2 * R + 3)
    return dict(s1=dense_init(ks[0], din, hid), s2=dense_init(ks[1], hid, hid), o=dense_init(ks[2], hid, dout),
                r1=[dense_init(ks[3 + r], din, hid) for r in range(R)],
                r2=[dense_init(ks[3 + R + r], hid, hid) for r in range(R)])


def rgcn(p, X, edges, key, train, drop=0.1):
    n = X.shape[0]
    h = dense(p['s1'], X)
    for r, (s, d) in enumerate(edges['rel']):
        h = h + dense(p['r1'][r], seg_mean(X[s], d, n))
    h = dropout(key, jax.nn.relu(h), drop, train)
    g = dense(p['s2'], h)
    for r, (s, d) in enumerate(edges['rel']):
        g = g + dense(p['r2'][r], seg_mean(h[s], d, n))
    g = jax.nn.relu(g)
    return dense(p['o'], g), g


def rgt_init(key, din, R, hid=64, heads=4, dout=2):
    ks = jax.random.split(key, 6 * R + 6)
    lay = lambda i, dd: dict(q=dense_init(ks[i], dd, hid), k=dense_init(ks[i + 1], dd, hid),
                             v=dense_init(ks[i + 2], dd, hid), g=dense_init(ks[i + 3], 2 * hid, hid))
    return dict(inp=dense_init(ks[0], din, hid),
                rel=[lay(6 + 4 * r, hid) for r in range(R)],
                sem_w=dense_init(ks[1], hid, hid), sem_q=0.1 * jax.random.normal(ks[2], (hid,)),
                f=dense_init(ks[3], hid, hid), o=dense_init(ks[4], hid, dout), heads=heads)


def rgt(p, X, edges, key, train, drop=0.1):
    """relation-specific neighbour attention with gated residual + semantic attention (RGT-style)."""
    n = X.shape[0]; H = p['heads']
    h0 = jax.nn.relu(dense(p['inp'], X)); hd = h0.shape[1] // H
    outs = []
    for r, (s, d) in enumerate(edges['rel']):
        L = p['rel'][r]
        q = dense(L['q'], h0).reshape(n, H, hd); k = dense(L['k'], h0).reshape(n, H, hd)
        v = dense(L['v'], h0).reshape(n, H, hd)
        sc = (q[d] * k[s]).sum(-1) / np.sqrt(hd)                     # E x H
        mx = jax.ops.segment_max(sc, d, n)
        ex = jnp.exp(sc - mx[d])
        den = jax.ops.segment_sum(ex, d, n)
        msg = jax.ops.segment_sum(ex[..., None] * v[s], d, n) / jnp.maximum(den, 1e-9)[..., None]
        msg = msg.reshape(n, -1)
        gate = jax.nn.sigmoid(dense(L['g'], jnp.concatenate([h0, msg], -1)))
        outs.append(gate * msg + (1 - gate) * h0)
    Z = jnp.stack(outs, 1)                                           # n x R x hid
    w = jnp.tanh(dense(p['sem_w'], Z)) @ p['sem_q']                  # n x R
    beta = jax.nn.softmax(w.mean(0))                                 # relation importance (global)
    h = (Z * beta[None, :, None]).sum(1)
    h = dropout(key, jax.nn.relu(dense(p['f'], h)), drop, train)
    return dense(p['o'], h), h


def bwgnn_init(key, din, hid=64, C=2, dout=2):
    ks = jax.random.split(key, 4)
    return dict(l1=dense_init(ks[0], din, hid), l2=dense_init(ks[1], hid, hid),
                f=dense_init(ks[2], hid * (C + 1), hid), o=dense_init(ks[3], hid, dout), C=C)


def bwgnn(p, X, edges, key, train, drop=0.1):
    """Beta-wavelet band-pass filters (C=2) on the symmetric normalised Laplacian."""
    from scipy.special import beta as Bfun
    src, dst = edges['homo']; w = edges['homo_w']; n = X.shape[0]; C = p['C']
    h = jax.nn.relu(dense(p['l2'], jax.nn.relu(dense(p['l1'], X))))
    def L(x):     # L x = x - D^-1/2 A D^-1/2 x
        return x - jax.ops.segment_sum(w[:, None] * x[src], dst, n)
    Lh = [h]
    for _ in range(C):
        Lh.append(L(Lh[-1]))
    # beta_i(L) = (L/2)^i (I - L/2)^(C-i) / B(i+1, C+1-i)
    from math import comb
    outs = []
    for i in range(C + 1):
        acc = 0
        # expand (L/2)^i (I-L/2)^(C-i) as polynomial in L
        for j in range(C - i + 1):
            coef = comb(C - i, j) * ((-0.5) ** j) * (0.5 ** i)
            acc = acc + coef * Lh[i + j]
        outs.append(acc / Bfun(i + 1, C + 1 - i))
    hcat = jnp.concatenate(outs, -1)
    hcat = dropout(key, jax.nn.relu(dense(p['f'], hcat)), drop, train)
    return dense(p['o'], hcat), hcat
