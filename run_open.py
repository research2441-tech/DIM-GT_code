"""Leave-one-spam-mode-out open-set protocol on the public graphs.
Spam nodes are partitioned into 3 behavioural modes (k-means on standardised raw
features).  One mode is withheld from training/validation/calibration and appears
only at test time as an unseen spam strategy; the other two modes are known
classes.  Every method flags a node when its closed-set label is a spam class or
its unknown score exceeds the 95th percentile of known calibration nodes."""
import os, sys, json, time, numpy as np, jax, jax.numpy as jnp
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score
from graphdata import load_real
from train import Model, ImmuneConformal, RawSelf
import ais, data_real, metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'open')
os.makedirs(OUT, exist_ok=True)


def modes(D, seed=0, k=3):
    """three equal-sized behavioural modes: tertiles of spam nodes along the leading
    principal component of their standardised features."""
    spam = np.where(D.y > 0)[0]
    Xs = D.X[spam] - D.X[spam].mean(0)
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    pc = Xs @ Vt[0]
    q = np.quantile(pc, [1 / 3, 2 / 3])
    lab = np.zeros(len(D.y), int); lab[spam] = 1 + (pc > q[0]).astype(int) + (pc > q[1]).astype(int)
    return lab


def fpr_at_95(y_unk, s):
    thr = np.quantile(s[y_unk == 1], 0.05)
    return float((s[y_unk == 0] >= thr).mean())


def run(ds, held, seed):
    fn = os.path.join(OUT, '%s_m%d_s%d.json' % (ds, held, seed))
    if os.path.exists(fn): return json.load(open(fn))
    D = load_real(ds, seed=seed)
    md = modes(D)
    tr, va, ca, te = data_real.split(md, seed)
    keep = lambda idx: idx[md[idx] != held]
    tr, va, ca = keep(tr), keep(va), keep(ca)
    known = [m for m in (1, 2, 3) if m != held]
    cmap = {0: 0, known[0]: 1, known[1]: 2, held: -1}
    ycls = np.array([cmap[m] for m in md])
    D.y = np.maximum(ycls, 0)
    unk = (ycls[te] == -1).astype(int)
    yb = (md[te] > 0).astype(int)
    out = dict(ds=ds, held=held, seed=seed, n_unknown=int(unk.sum()), n_test=len(te))

    nl = (unk == 1) | (yb == 0)

    def evaluate(name, closed, u_te, u_ca, spam_score=None):
        thr = np.quantile(u_ca, 0.95)
        flagged = (closed > 0) | (u_te > thr)
        r = dict(open_auroc=roc_auc_score(unk, u_te), fpr95=fpr_at_95(unk, u_te),
                 novel_recall=float(flagged[unk == 1].mean()), novel_isolated=float((u_te > thr)[unk == 1].mean()),
                 legit_fpr=float(flagged[yb == 0].mean()),
                 macro_f1=f1_score(yb, flagged.astype(int), average='macro'))
        if spam_score is not None:
            r['novel_vs_legit_auroc'] = roc_auc_score(unk[nl], spam_score[nl])
        out[name] = {k: float(v) for k, v in r.items()}

    # ----- token-transformer backbone with post-hoc scorers
    m = Model('tokce', 3, dict()).fit(D, tr, va, seed=seed)
    pt, lt, et = m.proba(D, te); pc, lc, ec = m.proba(D, ca); ptr, ltr, etr = m.proba(D, tr)
    closed = pt.argmax(1)
    evaluate('msp', closed, 1 - pt.max(1), 1 - pc.max(1), 1 - pt[:, 0])
    en = lambda l: -np.asarray(jax.scipy.special.logsumexp(jnp.asarray(l), -1))
    evaluate('energy', closed, en(lt), en(lc), 1 - pt[:, 0])
    # Mahalanobis with tied covariance on penultimate embeddings
    ytr = D.y[tr]
    mus = np.stack([etr[ytr == c].mean(0) for c in range(3)])
    Xc = np.concatenate([etr[ytr == c] - mus[c] for c in range(3)])
    cov = np.nan_to_num(Xc.T @ Xc / len(Xc))
    prec = np.linalg.pinv(cov + 1e-3 * np.eye(cov.shape[0]))
    def maha(e):
        d = np.stack([np.einsum('ij,jk,ik->i', e - mu, prec, e - mu) for mu in mus], 1)
        return d.min(1)
    evaluate('mahalanobis', closed, maha(et), maha(ec), 1 - pt[:, 0])
    # deep k-NN (cosine, k=10)
    nz = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    bank = nz(etr)
    def knn(e, k=10):
        out_ = []
        for s in range(0, len(e), 2048):
            sim = nz(e[s:s + 2048]) @ bank.T
            out_.append(-np.sort(sim, 1)[:, -k])
        return np.concatenate(out_)
    evaluate('knn', closed, knn(et), knn(ec), 1 - pt[:, 0])
    # ----- GNNSafe: energy of a GraphSAGE classifier propagated over the graph
    g = Model('sage', 3, dict()).fit(D, tr, va, seed=seed)
    lg, _ = g.forward(D, np.arange(D.N))
    E = en(lg)
    src, dst = D.edges['homo']
    for _ in range(2):
        agg = np.bincount(dst, weights=E[src], minlength=D.N) / np.maximum(np.bincount(dst, minlength=D.N), 1)
        E = 0.5 * E + 0.5 * agg
    pg_ = np.asarray(jax.nn.softmax(jnp.asarray(lg[te]), -1))
    evaluate('gnnsafe', lg[te].argmax(1), E[te], E[ca], 1 - pg_[:, 0])
    # ----- GBDT + MSP
    gb = ais.Classic('gbdt', seed).fit(D, tr, va)
    pg = gb.proba(D, te); pgc = gb.proba(D, ca)
    evaluate('gbdt_msp', pg.argmax(1), 1 - pg.max(1), 1 - pgc.max(1), 1 - pg[:, 0])
    # ----- classical negative selection (self-only detectors): coverage score
    ns = ais.NSA(seed).fit(D, tr, va)
    s_te = ns._score(ns.space(D, te), ns.C, ns.R); s_ca = ns._score(ns.space(D, ca), ns.C, ns.R)
    evaluate('nsa', (s_te > 0).astype(int), s_te, s_ca, s_te)
    # ----- DIM-GT: conformal immune decision
    dm = Model('dimgt', 3, dict()).fit(D, tr, va, seed=seed)
    pt, lt, et = dm.proba(D, te); pc, lc, ec = dm.proba(D, ca)
    ic = ImmuneConformal(dm).calibrate(ec, D.y[ca])
    P_te = ic.pvalues(et); P_ca = ic.pvalues(ec)
    # DIM-GT as specified: conformal immune decision (Eqs. 15-16) at alpha = 0.05
    G = P_te > 0.05; flag = (G.sum(1) == 0) | (P_te.argmax(1) > 0)
    out['dimgt'] = dict(open_auroc=float(roc_auc_score(unk, 1 - P_te.max(1))), fpr95=fpr_at_95(unk, 1 - P_te.max(1)),
                        novel_recall=float(flag[unk == 1].mean()), novel_isolated=float((G.sum(1) == 0)[unk == 1].mean()),
                        legit_fpr=float(flag[yb == 0].mean()), macro_f1=float(f1_score(yb, flag.astype(int), average='macro')),
                        novel_vs_legit_auroc=float(roc_auc_score(unk[nl], (1 - P_te[:, 0])[nl])),
                        defer=float((G.sum(1) > 1).mean()), self_cov=float(G[yb == 0, 0].mean()))
    # same model, softmax read-out with the matched 5 % rule (isolates the decision layer)
    evaluate('dimgt_msp', pt.argmax(1), 1 - pt.max(1), 1 - pc.max(1), 1 - pt[:, 0])
    # analysis variant: self nonconformity also checks input-space distance to legitimate accounts
    rs = RawSelf(D, tr[D.y[tr] == 0], seed=seed)
    r_ca, r_te = rs.score(ca), rs.score(te)
    ic2 = ImmuneConformal(dm).calibrate(ec, D.y[ca], r_ca)
    Q = ic2.pvalues(et, r_te); G2 = Q > 0.05; flag2 = (G2.sum(1) == 0) | (Q.argmax(1) > 0)
    out['dimgt_dual'] = dict(open_auroc=float(roc_auc_score(unk, 1 - Q.max(1))), fpr95=fpr_at_95(unk, 1 - Q.max(1)),
                             novel_recall=float(flag2[unk == 1].mean()), novel_isolated=float((G2.sum(1) == 0)[unk == 1].mean()),
                             legit_fpr=float(flag2[yb == 0].mean()), macro_f1=float(f1_score(yb, flag2.astype(int), average='macro')),
                             novel_vs_legit_auroc=float(roc_auc_score(unk[nl], (1 - Q[:, 0])[nl])),
                             defer=float((G2.sum(1) > 1).mean()), self_cov=float(G2[yb == 0, 0].mean()))
    json.dump(out, open(fn, 'w'))
    return out


if __name__ == '__main__':
    ds = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(',')]
    for s in seeds:
        for h in (1, 2, 3):
            t = time.time(); r = run(ds, h, s)
            print(ds, 'mode', h, 's', s, 'unk', r['n_unknown'], ' '.join('%s %.3f/%.3f' % (k, r[k]['open_auroc'], r[k]['novel_recall'])
                  for k in r if isinstance(r[k], dict)), '%.0fs' % (time.time() - t), flush=True)
