"""Open-world chronological stream on SpamSim-OW (E3/E5/E8/E9 of the protocol).

Users are partitioned once into a labelling pool (60 %) and a never-labelled
evaluation pool (40 %).  Windows 0-2 of pool users train the initial model
(families 0-3 known).  For t = 3..9 every method (i) scores window t,
(ii) receives B analyst labels from pool users active at t, (iii) updates.
A node is flagged when its closed-set label is a spam family OR its unknown score
exceeds the 5 %-FPR threshold estimated on recent labelled known-class nodes.
"""
import os, sys, json, time, argparse, numpy as np, jax, jax.numpy as jnp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.cluster import KMeans
import scipy.sparse as sp
from train import Data, Model, ImmuneConformal
import ais, metrics, nets

SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sim')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'stream')
os.makedirs(OUT, exist_ok=True)
RELS = ['FOLLOW', 'REPOST', 'REPLY', 'MENTION', 'CO-URL', 'CO-TAG']
BUDGET = 150
CAL_KEEP = 300
T0 = 3


class World:
    def __init__(self, seed, preset='A', stats=None):
        z = np.load(os.path.join(SIM, 'sim_%s_%d.npz' % (preset, seed)))
        self.Tw = z['T']                       # W x N x K x d (float16)
        self.Y = z['y']; self.A = z['active']; self.W, self.N = self.Y.shape
        order = np.argsort(z['time_id'], kind='stable')      # current-window tokens first (self token at 0)
        self.rel_id = z['rel_id'][order]; self.time_id = z['time_id'][order]
        self.z = z
        T = self.Tw.astype(np.float32)[:, :, order].reshape(self.W * self.N, len(order), self.Tw.shape[-1])
        if stats is None:
            base = T[:3 * self.N][self.A[:3].reshape(-1)]
            mu = base.reshape(-1, base.shape[-1]).mean(0); sd = base.reshape(-1, base.shape[-1]).std(0) + 1e-3
            stats = (mu, sd)
        self.stats = stats
        self.T = ((T - stats[0]) / stats[1]).clip(-8, 8).astype(np.float32)
        self.yflat = self.Y.reshape(-1)
        self.actflat = self.A.reshape(-1)

    def idx(self, t, users):
        return t * self.N + np.asarray(users)

    def graph_edges(self, windows, cap=32, seed=0):
        rng = np.random.default_rng(seed)
        rel = []
        for r in RELS:
            S, Dd = [], []
            for t in windows:
                e = self.z['t%d_%s' % (t, r)]
                s, d = np.r_[e[0], e[1]], np.r_[e[1], e[0]]          # symmetrise
                # cap in-degree
                o = rng.permutation(len(d)); s, d = s[o], d[o]
                cnt = {}
                order = np.argsort(d, kind='stable'); s, d = s[order], d[order]
                bnd = np.r_[0, np.where(np.diff(d) != 0)[0] + 1, len(d)]
                keep = np.zeros(len(d), bool)
                for a, b in zip(bnd[:-1], bnd[1:]):
                    keep[a:a + cap] = True
                S.append(s[keep] + t * self.N); Dd.append(d[keep] + t * self.N)
            rel.append((np.concatenate(S).astype(np.int32), np.concatenate(Dd).astype(np.int32)))
        hs = np.concatenate([r[0] for r in rel]); hd = np.concatenate([r[1] for r in rel])
        deg = np.bincount(hd, minlength=self.W * self.N).astype(np.float32)
        w = (1 / np.sqrt(np.maximum(deg[hd], 1) * np.maximum(deg[hs], 1))).astype(np.float32)
        return dict(homo=(hs, hd), homo_w=w, rel=rel)


def make_data(world, ylab, edges=None, temporal=True):
    T = world.T
    rel_id, time_id = world.rel_id, world.time_id
    if not temporal:                       # current window only
        keep = time_id == 0
        T = T[:, keep]; rel_id = rel_id[keep]; time_id = time_id[keep]
    D = Data(T, rel_id, time_id, ylab, X=world.T[:, 0], edges=edges, time_fourier=True)
    return D


# ------------------------------------------------------------------ helpers for class-incremental heads
def expand_head(model, new_C, data=None, idx=None, y=None, key=0):
    old = model.C
    if new_C == old: return
    p = model.params
    if model.kind in ('dimgt',) and 'mem' in p:
        P = np.asarray(p['mem']['P'])
        add = []
        for c in range(old, new_C):
            add.append(np.full(model.cfg['m_ns'], c))
        new_owner = np.r_[model.owner, np.concatenate(add)]
        newP = np.random.default_rng(key).normal(size=(len(new_owner) - len(P), P.shape[1]))
        p['mem']['P'] = jnp.asarray(np.vstack([P, newP / np.linalg.norm(newP, axis=1, keepdims=True)]))
        model.owner = new_owner; model.C = new_C
        if data is not None:
            model.seed_prototypes(data, idx, y, classes=range(old, new_C))
        if getattr(model, 'censor', True):
            censor_at_birth(model, range(old, new_C))
    else:
        if model.kind in ('tokce',):
            h = p['head']
        elif model.kind == 'mlp':
            h = p['o']
        elif model.kind in ('lstm', 'cnn', 'rgcn', 'rgt', 'sage', 'bwgnn'):
            h = p['o']
        W = np.asarray(h['w']); b = np.asarray(h['b'])
        rng = np.random.default_rng(key)
        W2 = np.hstack([W, rng.normal(0, 0.01, (W.shape[0], new_C - old))])
        b2 = np.r_[b, np.full(new_C - old, b[1:].mean() if len(b) > 1 else 0)]
        h['w'] = jnp.asarray(W2, jnp.float32); h['b'] = jnp.asarray(b2, jnp.float32)
        model.C = new_C


def censor_at_birth(model, classes, m_cen=None, step=0.5, iters=50):
    """negative selection for newborn memory cells: a new non-self prototype that lies inside the
    self region (cosine to a self prototype above m_cen) is pushed away from its nearest self cell
    until it is self-tolerant, mirroring the deletion of self-reactive detectors."""
    m_cen = model.cfg['m_cens'] if m_cen is None else m_cen
    P = np.array(model.params['mem']['P']); P /= np.linalg.norm(P, axis=1, keepdims=True)
    S = P[model.owner == 0]
    for c in classes:
        for j in np.where(model.owner == c)[0]:
            mu = P[j]
            for _ in range(iters):
                cs = S @ mu
                if cs.max() <= m_cen: break
                mu = mu - step * (cs.max() - m_cen) * S[cs.argmax()]
                mu /= np.linalg.norm(mu)
            P[j] = mu
    model.params['mem']['P'] = jnp.asarray(P)


def energy(logits):
    return -jax.scipy.special.logsumexp(jnp.asarray(logits), -1)


# ------------------------------------------------------------------ stream runner
class Labels:
    """analyst label store: family -> class mapping grows when a family is first labelled."""
    def __init__(self, known=(0, 1, 2, 3)):
        self.fam2cls = {f: i for i, f in enumerate(known)}
        self.train, self.cal = [], []          # flat node ids

    def cls(self, fam):
        return np.array([self.fam2cls.get(int(f), -1) for f in fam])

    def add_family(self, f):
        if f not in self.fam2cls: self.fam2cls[f] = len(self.fam2cls)

    @property
    def C(self):
        return len(self.fam2cls)


def run_stream(seed, method, preset='A', cfgover=None, budget=BUDGET, verbose=True):
    tag = method
    fn = os.path.join(OUT, '%s_%s_s%d.json' % (preset, tag.replace(':', '_').replace('=', '').replace(',', '_'), seed))
    if os.path.exists(fn):
        return json.load(open(fn))
    t_start = time.time()
    W = World(seed, preset)
    rng = np.random.default_rng(100 + seed)
    users = rng.permutation(W.N)
    pool, evalu = users[:int(.6 * W.N)], users[int(.6 * W.N):]
    in_pool = np.zeros(W.N, bool); in_pool[pool] = True
    # initial labelled data: windows 0-2, pool users, families 0-3
    pu = rng.permutation(pool)
    n = len(pu); tr_u, va_u, ca_u = pu[:int(n * 2 / 3)], pu[int(n * 2 / 3):int(n * 5 / 6)], pu[int(n * 5 / 6):]
    def nodes(us, ts):
        out = np.concatenate([W.idx(t, us) for t in ts])
        return out[W.actflat[out]]
    L = Labels()
    tr = nodes(tr_u, range(3)); va = nodes(va_u, range(3)); ca = nodes(ca_u, range(3))
    labelled = np.zeros(W.N, bool); labelled[tr_u] = labelled[va_u] = labelled[ca_u] = True

    # ---------------- method configuration
    base = method.split('+')[0]
    opts = set(method.split('+')[1:])
    temporal = 'notemp' not in opts
    cfg = dict(epochs=80, patience=12)
    if 'self' in opts: cfg['self_only'] = True
    if 'nons' in opts: cfg['use_ns'] = False
    cfg.update(cfgover or {})
    graph = base in ('rgcn', 'rgt')
    is_dim = base == 'dimgt'
    strategy = 'joint'                                   # joint retraining from scratch every window
    if base in ('ft', 'ewc', 'er', 'lwf') or is_dim:
        strategy = base if not is_dim else 'dim'
    kind = {'ft': 'tokce', 'ewc': 'tokce', 'er': 'tokce', 'lwf': 'tokce', 'tokce': 'tokce'}.get(base, base)

    def data_for(windows):
        ylab = np.full(W.W * W.N, -1)
        ylab_all = L.cls(W.yflat)
        ylab[:] = ylab_all
        edges = W.graph_edges(windows, seed=seed) if graph else None
        D = make_data(W, np.maximum(ylab, 0), edges, temporal)
        D.ylab = ylab
        return D

    D = data_for(range(3))
    model = None; fisher = None; buffer = {}
    t_fit = 0.0

    def fit_initial():
        nonlocal model, t_fit
        t0 = time.time()
        if base in ('gbdt', 'nsaga', 'rf'):
            m = ais.Classic('gbdt', seed, fast=True) if base == 'gbdt' else (ais.Classic('rf', seed) if base == 'rf' else ais.NSA(seed, ga=True))
            m.fit(D, tr, va)
            model = m
        else:
            model = Model(kind, L.C, cfg).fit(D, tr, va, seed=seed)
            model.censor = 'nocensor' not in opts
        t_fit += time.time() - t0

    fit_initial()
    recent = {c: list(ca[D.ylab[ca] == c]) for c in range(L.C)}     # calibration pool (most recent first)
    hist = []
    # memory-cell buffer for replay (dimgt) / reservoir (er)
    def refresh_buffer(model, pool_nodes, per_class=60):
        lab = D.ylab[pool_nodes]
        buf = {}
        if is_dim and 'mem' in model.params:
            _, _, e = model.proba(D, pool_nodes)
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            P = np.asarray(model.params['mem']['P']); P = P / np.linalg.norm(P, axis=1, keepdims=True)
            r_ = np.random.default_rng(seed)
            for c in range(model.C):
                m = np.where(lab == c)[0]
                if len(m) == 0: continue
                pc = per_class * (3 if (c == 0 and 'nomix' not in opts) else 1)
                prot = np.where(model.owner == c)[0]
                n_near = pc // 2 if 'nomix' not in opts else pc
                k = max(1, n_near // len(prot))
                sel = set()
                for j in prot:
                    d = e[m] @ P[j]
                    sel.update(m[np.argsort(-d)[:k]].tolist())
                if 'nomix' not in opts:       # add boundary coverage: random cells of the class
                    rest = np.setdiff1d(m, np.array(sorted(sel)))
                    if len(rest):
                        sel.update(r_.choice(rest, min(pc - len(sel), len(rest)), replace=False).tolist())
                buf[c] = pool_nodes[np.array(sorted(sel))]
        else:
            r = np.random.default_rng(seed)
            for c in np.unique(lab):
                m = np.where(lab == c)[0]
                buf[c] = pool_nodes[r.choice(m, min(per_class, len(m)), replace=False)]
        return buf

    if strategy in ('er', 'dim'):
        buffer = refresh_buffer(model, tr)
    if strategy == 'ewc':
        fisher = (model.fisher(D, tr, seed=seed), jax.tree_util.tree_map(lambda x: x, model.params))
    all_lab_train = list(tr)

    for t in range(T0 - 1, W.W):
        # ---------------- data view for this window
        if graph:
            D = data_for(range(t + 1))
        else:
            D.ylab = L.cls(W.yflat); D.y = np.maximum(D.ylab, 0)
        ev = W.idx(t, evalu); ev = ev[W.actflat[ev]]
        fam = W.yflat[ev]
        known_fams = set(L.fam2cls.keys())
        is_unknown = np.array([f not in known_fams for f in fam])
        yb = (fam > 0).astype(int)
        # ---------------- scoring
        if isinstance(model, Model):
            probs, logits, emb = model.proba(D, ev)
        else:
            probs = model.proba(D, ev); logits = np.log(probs + 1e-9); emb = None
        calnodes = np.concatenate([np.array(v[:CAL_KEEP], int) for v in recent.values() if len(v)])
        if isinstance(model, Model):
            cp, cl, ce = model.proba(D, calnodes)
        else:
            cp = model.proba(D, calnodes); cl = np.log(cp + 1e-9); ce = None
        if is_dim and 'mem' in model.params:
            ic = ImmuneConformal(model).calibrate(ce, D.ylab[calnodes])
            G, dec, Pv = ic.decide(emb, 0.05)
            unk = 1 - Pv.max(1)
            flag_unknown = dec == -1
            closed = Pv.argmax(1)            # immune decision on calibrated p-values (Eq. 13)
            abstain = dec == -2
            spam_score = 1 - probs[:, 0]
        else:
            unk = np.asarray(energy(logits)) if isinstance(model, Model) else 1 - probs.max(1)
            cu = np.asarray(energy(cl)) if isinstance(model, Model) else 1 - cp.max(1)
            thr_u = np.quantile(cu, 0.95)
            flag_unknown = unk > thr_u
            closed = probs.argmax(1)
            abstain = np.zeros(len(ev), bool)
            spam_score = 1 - probs[:, 0]
        flagged = (closed > 0) | flag_unknown
        rec = dict(t=t)
        rec['macro_f1'] = f1_score(yb, flagged.astype(int), average='macro')
        rec['mcc'] = matthews_corrcoef(yb, flagged.astype(int))
        tpr = flagged[yb == 1].mean(); tnr = 1 - flagged[yb == 0].mean()
        rec['gmean'] = float(np.sqrt(tpr * tnr)); rec['bacc'] = float((tpr + tnr) / 2)
        rec['legit_fpr'] = float(flagged[yb == 0].mean())
        rec['legit_fpr_closed'] = float((closed > 0)[yb == 0].mean())
        rec['legit_fpr_novel'] = float((flag_unknown & (closed == 0))[yb == 0].mean())
        rec['pr_auc'] = average_precision_score(yb, spam_score)
        rec['ece'] = metrics.ece(yb, spam_score)
        if is_unknown.any():
            rec['novel_recall'] = float(flagged[is_unknown].mean())
            rec['novel_isolated'] = float(flag_unknown[is_unknown].mean())
            rec['open_auroc'] = roc_auc_score(is_unknown.astype(int), unk)
        kn = ~is_unknown
        cls_true = L.cls(fam[kn])
        rec['known_family_macro_recall'] = float(np.mean([(closed[kn][cls_true == c] == c).mean()
                                                          for c in np.unique(cls_true)]))
        old = np.isin(fam, [1, 2, 3])
        rec['old_family_recall'] = float(flagged[old].mean())
        rec['old_family_exact'] = float((closed[old] == L.cls(fam[old])).mean())
        per_fam = {int(f): float(flagged[fam == f].mean()) for f in np.unique(fam) if f > 0}
        rec['per_family_recall'] = per_fam
        rec['abstain'] = float(abstain.mean())
        if abstain.any() or is_dim:
            acc_mask = ~abstain & ~flag_unknown
            rec['accepted_err'] = float(((closed[acc_mask] > 0).astype(int) != yb[acc_mask]).mean()) if acc_mask.any() else np.nan
        rec['n_known_classes'] = L.C
        hist.append(rec)
        if verbose:
            print('  t=%d F1 %.3f PR %.3f FPR %.3f (cl %.3f nov %.3f) novel %s openAUC %s oldrec %.3f C=%d' % (
                t, rec['macro_f1'], rec['pr_auc'], rec['legit_fpr'], rec['legit_fpr_closed'], rec['legit_fpr_novel'], '%.3f' % rec.get('novel_recall', np.nan),
                '%.3f' % rec.get('open_auroc', np.nan), rec['old_family_recall'], L.C), flush=True)
        if t == W.W - 1 or t < T0 - 1 + 1 and t == T0 - 1 and False:
            break
        if t == T0 - 1:
            continue                                      # window 2 is the in-distribution reference
        # ---------------- analyst labels
        cand = W.idx(t, pool); cand = cand[W.actflat[cand]]
        cand = cand[~np.isin(cand, all_lab_train)]
        if is_dim and 'random' not in opts:
            ce2 = model.proba(D, cand)[2]
            G2, d2, _ = ic.decide(ce2, 0.05)
            pri = np.where(d2 < 0)[0]
            k1 = min(len(pri), int(budget * 0.6))
            pick = list(rng.choice(pri, k1, replace=False)) if k1 else []
            rest = np.setdiff1d(np.arange(len(cand)), pick)
            pick += list(rng.choice(rest, budget - k1, replace=False))
            newl = cand[np.array(pick)]
        else:
            newl = rng.choice(cand, min(budget, len(cand)), replace=False)
        for f in np.unique(W.yflat[newl]):
            L.add_family(int(f))
        D.ylab = L.cls(W.yflat); D.y = np.maximum(D.ylab, 0)
        perm = rng.permutation(newl)
        n_cal = int(.3 * len(perm))
        new_cal, new_tr = perm[:n_cal], perm[n_cal:]
        for c in range(L.C):
            recent.setdefault(c, [])
            recent[c] = list(new_cal[D.ylab[new_cal] == c]) + recent[c]
        all_lab_train += list(new_tr)
        # ---------------- update
        t0 = time.time()
        if graph:
            D = data_for(range(t + 1))
        if strategy == 'joint':
            idx_all = np.array(all_lab_train)
            vv = np.concatenate([np.array(v[:CAL_KEEP], int) for v in recent.values() if len(v)])
            if isinstance(model, Model):
                # cumulative fine-tuning on every label collected so far (warm start, head expanded)
                expand_head(model, L.C, key=seed + t)
                model.fit(D, idx_all, vv, seed=seed + t, epochs=20, init=False, lr=1e-3)
            else:
                model.fit(D, idx_all, vv)
        else:
            old_C = model.C
            ytr_new = D.ylab[new_tr]
            prev = None
            if strategy in ('lwf', 'dim') and 'nokd' not in opts:
                import copy
                prev = Model(model.kind, old_C, model.cfg); prev.params = jax.tree_util.tree_map(lambda x: x, model.params)
                prev.owner = None if model.owner is None else model.owner.copy()
            newcls = [c for c in range(old_C, L.C)]
            if is_dim:
                # clonal expansion: new memory prototypes seeded from the newly labelled family
                expand_head(model, L.C, D, new_tr[np.isin(ytr_new, newcls)] if newcls else None,
                            ytr_new[np.isin(ytr_new, newcls)] if newcls else None, key=seed + t)
            else:
                expand_head(model, L.C, key=seed + t)
            if strategy in ('er', 'dim') and 'noreplay' not in opts:
                rep = np.concatenate([v for v in buffer.values()]) if buffer else np.zeros(0, int)
                idx_upd = np.r_[new_tr, rep].astype(int)
            else:
                idx_upd = new_tr
            vv = np.concatenate([np.array(v[:CAL_KEEP], int) for v in recent.values() if len(v)])
            ew = None
            if strategy == 'ewc':
                F, oldp = fisher
                # pad fisher/old params to the expanded head shape
                def pad(a, b):
                    a = np.asarray(a); b = np.asarray(b)
                    if a.shape == b.shape: return jnp.asarray(a)
                    out = np.zeros(b.shape, np.float32); sl = tuple(slice(0, s) for s in a.shape); out[sl] = a
                    return jnp.asarray(out)
                F = jax.tree_util.tree_map(pad, F, model.params); oldp = jax.tree_util.tree_map(pad, oldp, model.params)
                # Fisher normalised to unit mean so that the penalty weight is scale-free
                fm = float(np.mean(np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(F)])))
                F = jax.tree_util.tree_map(lambda x: x / max(fm, 1e-12), F)
                ew = (F, oldp, 100.0)
            prior = None          # logit adjustment uses the class balance of the update set itself
            es = None
            if is_dim:
                es = 0.0 if 'encfreeze' in opts else (None if 'encfull' in opts else 0.1)
            if prev is not None:
                # distil on all update nodes, old classes only
                model.fit(D, idx_upd, vv, seed=seed + t, epochs=25, init=False, old_model=prev, prior=prior, ewc=ew, lr=1e-3, enc_scale=es)
            else:
                model.fit(D, idx_upd, vv, seed=seed + t, epochs=25, init=False, prior=prior, ewc=ew, lr=1e-3, enc_scale=es)
            if strategy in ('er', 'dim'):
                buffer = refresh_buffer(model, np.array(all_lab_train))
            if strategy == 'ewc':
                fisher = (model.fisher(D, np.array(all_lab_train), seed=seed), jax.tree_util.tree_map(lambda x: x, model.params))
            # drift-adaptive self memory: bounded EMA towards confidently-self unlabelled nodes
            if is_dim and 'nodrift' not in opts:
                cur = W.idx(t, users); cur = cur[W.actflat[cur]]
                _, _, ecur = model.proba(D, cur)
                calnodes = np.concatenate([np.array(v[:CAL_KEEP], int) for v in recent.values() if len(v)])
                ic = ImmuneConformal(model).calibrate(model.proba(D, calnodes)[2], D.ylab[calnodes])
                _, _, Pv = ic.decide(ecur, 0.05)
                conf_self = (Pv[:, 0] > 0.5) & (Pv[:, 1:].max(1) <= 0.05)
                if conf_self.sum() > 50:
                    e = ecur[conf_self]; e = e / np.linalg.norm(e, axis=1, keepdims=True)
                    P = np.array(model.params['mem']['P']); Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
                    sidx = np.where(model.owner == 0)[0]
                    assign = (e @ Pn[sidx].T).argmax(1)
                    for j, pj in enumerate(sidx):
                        m = assign == j
                        if m.sum() < 10: continue
                        delta = e[m].mean(0) - Pn[pj]
                        nrm = np.linalg.norm(delta)
                        if nrm > 0.15: delta *= 0.15 / nrm
                        P[pj] = Pn[pj] + 0.3 * delta
                        P[pj] /= np.linalg.norm(P[pj])
                    model.params['mem']['P'] = jnp.asarray(P)
            # tolerance induction: analyst-confirmed legitimate accounts that the memory would flag
            # spawn new self cells (the self-side counterpart of clonal expansion)
            if is_dim and 'notolind' not in opts:
                labs = np.r_[new_tr, new_cal].astype(int)
                legit = labs[D.ylab[labs] == 0]
                if len(legit) >= 5:
                    calnodes = np.concatenate([np.array(v[:CAL_KEEP], int) for v in recent.values() if len(v)])
                    ic2 = ImmuneConformal(model).calibrate(model.proba(D, calnodes)[2], D.ylab[calnodes])
                    el = model.proba(D, legit)[2]
                    Pl = ic2.pvalues(el)
                    miss = (Pl.argmax(1) != 0) | (Pl.max(1) <= 0.05)
                    if miss.sum() >= 5:
                        em = el[miss] / np.linalg.norm(el[miss], axis=1, keepdims=True)
                        k = int(min(3, miss.sum() // 5))
                        cen = KMeans(k, n_init=3, random_state=seed).fit(em).cluster_centers_
                        cen = cen / np.linalg.norm(cen, axis=1, keepdims=True)
                        P = np.array(model.params['mem']['P']); own = model.owner.copy()
                        P = np.vstack([P, cen]); own = np.r_[own, np.zeros(k, int)]
                        # cap the self repertoire: drop the least-used self cells
                        sidx = np.where(own == 0)[0]
                        if len(sidx) > 18:
                            ecal = model.proba(D, calnodes[D.ylab[calnodes] == 0])[2]
                            ecal = ecal / np.linalg.norm(ecal, axis=1, keepdims=True)
                            Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
                            use = np.bincount((ecal @ Pn[sidx].T).argmax(1), minlength=len(sidx))
                            drop = sidx[np.argsort(use)[:len(sidx) - 18]]
                            keep = np.setdiff1d(np.arange(len(own)), drop)
                            P, own = P[keep], own[keep]
                        model.params['mem']['P'] = jnp.asarray(P); model.owner = own
        t_fit += time.time() - t0
    res = dict(seed=seed, method=method, preset=preset, hist=hist, t_fit=t_fit, t_total=time.time() - t_start,
               n_params=model.n_params(), fam2cls={str(k): v for k, v in L.fam2cls.items()})
    json.dump(res, open(fn, 'w'), default=float)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('methods'); ap.add_argument('seeds'); ap.add_argument('--preset', default='A')
    a = ap.parse_args()
    for m in a.methods.split(';'):
        for s in [int(x) for x in a.seeds.split(',')]:
            t = time.time()
            print('== %s seed %d' % (m, s), flush=True)
            try:
                r = run_stream(s, m, a.preset)
            except Exception as e:
                import traceback; traceback.print_exc(); continue
            h = r['hist']
            print('   mean F1(t>=3) %.4f  %.0fs' % (np.mean([x['macro_f1'] for x in h if x['t'] >= 3]), time.time() - t), flush=True)
