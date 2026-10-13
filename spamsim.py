"""SpamSim-OW: open-world temporal heterogeneous social-spam simulator.

Users post in T windows. Each post may carry a URL, hashtags, mentions, be a
reply or a repost. User-user relations per window are derived from the posts:
FOLLOW (slowly growing), REPOST, REPLY, MENTION, CO-URL, CO-HASHTAG.

Ground truth: family 0 = legitimate (five personas incl. benign automation);
families 1..7 = spam strategies with staggered emergence:
  1 URL promotion      (t>=0)     2 hashtag hijacking   (t>=0)
  3 follower farms     (t>=0)     4 reply/mention spam  (t>=3)
  5 repost rings       (t>=5)     6 LLM-assisted mimics (t>=7)
  7 account takeover   (t>=8, previously legitimate accounts)
Legitimate behaviour drifts (activity, topics, URL use) and a trending event at
t=6 raises hashtag/trend usage among legitimate users.  Known families adapt
(family 1 lowers URL rate and writes on-topic text from t=5; family 2 uses
fewer hashtags from t=4).  Platform presets change the persona mix and base
rates for cross-platform transfer.
"""
import numpy as np

FEATS = ['log_posts', 'log_reposts', 'log_replies', 'url_ratio', 'hashtags_pp', 'mentions_pp',
         'dup_ratio', 'lex_div', 'time_entropy', 'burstiness', 'log_age', 'log_followers',
         'log_following', 'ff_ratio', 'domain_div', 'trend_ratio'] + ['topic%d' % i for i in range(8)]
RELS = ['FOLLOW', 'REPOST', 'REPLY', 'MENTION', 'CO-URL', 'CO-TAG']
FAMILIES = ['legit', 'url-promo', 'hashtag-hijack', 'follower-farm', 'reply-mention',
            'repost-ring', 'llm-mimic', 'takeover']
EMERGE = {1: 0, 2: 0, 3: 0, 4: 3, 5: 5, 6: 7, 7: 8}
NTOP = 8


class Platform:
    def __init__(self, preset='A'):
        if preset == 'A':
            self.n_legit = 6000
            self.persona_p = [0.47, 0.20, 0.10, 0.13, 0.10]   # casual, news, creator, org, benign-bot
            self.rate_scale = 1.0; self.url_scale = 1.0; self.tag_scale = 1.0
        else:                                                 # platform B: more creators/orgs, more URLs, fewer tags
            self.n_legit = 6000
            self.persona_p = [0.35, 0.15, 0.20, 0.18, 0.12]
            self.rate_scale = 1.35; self.url_scale = 1.4; self.tag_scale = 0.7
        self.n_fam = {1: 320, 2: 300, 3: 420, 4: 320, 5: 260, 6: 300, 7: 220}


# behavioural templates: rate, p_url, tags/post, mentions/post, p_reply, p_repost,
# duplicate ratio, lexical diversity, timing regularity (0 human .. 1 clockwork), burst
PERSONA = {
    0: dict(rate=6, url=.10, tag=.4, men=.5, rep=.30, rp=.25, dup=.04, lex=.62, reg=.10, bur=.25),
    1: dict(rate=9, url=.35, tag=.3, men=.2, rep=.10, rp=.45, dup=.05, lex=.58, reg=.15, bur=.25),
    2: dict(rate=16, url=.25, tag=1.0, men=.6, rep=.25, rp=.10, dup=.06, lex=.66, reg=.20, bur=.30),
    3: dict(rate=12, url=.55, tag=.9, men=.3, rep=.15, rp=.05, dup=.18, lex=.52, reg=.45, bur=.20),
    4: dict(rate=22, url=.80, tag=.6, men=.0, rep=.00, rp=.02, dup=.45, lex=.40, reg=.85, bur=.10),
}
SPAM = {
    1: dict(rate=20, url=.85, tag=.5, men=.3, rep=.05, rp=.05, dup=.55, lex=.35, reg=.55, bur=.65),
    2: dict(rate=18, url=.40, tag=3.2, men=.2, rep=.05, rp=.10, dup=.40, lex=.40, reg=.45, bur=.60),
    3: dict(rate=1.5, url=.10, tag=.2, men=.1, rep=.05, rp=.50, dup=.30, lex=.45, reg=.35, bur=.40),
    4: dict(rate=24, url=.30, tag=.3, men=2.4, rep=.75, rp=.02, dup=.50, lex=.38, reg=.50, bur=.55),
    5: dict(rate=26, url=.20, tag=.6, men=.2, rep=.02, rp=.85, dup=.25, lex=.45, reg=.60, bur=.80),
    6: dict(rate=10, url=.30, tag=.7, men=.6, rep=.30, rp=.25, dup=.07, lex=.64, reg=.22, bur=.33),
    7: dict(rate=21, url=.75, tag=.9, men=.9, rep=.10, rp=.10, dup=.42, lex=.42, reg=.40, bur=.60),
}


def _dirichlet_rows(rng, alpha, n):
    return rng.dirichlet(alpha, size=n)


def simulate(seed=0, T=10, preset='A'):
    rng = np.random.default_rng(seed)
    P = Platform(preset)
    # ---------------- population ----------------
    persona = rng.choice(5, size=P.n_legit, p=P.persona_p)
    fam = np.zeros(P.n_legit, int)
    birth = np.zeros(P.n_legit, int)
    age0 = rng.gamma(3.0, 500, P.n_legit)                       # days
    for f in range(1, 7):
        n = P.n_fam[f]
        persona = np.r_[persona, -np.ones(n, int)]
        fam = np.r_[fam, np.full(n, f)]
        birth = np.r_[birth, np.full(n, EMERGE[f])]
        # spam accounts are young and created in batches
        aged = rng.random(n) < (.75 if f == 6 else .45)     # purchased / aged accounts
        age0 = np.r_[age0, np.where(aged, rng.gamma(3.0, 450, n), rng.gamma(2.0, 30, n))]
    N = len(fam)
    # each spam account interleaves spam with ordinary behaviour of a cover persona
    rho = np.zeros(N)
    cover = rng.choice(5, size=N, p=[.5, .2, .15, .15, 0.0])
    for f in range(1, 7):
        m = fam == f
        rho[m] = rng.beta(1.6, 4.0, m.sum()) if f == 6 else rng.beta(2.0, 2.2, m.sum())
    jitter = rng.lognormal(0, .5, (N, 6))              # individual behavioural heterogeneity
    # takeover victims: random legitimate casual/news/creator accounts
    cand = np.where((fam == 0) & (persona <= 2))[0]
    victims = rng.choice(cand, P.n_fam[7], replace=False)
    takeover = np.zeros(N, bool); takeover[victims] = True
    rho[victims] = rng.beta(3, 3, len(victims)); cover[victims] = persona[victims]
    # topic preferences
    topic = _dirichlet_rows(rng, np.full(NTOP, .35), N)
    # spam families share a family-level topic signature (except mimics, who copy legit profiles)
    fam_topic = {f: rng.dirichlet(np.full(NTOP, .5)) for f in range(1, 7)}
    for f in range(1, 7):
        idx = fam == f
        if f == 6:
            donors = rng.choice(np.where(fam == 0)[0], idx.sum())
            topic[idx] = topic[donors] * .85 + .15 * fam_topic[f]
        else:
            topic[idx] = .6 * fam_topic[f] + .4 * _dirichlet_rows(rng, np.full(NTOP, .5), idx.sum())
    # follower graph (static base with growth)
    popularity = np.where(fam == 0, np.where(persona == 2, 8., np.where(persona == 3, 6., np.where(persona == 1, 1.5, 1.))), .3)
    popularity *= rng.lognormal(0, .8, N)
    legit_idx = np.where(fam == 0)[0]
    follow_src, follow_dst = [], []
    pop_p = popularity[legit_idx] / popularity[legit_idx].sum()
    for u in legit_idx:
        k = int(rng.gamma(2, 25)) + 3
        nb = rng.choice(legit_idx, k, p=pop_p)
        follow_src += [u] * k; follow_dst += list(nb)
    # spam following behaviour
    customers = rng.choice(np.where(fam == 1)[0], 60, replace=False)
    for f in range(1, 7):
        idx = np.where(fam == f)[0]
        for u in idx:
            if f == 3:      # farm: follow paying customers + random legit
                k = int(rng.gamma(3, 60)) + 50
                nb = np.r_[rng.choice(customers, min(40, k)), rng.choice(legit_idx, k)]
            elif f == 6:    # mimics follow legit like a normal user, plus a few co-members
                k = int(rng.gamma(2, 25)) + 3
                nb = np.r_[rng.choice(legit_idx, k, p=pop_p), rng.choice(idx, 3)]
            else:
                k = int(rng.gamma(2, 40)) + 10
                nb = np.r_[rng.choice(legit_idx, k, p=pop_p), rng.choice(idx, 5)]
            follow_src += [u] * len(nb); follow_dst += list(nb)
    # organic follow-backs received by spam accounts (aged and mimic accounts attract more)
    for u in np.where(fam > 0)[0]:
        if fam[u] == 3: continue
        k = int(rng.gamma(2, 20 if fam[u] == 6 else 8)) + 1
        follow_src += list(rng.choice(legit_idx, k)); follow_dst += [u] * k
    follow_src = np.array(follow_src); follow_dst = np.array(follow_dst)
    keep = follow_src != follow_dst
    follow_src, follow_dst = follow_src[keep], follow_dst[keep]
    # farms are followed by almost nobody; spam followers mostly come from farms
    farm = np.where(fam == 3)[0]

    # entity pools
    legit_domains = 4000
    fam_domains = {f: legit_domains + 100 * f + np.arange(60) for f in range(1, 8)}
    tags_per_topic = 60
    ring_groups = {}
    idx5 = np.where(fam == 5)[0]; ring_groups = np.array_split(rng.permutation(idx5), 10)
    ring_of = -np.ones(N, int)
    for g, mem in enumerate(ring_groups): ring_of[mem] = g
    idx6 = np.where(fam == 6)[0]; cell6 = np.array_split(rng.permutation(idx6), 15)
    cell_of = -np.ones(N, int)
    for g, mem in enumerate(cell6): cell_of[mem] = g
    hidden6 = fam_domains[6]

    windows = []
    url_counter = [0]
    for t in range(T):
        active = birth <= t
        y = fam.copy()
        if t >= EMERGE[7]:
            y[takeover] = 7
        drift = t / (T - 1)
        trend_tags = 10000 + 50 * t + np.arange(25)
        event = 1.0 if t >= 6 else 0.0
        posts_u, posts_url, posts_dom, posts_tags, posts_men, posts_rep, posts_rp, posts_time, posts_topic = [], [], [], [], [], [], [], [], []
        feats = np.zeros((N, len(FEATS)), np.float32)
        users = np.where(active)[0]
        for u in users:
            f = y[u]
            if f == 0:
                b = dict(PERSONA[persona[u]])
                b['rate'] *= P.rate_scale * (1 + .35 * drift)
                b['url'] = min(.95, b['url'] * P.url_scale * (1 + .4 * drift))
                b['tag'] *= P.tag_scale * (1 + 1.6 * event * (persona[u] != 4))
                trend_p = .08 + .45 * event
            else:
                bs = dict(SPAM[f])
                trend_p = .85 if f == 2 else (.15 if f != 7 else .4)
                if f == 1 and t >= 5:      # adaptive evasion of URL promoters
                    bs['url'] *= .6; bs['dup'] *= .6; bs['lex'] = .5
                if f == 2 and t >= 4:
                    bs['tag'] *= .55
                if f == 6:
                    trend_p = .12 + .35 * event
                bl = PERSONA[cover[u]]
                r_ = rho[u]
                b = {k: r_ * bs[k] + (1 - r_) * bl[k] for k in bs}
                trend_p = r_ * trend_p + (1 - r_) * (.08 + .45 * event)
            for j, k in enumerate(('rate', 'url', 'tag', 'men', 'rep', 'rp')):
                b[k] = b[k] * jitter[u, j]
                if k in ('url', 'rep', 'rp'): b[k] = min(b[k], .97)
            lam = b['rate'] * rng.lognormal(0, .45)
            n = rng.poisson(lam)
            if n == 0:
                continue
            # topics of posts
            tp = topic[u].copy()
            if f == 0:
                tp = tp * (1 - .3 * drift) + .3 * drift * np.roll(tp, 1)
            if f == 1 and t >= 5:
                tp = .5 * tp + .5 * topic[rng.choice(legit_idx)]
            if f >= 1:
                tp = rho[u] * tp + (1 - rho[u]) * topic[(u * 31) % P.n_legit]
            pt = rng.choice(NTOP, n, p=tp / tp.sum())
            has_url = rng.random(n) < b['url']
            dom = np.where(rng.random(n) < (.45 * rho[u] if f == 6 else (.9 * rho[u] if f >= 1 else 0)),
                           rng.choice(fam_domains.get(f, fam_domains[1]), n),
                           rng.zipf(1.6, n) % legit_domains)
            if f == 0 and persona[u] == 4:
                dom = (u * 7 + rng.integers(0, 3, n)) % legit_domains     # feed bots reuse own domains
            # URL id: domain + path; spam reuses few paths
            popular = dom * 100000 + (rng.zipf(1.4, n) % 3000)          # shared news articles
            if f >= 1 and f != 6:
                urlid = np.where(dom >= legit_domains, dom * 100000 + rng.integers(0, 40, n), popular)
            elif f == 6:
                urlid = np.where(np.isin(dom, hidden6), dom * 100000 + rng.integers(0, 60, n), popular)
            else:
                urlid = popular
            urlid = np.where(has_url, urlid, -1)
            ntag = rng.poisson(b['tag'], n)
            tags = []
            for i in range(n):
                k = ntag[i]
                tr = rng.random(k) < trend_p
                tg = np.where(tr, rng.choice(trend_tags, k), pt[i] * tags_per_topic + rng.zipf(1.8, k) % tags_per_topic)
                tags.append(tg)
            nmen = rng.poisson(b['men'], n)
            is_rep = rng.random(n) < b['rep']
            is_rp = rng.random(n) < b['rp']
            # timing
            if f == 5:
                base_t = (ring_of[u] * 0.37 + t) % 1.0
                times = (base_t + rng.normal(0, .004, n)) % 1.0
            elif b['reg'] > .7:
                times = (np.arange(n) / n + rng.normal(0, .002, n)) % 1.0
            else:
                bursty = rng.random(n) < b['bur']
                centre = rng.random()
                times = np.where(bursty, (centre + rng.normal(0, .01, n)) % 1.0,
                                 (rng.beta(2.2, 1.6, n)) % 1.0)
            posts_u.append(np.full(n, u)); posts_url.append(urlid); posts_dom.append(np.where(has_url, dom, -1))
            posts_tags.extend(tags); posts_men.append(nmen); posts_rep.append(is_rep); posts_rp.append(is_rp)
            posts_time.append(times); posts_topic.append(pt)
            # per-user features
            hist = np.histogram(times, bins=24, range=(0, 1))[0] + 1e-9
            pr = hist / hist.sum(); ent = -(pr * np.log(pr)).sum() / np.log(24)
            gaps = np.diff(np.sort(times)) if n > 2 else np.array([.5])
            burst = (gaps.std() - gaps.mean()) / (gaps.std() + gaps.mean() + 1e-9)
            dup = np.clip(rng.beta(1 + 8 * b['dup'], 1 + 8 * (1 - b['dup'])), 0, 1)
            lex = np.clip(rng.normal(b['lex'], .08), 0, 1)
            feats[u, 0] = np.log1p(n); feats[u, 1] = np.log1p(is_rp.sum()); feats[u, 2] = np.log1p(is_rep.sum())
            feats[u, 3] = has_url.mean(); feats[u, 4] = ntag.mean(); feats[u, 5] = nmen.mean()
            feats[u, 6] = dup; feats[u, 7] = lex; feats[u, 8] = ent; feats[u, 9] = burst
            feats[u, 10] = np.log1p(age0[u] + 30 * t)
            feats[u, 14] = len(np.unique(dom[has_url])) / max(1, has_url.sum()) if has_url.any() else 1.0
            ntg = ntag.sum()
            feats[u, 15] = (np.concatenate(tags) >= 10000).mean() if ntg > 0 else 0.0
            feats[u, 16:24] = np.bincount(pt, minlength=NTOP) / n
        # follower stats
        alive = active[follow_src] & active[follow_dst]
        fs, fd = follow_src[alive], follow_dst[alive]
        indeg = np.bincount(fd, minlength=N); outdeg = np.bincount(fs, minlength=N)
        indeg = indeg + (fam == 1) * rng.poisson(40, N)                  # bought followers
        feats[:, 11] = np.log1p(indeg); feats[:, 12] = np.log1p(outdeg)
        feats[:, 13] = np.log1p(indeg) - np.log1p(outdeg)
        feats += rng.normal(0, .08, feats.shape).astype(np.float32)
        feats[~active] = 0
        # ---------------- relations ----------------
        pu = np.concatenate(posts_u); purl = np.concatenate(posts_url)
        pmen = np.concatenate(posts_men); prep = np.concatenate(posts_rep); prp = np.concatenate(posts_rp)
        rels = {}
        rels['FOLLOW'] = (fs, fd)
        legit_act = users[y[users] == 0]
        pop_act = popularity[legit_act] / popularity[legit_act].sum()
        # reposts
        src = pu[prp]
        dst = np.empty_like(src)
        for f in np.unique(y[src]):
            m = y[src] == f
            if f == 5:
                g = ring_of[src[m]]
                dst[m] = [rng.choice(ring_groups[gg]) for gg in g]
            elif f == 6:
                dst[m] = np.where(rng.random(m.sum()) < .35, [rng.choice(cell6[c]) for c in cell_of[src[m]]],
                                  rng.choice(legit_act, m.sum(), p=pop_act))
            elif f == 3:
                dst[m] = rng.choice(customers, m.sum())
            else:
                dst[m] = rng.choice(legit_act, m.sum(), p=pop_act)
        rels['REPOST'] = (src, dst)
        # replies
        src = pu[prep]
        dst = np.empty_like(src)
        for f in np.unique(y[src]):
            m = y[src] == f
            if f in (4, 7):
                top = legit_act[np.argsort(-popularity[legit_act])[:150]]
                dst[m] = rng.choice(top, m.sum())
            else:
                dst[m] = rng.choice(legit_act, m.sum(), p=pop_act)
        rels['REPLY'] = (src, dst)
        # mentions
        src = np.repeat(pu, pmen)
        dst = np.empty_like(src)
        for f in np.unique(y[src]):
            m = y[src] == f
            if f in (4, 7):
                top = legit_act[np.argsort(-popularity[legit_act])[:150]]
                dst[m] = rng.choice(top, m.sum())
            elif f == 6:
                dst[m] = np.where(rng.random(m.sum()) < .3, [rng.choice(cell6[c]) for c in cell_of[src[m]]],
                                  rng.choice(legit_act, m.sum(), p=pop_act))
            else:
                dst[m] = rng.choice(legit_act, m.sum(), p=pop_act)
        rels['MENTION'] = (src, dst)
        # co-URL and co-hashtag (bipartite projection with capped pairs)
        def co(ent_u, ent_id, cap=12):
            ok = ent_id >= 0
            eu, ei = ent_u[ok], ent_id[ok]
            order = np.argsort(ei, kind='stable'); eu, ei = eu[order], ei[order]
            bnd = np.r_[0, np.where(np.diff(ei) != 0)[0] + 1, len(ei)]
            s, d = [], []
            for a, b2 in zip(bnd[:-1], bnd[1:]):
                us = np.unique(eu[a:b2])
                if len(us) < 2: continue
                if len(us) > 60: us = rng.choice(us, 60, replace=False)
                for i in range(len(us)):
                    nb = us[(i + 1 + rng.integers(0, len(us) - 1, min(cap, len(us) - 1))) % len(us)]
                    nb = nb[nb != us[i]]
                    s += [us[i]] * len(nb); d += list(nb)
            return np.array(s, int), np.array(d, int)
        rels['CO-URL'] = co(pu, purl)
        tag_u = np.concatenate([np.full(len(tg), u) for u, tg in zip(pu, posts_tags)]) if len(posts_tags) else np.array([], int)
        tag_id = np.concatenate(posts_tags) if len(posts_tags) else np.array([], int)
        nontrend = tag_id < 10000                 # trending tags connect everyone -> drop
        rels['CO-TAG'] = co(tag_u[nontrend], tag_id[nontrend], cap=6)
        windows.append(dict(t=t, X=feats, y=y.copy(), active=active.copy(), rels=rels))
    meta = dict(N=N, family=fam, takeover=takeover, persona=persona, preset=preset)
    return windows, meta


if __name__ == '__main__':
    import time
    t0 = time.time()
    W, M = simulate(0)
    print('N', M['N'], '%.1fs' % (time.time() - t0))
    for w in W:
        a = w['active']; yy = w['y'][a]
        print(w['t'], 'active', a.sum(), 'fam counts', np.bincount(yy, minlength=8),
              {k: len(v[0]) for k, v in w['rels'].items()})
