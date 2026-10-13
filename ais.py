"""Classical and artificial-immune baselines.
NSA  : real-valued negative selection with variable-radius detectors (V-detector).
NSAGA: negative selection whose detector centres are evolved by a genetic algorithm
       to cover labelled non-self samples while remaining self-tolerant."""
import numpy as np, time
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score


class Classic:
    def __init__(self, kind, seed=0, fast=False):
        self.kind = kind; self.seed = seed; self.fast = fast

    def _feat(self, D, idx):
        if self.kind == 'gbdt':
            return D.T[idx].reshape(len(idx), -1)
        return D.X[idx]

    def fit(self, D, tr, va):
        y = D.y[tr]
        if self.kind == 'lr':
            self.m = LogisticRegression(class_weight='balanced', max_iter=3000, C=1.0)
        elif self.kind == 'rf':
            self.m = RandomForestClassifier(300, class_weight='balanced_subsample', n_jobs=2,
                                            min_samples_leaf=2, random_state=self.seed)
        else:
            self.m = HistGradientBoostingClassifier(max_iter=120 if self.fast else 400, learning_rate=0.1 if self.fast else 0.06,
                                                    max_leaf_nodes=31, max_bins=64 if self.fast else 255,
                                                    class_weight='balanced', early_stopping=True,
                                                    validation_fraction=0.15, random_state=self.seed)
        if self.kind == 'gbdt' and (np.bincount(y)[np.bincount(y) > 0].min() < 5):
            self.m.set_params(early_stopping=False, max_iter=min(self.m.max_iter, 150))
        self.m.fit(self._feat(D, tr), y)
        return self

    def proba(self, D, idx):
        p = self.m.predict_proba(self._feat(D, idx))
        C = max(2, int(np.max(self.m.classes_)) + 1)
        out = np.zeros((len(idx), C)); out[:, self.m.classes_.astype(int)] = p
        return out

    def n_params(self):
        if self.kind == 'lr': return int(self.m.coef_.size + self.m.intercept_.size)
        if self.kind == 'rf': return int(sum(e.tree_.node_count for e in self.m.estimators_))
        return int(sum(p.get_n_leaf_nodes() * 2 for it in self.m._predictors for p in it))


class _Space:
    """PCA projection of flattened tokens, min-max scaled to the unit hypercube."""
    def __init__(self, D, tr, k, seed):
        Z = D.T[tr].reshape(len(tr), -1)
        self.pca = PCA(k, random_state=seed).fit(Z)
        P = self.pca.transform(Z)
        self.lo, self.hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)

    def __call__(self, D, idx):
        P = self.pca.transform(D.T[idx].reshape(len(idx), -1))
        return np.clip((P - self.lo) / (self.hi - self.lo + 1e-9), 0, 1)


class NSA:
    """V-detector negative selection (variable radius); detectors are trained on self only."""
    def __init__(self, seed=0, k=8, max_det=2000, target_cov=0.995, ga=False):
        self.seed = seed; self.k = k; self.max_det = max_det; self.cov = target_cov; self.ga = ga

    def _detectors_random(self, rng, nn_self, r_s):
        C, R = [], []
        covered_streak = 0; tries = 0
        while len(C) < self.max_det and tries < 60 * self.max_det:
            x = rng.random(self.k); tries += 1
            if C:
                dd = np.linalg.norm(np.asarray(C) - x, axis=1)
                if (dd < np.asarray(R)).any():
                    covered_streak += 1
                    if covered_streak > 1 / (1 - self.cov):
                        break
                    continue
            covered_streak = 0
            d = nn_self.kneighbors(x[None], 1)[0][0, 0]
            if d > r_s:
                C.append(x); R.append(d - r_s)
        return np.asarray(C), np.asarray(R)

    def _detectors_ga(self, rng, nn_self, r_s, Ns, pop=120, gens=40):
        """GA: individuals are detector centres; fitness = newly covered non-self
        samples (niche-shared) with self-tolerant radius d_self - r_s."""
        dim = self.k
        def radius(X):
            return np.maximum(nn_self.kneighbors(X, 1)[0][:, 0] - r_s, 0)
        # seed population around labelled non-self samples
        X = np.clip(Ns[rng.choice(len(Ns), pop)] + rng.normal(0, .02, (pop, dim)), 0, 1)
        archive_C, archive_R = [], []
        covered = np.zeros(len(Ns), bool)
        for g in range(gens):
            R = radius(X)
            dist = np.linalg.norm(Ns[None] - X[:, None], axis=2)          # pop x Nns
            cov = dist < R[:, None]
            new = (cov & ~covered[None]).sum(1)
            fit = new + 0.1 * cov.sum(1) - 5.0 * (R <= 0)
            # elitist archive: add the best detector of the generation if it covers new non-self
            b = np.argmax(fit)
            if new[b] > 0 and R[b] > 0:
                archive_C.append(X[b].copy()); archive_R.append(R[b]); covered |= cov[b]
            # tournament selection + blend crossover + gaussian mutation
            i1, i2 = rng.integers(0, pop, (2, pop))
            par = np.where((fit[i1] >= fit[i2])[:, None], X[i1], X[i2])
            j1, j2 = rng.integers(0, pop, (2, pop))
            mate = np.where((fit[j1] >= fit[j2])[:, None], X[j1], X[j2])
            a = rng.random((pop, 1))
            child = a * par + (1 - a) * mate
            child += rng.normal(0, .03, child.shape) * (rng.random(child.shape) < .3)
            # re-inject uncovered non-self samples (keeps diversity)
            unc = np.where(~covered)[0]
            if len(unc):
                m = rng.random(pop) < .2
                child[m] = Ns[rng.choice(unc, m.sum())] + rng.normal(0, .01, (m.sum(), dim))
            X = np.clip(child, 0, 1)
            X[0] = archive_C[-1] if archive_C else X[0]
        # complete with random self-tolerant detectors for the remaining non-self space (small budget)
        C2, R2 = self._detectors_random(rng, nn_self, r_s) if len(archive_C) < 20 else (np.zeros((0, dim)), np.zeros(0))
        C = np.vstack([np.asarray(archive_C).reshape(-1, dim), C2]); R = np.r_[np.asarray(archive_R), R2]
        return C, R

    def fit(self, D, tr, va):
        rng = np.random.default_rng(self.seed)
        self.space = _Space(D, tr, self.k, self.seed)
        Z = self.space(D, tr); y = D.y[tr]
        S = Z[y == 0]
        if len(S) > 4000: S = S[rng.choice(len(S), 4000, replace=False)]
        nn_self = NearestNeighbors(n_neighbors=2).fit(S)
        med = np.median(nn_self.kneighbors(S, 2)[0][:, 1])
        Zv = self.space(D, va); yv = (D.y[va] > 0).astype(int)
        best = (-1, None)
        for mult in (1.0, 2.0, 4.0):      # self radius chosen on validation
            r_s = mult * med
            if self.ga:
                C, R = self._detectors_ga(rng, nn_self, r_s, Z[y > 0])
            else:
                C, R = self._detectors_random(rng, nn_self, r_s)
            if len(C) == 0: continue
            sc = self._score(Zv, C, R)
            ap = average_precision_score(yv, sc)
            if ap > best[0]: best = (ap, (C, R, r_s))
        self.C, self.R, self.r_s = best[1]
        return self

    @staticmethod
    def _score(Z, C, R, bs=2048):
        out = []
        for s in range(0, len(Z), bs):
            d = np.linalg.norm(Z[s:s + bs, None] - C[None], axis=2)
            out.append((R[None] - d).max(1))
        return np.concatenate(out)

    def proba(self, D, idx):
        s = self._score(self.space(D, idx), self.C, self.R)
        p = 1 / (1 + np.exp(-s / (np.std(s) + 1e-9) * 3))
        return np.c_[1 - p, p]

    def n_params(self):
        return int(self.C.size + self.R.size)
