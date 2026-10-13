import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix


def binary_report(y, score, thr):
    """y: 0 legit / 1 spam, score: spam score, thr: decision threshold (chosen on validation)."""
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    return dict(macro_f1=f1_score(y, pred, average='macro', zero_division=0),
                pr_auc=average_precision_score(y, score) if y.any() else np.nan,
                auroc=roc_auc_score(y, score) if 0 < y.mean() < 1 else np.nan,
                mcc=matthews_corrcoef(y, pred), gmean=np.sqrt(tpr * tnr), bacc=(tpr + tnr) / 2,
                tpr=tpr, fpr=1 - tnr)


def best_threshold(y, score):
    """threshold maximising macro-F1 on validation data."""
    qs = np.unique(np.quantile(score, np.linspace(0.5, 0.999, 300)))
    best, bt = -1, 0.5
    for t in qs:
        f = f1_score(y, (score >= t).astype(int), average='macro', zero_division=0)
        if f > best: best, bt = f, t
    return bt


def ece(y, p, bins=15):
    """expected calibration error of the spam probability (binary reliability)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    conf = np.maximum(p, 1 - p); pred = (p >= .5).astype(int); acc = (pred == y)
    edges = np.linspace(.5, 1, bins + 1); e = 0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def brier(y, p):
    return np.mean((p - y) ** 2)


def nll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def aurc(err, conf):
    """area under the risk-coverage curve (lower is better)."""
    o = np.argsort(-conf); e = err[o].astype(float)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return risk.mean()


def risk_at_coverage(err, conf, cov):
    o = np.argsort(-conf); k = max(1, int(round(cov * len(o))))
    return err[o[:k]].mean()
