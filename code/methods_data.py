"""Data-driven methods M7-M9 (single backbone, CPU, frozen features). Called from run_methods.py.

Each method answers one problem measured in the data analysis:

M7 'erase'    PROBLEM: features encode *which dataset / how the photo was taken* (dataset identity is
              linearly decodable at 93-98%; same-class centroids of different datasets are as far apart
              as different classes). FIX: LEACE (Belrose et al., NeurIPS'23) closed-form linear erasure of
              nuisance information before the probe, fitted on source-train only.
                erase_nuis : erase pixel-level nuisance descriptors (background colour/texture, resolution)
                erase_ds   : erase source-dataset identity (multi-source/LODO only)
                erase_both : both
M8 'prior'    PROBLEM: class priors differ between datasets (e.g. glass 47% in Garbage12 vs 14-21% elsewhere)
              and the predicted class distribution on targets is biased (total-variation up to 14 pts).
              FIX: label-shift adaptation, label-free: Bias-Corrected Temperature Scaling on source val
              + EM prior re-estimation on the unlabelled target (Saerens'02; Alexandari et al., ICML'20).
                bcts   : calibration only        em : TS + EM        bcts_em : BCTS + EM
M9 'subclass' PROBLEM: classes are multi-modal across datasets ('paper' = sheets, crumpled texture, cups,
              newspaper; largest cross-dataset centroid spread and worst OOD recall). One linear boundary per
              class favours the dominant source mode. FIX: cluster every class of the source-train set into
              K sub-concepts (k-means), train the probe on sub-classes with balanced sub-class weights
              (GEORGE-style, Sohoni et al., NeurIPS'20), sum sub-class probabilities per class.
                subK2 / subK4 / subK8
"""
import time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax, log_softmax
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from common import CORE, CORE_TO_ID, DATASET_NAMES, make_split
from probe import fit_probe, load_kept

NC = len(CORE)


# ----------------------------------------------------------------------------- M7: LEACE
class Leace:
    """x' = x - P (x - mu),  P = W^+ P_{W Sxz} W,  W = Sxx^{-1/2}  (Belrose et al. 2023, Thm 4.3)."""
    def __init__(self, X, Z, eps=1e-6):
        self.mu = X.mean(0)
        Xc, Zc = X - self.mu, Z - Z.mean(0)
        Sxx = Xc.T @ Xc / len(X)
        Sxz = Xc.T @ Zc / len(X)
        lam, V = np.linalg.eigh(Sxx)
        keep = lam > eps * lam.max()
        V, lam = V[:, keep], lam[keep]
        W = (V / np.sqrt(lam)) @ V.T
        W_pinv = (V * np.sqrt(lam)) @ V.T
        A = W @ Sxz
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        U = U[:, s > 1e-8 * s.max()] if s.size else U[:, :0]
        self.P = W_pinv @ (U @ U.T) @ W

    def __call__(self, X):
        return X - (X - self.mu) @ self.P.T


# ----------------------------------------------------------------------------- M8: label shift
def fit_bcts(logits, y, bias=True):
    """Minimise NLL over temperature T (and class biases b if bias=True)."""
    k = logits.shape[1]

    def nll(p):
        T, b = np.exp(p[0]), (p[1:] if bias else np.zeros(k))
        return -log_softmax(logits / T + b, 1)[np.arange(len(y)), y].mean()
    p0 = np.zeros(1 + (k if bias else 0))
    p = minimize(nll, p0, method="L-BFGS-B").x
    return np.exp(p[0]), (p[1:] if bias else np.zeros(k))


def em_prior(prob, src_prior, iters=200, tol=1e-7):
    """Saerens et al. 2002: EM estimate of target priors, returns adapted posteriors."""
    pi = src_prior.copy()
    for _ in range(iters):
        w = prob * (pi / src_prior)
        w /= w.sum(1, keepdims=True)
        new = w.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    w = prob * (pi / src_prior)
    return w / w.sum(1, keepdims=True), pi


# ----------------------------------------------------------------------------- shared
def _record(rows, preds, method, var, name, seed, proto, srcs, tgt, prob, y_te, rid_te):
    from run_methods import metrics
    mt, pred = metrics(prob, y_te)
    rows.append({"method": method, "variant": var, "backbone": name, "seed": seed, "protocol": proto,
                 "train_on": "+".join(srcs), "test_on": tgt, "in_domain": int(tgt in srcs), **mt})
    preds.append(pd.DataFrame({"variant": var, "backbone": name, "protocol": proto, "train_on": "+".join(srcs),
                               "test_on": tgt, "row_id": rid_te, "y_true": np.array(CORE)[y_te],
                               "y_pred": np.array(CORE)[pred], "conf": prob.max(1).round(4)}))


def run_data_method(method, name, seed, out, man):
    from run_methods import PROTOCOLS
    t0 = time.time()
    X, row_id = load_kept(np.load(out / "feats" / f"{name}.npz"), man)
    m = man.loc[row_id]
    ds, y = m.dataset.to_numpy(), m.label.map(CORE_TO_ID).to_numpy()
    split = make_split(ds, y, seed=seed)
    if method == "erase":
        zn = np.load(out / "nuisance.npz")
        pos = {r: i for i, r in enumerate(zn["row_id"])}
        N = zn["Z"][[pos[r] for r in row_id]].astype(np.float64)
    rows, preds = [], []
    for proto, srcs in PROTOCOLS:
        src = np.isin(ds, srcs)
        tr, va = src & (split == "train"), src & (split == "val")
        targets = DATASET_NAMES if proto == "single" else [d for d in DATASET_NAMES if d not in srcs]
        tests = {t: (ds == t) & (split == "test") for t in targets}
        rec = lambda var, tgt, prob: _record(rows, preds, method, var, name, seed, proto, srcs, tgt,
                                             prob, y[tests[tgt]], row_id[tests[tgt]])
        sc, clf, C, _ = fit_probe(X[tr], y[tr], X[va], y[va], seed)
        for tgt, te in tests.items():
            rec("b0", tgt, clf.predict_proba(sc.transform(X[te])))

        if method == "erase":
            Xs = StandardScaler().fit(X[tr])
            Xtr = Xs.transform(X[tr]).astype(np.float64)
            Zsets = {"erase_nuis": N[tr]}
            if len(srcs) > 1:
                D = np.stack([(ds[tr] == d).astype(float) for d in srcs[:-1]], 1)   # dataset one-hot (k-1)
                Zsets["erase_ds"] = D
                Zsets["erase_both"] = np.hstack([N[tr], D])
            for var, Z in Zsets.items():
                Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
                er = Leace(Xtr, Zs)
                f = lambda A: er(Xs.transform(A).astype(np.float64))
                sc2, clf2, _, _ = fit_probe(f(X[tr]), y[tr], f(X[va]), y[va], seed)
                for tgt, te in tests.items():
                    rec(var, tgt, clf2.predict_proba(sc2.transform(f(X[te]))))

        elif method == "prior":
            lv = clf.decision_function(sc.transform(X[va]))
            T_ts, _ = fit_bcts(lv, y[va], bias=False)
            T_b, b_b = fit_bcts(lv, y[va], bias=True)
            for var, (T, b) in {"ts": (T_ts, 0), "bcts": (T_b, b_b)}.items():
                src_prior = softmax(lv / T + b, 1).mean(0)       # source prior under calibrated model
                for tgt, te in tests.items():
                    p = softmax(clf.decision_function(sc.transform(X[te])) / T + b, 1)
                    if var == "bcts": rec("bcts", tgt, p)
                    rec(f"{var}_em", tgt, em_prior(p, src_prior)[0])

        elif method == "subclass":
            Xs = StandardScaler().fit(X[tr]); Ztr, Zva = Xs.transform(X[tr]), Xs.transform(X[va])
            for K in (2, 4, 8):
                sub = np.zeros(tr.sum(), int)
                for c in range(NC):
                    idx = np.where(y[tr] == c)[0]
                    k = min(K, max(1, len(idx) // 20))
                    lab = KMeans(k, n_init=4, random_state=seed).fit_predict(Ztr[idx]) if k > 1 else np.zeros(len(idx), int)
                    sub[idx] = c * K + lab
                best = None
                for Cr in [0.001, 0.01, 0.1, 1.0]:
                    clf_s = LogisticRegression(C=Cr, max_iter=3000, class_weight="balanced", random_state=seed).fit(Ztr, sub)
                    pv = _collapse(clf_s.predict_proba(Zva), clf_s.classes_, K)
                    f1 = f1_score(y[va], pv.argmax(1), average="macro")
                    if best is None or f1 > best[0]: best = (f1, clf_s)
                clf_s = best[1]
                for tgt, te in tests.items():
                    rec(f"subK{K}", tgt, _collapse(clf_s.predict_proba(Xs.transform(X[te])), clf_s.classes_, K))
    print(f"[{method}] {name} seed{seed} done ({time.time() - t0:.0f}s)", flush=True)
    return rows, pd.concat(preds, ignore_index=True)


def run_data_method_v2(method, name, seed, out, man):
    """N1 'reweight' : break the within-dataset background->class shortcut by inverse-propensity weights
                       w_i = 1 / p(y_i | acquisition descriptors_i)  (cross-fitted, clipped).
       N2 'cluster'  : SHOT-style (Liang et al., ICML'20) source-initialised centroid refinement on the
                       unlabelled target set; temperature fitted on source val.
       N3 'labelprop': label propagation (Zhou et al., NeurIPS'04) of source-probe predictions over the
                       target kNN graph."""
    from run_methods import PROTOCOLS, fit_temperature
    from sklearn.model_selection import StratifiedKFold
    t0 = time.time()
    X, row_id = load_kept(np.load(out / "feats" / f"{name}.npz"), man)
    m = man.loc[row_id]
    ds, y = m.dataset.to_numpy(), m.label.map(CORE_TO_ID).to_numpy()
    split = make_split(ds, y, seed=seed)
    if method == "reweight":
        zn = np.load(out / "nuisance.npz"); pos = {r: i for i, r in enumerate(zn["row_id"])}
        N = zn["Z"][[pos[r] for r in row_id]].astype(np.float64)
        N = N[:, list(range(10)) + [N.shape[1] - 2, N.shape[1] - 1]] if N.shape[1] > 12 else N
    rows, preds = [], []
    for proto, srcs in PROTOCOLS:
        src = np.isin(ds, srcs)
        tr, va = src & (split == "train"), src & (split == "val")
        targets = DATASET_NAMES if proto == "single" else [d for d in DATASET_NAMES if d not in srcs]
        tests = {t: (ds == t) & (split == "test") for t in targets}
        rec = lambda var, tgt, prob: _record(rows, preds, method, var, name, seed, proto, srcs, tgt,
                                             prob, y[tests[tgt]], row_id[tests[tgt]])
        sc, clf, C, _ = fit_probe(X[tr], y[tr], X[va], y[va], seed)
        base = {t: clf.decision_function(sc.transform(X[te])) for t, te in tests.items()}
        for tgt in tests: rec("b0", tgt, softmax(base[tgt], 1))

        if method == "reweight":
            Ntr, ytr = N[tr], y[tr]
            ph = np.zeros(len(ytr))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Ntr, ytr):
                s2 = StandardScaler().fit(Ntr[a])
                nc = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s2.transform(Ntr[a]), ytr[a])
                ph[b] = nc.predict_proba(s2.transform(Ntr[b]))[np.arange(len(b)), ytr[b]]
            for var, clip in [("rw", 5.0), ("rw_c10", 10.0)]:
                w = np.clip((1 / NC) / np.maximum(ph, 1e-6), 1 / clip, clip)
                cw = len(ytr) / (NC * np.bincount(ytr, minlength=NC))            # keep class balance
                w = w * cw[ytr]; w /= w.mean()
                Ztr, Zva = sc.transform(X[tr]), sc.transform(X[va])
                best = None
                for Cr in [0.001, 0.01, 0.1, 1.0]:
                    c2 = LogisticRegression(C=Cr, max_iter=3000, random_state=seed).fit(Ztr, ytr, sample_weight=w)
                    f1 = f1_score(y[va], c2.predict(Zva), average="macro")
                    if best is None or f1 > best[0]: best = (f1, c2)
                for tgt, te in tests.items():
                    rec(var, tgt, best[1].predict_proba(sc.transform(X[te])))

        elif method == "cluster":
            Zva = sc.transform(X[va])
            T = fit_temperature(shot_logits(Zva, clf.decision_function(Zva)), y[va])
            for tgt, te in tests.items():
                ps = softmax(shot_logits(sc.transform(X[te]), base[tgt]) / T, 1)
                rec("shot", tgt, ps)
                rec("shot_ens", tgt, softmax(0.5 * log_softmax(base[tgt], 1) + 0.5 * np.log(ps + 1e-12), 1))

        elif method == "labelprop":
            Zva = sc.transform(X[va]); p0va = softmax(clf.decision_function(Zva), 1)
            # (k, alpha) chosen on the labelled *source* validation set (transductively), no target labels
            best = None
            for k in (5, 10, 20):
                for a in (0.3, 0.5, 0.7, 0.9):
                    f1 = f1_score(y[va], label_prop(Zva, p0va, k, a).argmax(1), average="macro")
                    if best is None or f1 > best[0] + 1e-9: best = (f1, k, a)
            _, k_s, a_s = best
            T_lp = fit_temperature(np.log(label_prop(Zva, p0va, k_s, a_s) + 1e-12), y[va])
            T_sh = fit_temperature(shot_logits(Zva, clf.decision_function(Zva)), y[va])
            for tgt, te in tests.items():
                Z = sc.transform(X[te]); p0 = softmax(base[tgt], 1)
                rec("lprop_a05", tgt, label_prop(Z, p0, 10, 0.5))
                rec("lprop_a08", tgt, label_prop(Z, p0, 10, 0.8))
                rec("lprop_auto", tgt, softmax(np.log(label_prop(Z, p0, k_s, a_s) + 1e-12) / T_lp, 1))
                # cluster-then-propagate: seed the graph with SHOT-refined (ensembled) posteriors
                ps = softmax(shot_logits(Z, base[tgt]) / T_sh, 1)
                p_se = softmax(0.5 * log_softmax(base[tgt], 1) + 0.5 * np.log(ps + 1e-12), 1)
                rec("shot_lprop_auto", tgt, softmax(np.log(label_prop(Z, p_se, k_s, a_s) + 1e-12) / T_lp, 1))
                rows[-1]["sel_k"], rows[-1]["sel_alpha"] = k_s, a_s
    print(f"[{method}] {name} seed{seed} done ({time.time() - t0:.0f}s)", flush=True)
    return rows, pd.concat(preds, ignore_index=True)


def shot_logits(Z, logits, iters=2):
    """SHOT centroid refinement: soft class centroids from source posteriors, then nearest-centroid
    hard relabelling; returns cosine similarities to the refined centroids."""
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    p = softmax(logits, 1)
    for _ in range(iters):
        Cc = (p.T @ Zn) / (p.sum(0)[:, None] + 1e-8)
        Cc /= np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-8
        p = np.eye(NC)[(Zn @ Cc.T).argmax(1)]
    return Zn @ Cc.T


def label_prop(Z, p0, k=10, alpha=0.8, iters=30):
    """Zhou et al. 2004: F <- alpha S F + (1-alpha) P0 on the symmetric-normalised kNN cosine graph."""
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    S = Zn @ Zn.T; np.fill_diagonal(S, -np.inf)
    kk = min(k, len(Z) - 1)
    idx = np.argpartition(-S, kk, axis=1)[:, :kk]
    Wm = np.zeros_like(S); r = np.arange(len(Z))[:, None]
    Wm[r, idx] = np.maximum(S[r, idx], 0); Wm = np.maximum(Wm, Wm.T)
    d = Wm.sum(1); d[d == 0] = 1; Sn = Wm / np.sqrt(d[:, None] * d[None])
    F = p0.copy()
    for _ in range(iters): F = alpha * Sn @ F + (1 - alpha) * p0
    return F / F.sum(1, keepdims=True)


def _collapse(p_sub, classes, K):
    out = np.zeros((len(p_sub), NC))
    for j, s in enumerate(classes):
        out[:, s // K] += p_sub[:, j]
    return out
