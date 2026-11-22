"""Evaluate gap-reduction methods with the same protocol as B0 (single-source 4x4 + LODO).

Every run re-fits the B0 probe (variant 'b0') so each method is compared to B0 on identical splits.

--method tta   M3, label-free test-time adaptation on frozen features (CPU only)
    b0            : source-standardised features -> logistic regression
    featnorm      : standardise each target set with its *own* (unlabelled) mean/std
    t3a           : T3A prototypes (Iwasawa & Matsuo, NeurIPS'21), batch version: class prototype =
                    mean of {normalised classifier weight} U {K lowest-entropy target samples
                    pseudo-labelled as that class}; temperature fitted on source val
    featnorm_t3a  : both
    t3a_ens       : equal-weight log-linear ensemble of b0 and t3a (keeps in-domain accuracy)
--method text  M2, vision-language zero-shot head + linear probe ensemble (needs outputs/emb/)
    zs_plain / zs_definition : zero-shot with class-name / recycling-definition prompts
    lp            : linear probe on the same (projected) embeddings
    ens_<set>     : log-linear ensemble  (1-a)*log p_lp + a*log p_zs,  a fixed a priori = 0.5
    ens_definition_a{25,75} : sensitivity to a
--method clad  M1, CLAD-Aug: probe trained on originals + composites (needs outputs/feats_<tag>/)
    b0, clad_<tag> for every --tags entry

Usage: python run_methods.py --out outputs --method tta --seeds 0 1 2 [--backbones ...]
Output: outputs/results_methods/<method>/seed<k>/{probe_results.csv,predictions.csv.gz}
        outputs/results_methods/<method>/summary.csv  (mean ± std over seeds)
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import softmax, log_softmax
from sklearn.metrics import accuracy_score, f1_score
from common import CORE, CORE_TO_ID, DATASET_NAMES, make_split
from probe import fit_probe, ece_score, load_kept

PROTOCOLS = [("single", [d]) for d in DATASET_NAMES] + \
            [("lodo", [s for s in DATASET_NAMES if s != d]) for d in DATASET_NAMES]
T3A_K = 100   # filter size, T3A default


def metrics(prob, y):
    pred = prob.argmax(1)
    return {"acc": accuracy_score(y, pred) * 100,
            "macro_f1": f1_score(y, pred, average="macro", labels=range(len(CORE))) * 100,
            "ece": ece_score(prob, y) * 100, "n_test": len(y)}, pred


def fit_temperature(logits, y):
    best = (np.inf, 1.0)
    for T in np.logspace(-2, 2, 80):
        nll = -log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean()
        if nll < best[0]: best = (nll, T)
    return best[1]


def t3a_logits(Z, base_logits, W, K=T3A_K):
    p = softmax(base_logits, axis=1)
    yhat, ent = p.argmax(1), -(p * np.log(p + 1e-12)).sum(1)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    protos = []
    for c in range(W.shape[0]):
        sup = [W[c] / np.linalg.norm(W[c])]
        idx = np.where(yhat == c)[0]
        idx = idx[np.argsort(ent[idx])][:K]
        protos.append(np.vstack(sup + [Zn[idx]]).mean(0))
    P = np.stack(protos); P /= np.linalg.norm(P, axis=1, keepdims=True)
    return Z @ P.T


def standardize_self(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


# ------------------------------------------------------------------ per (backbone, seed)
def run_one(method, name, seed, out, man, tags):
    t0 = time.time()
    src_file = out / ("emb" if method == "text" else "feats") / f"{name}.npz"
    z = np.load(src_file)
    X, row_id = load_kept(z, man)
    m = man.loc[row_id]
    ds, y = m.dataset.to_numpy(), m.label.map(CORE_TO_ID).to_numpy()
    split = make_split(ds, y, seed=seed)
    pos = {r: i for i, r in enumerate(row_id)}

    extra = {}
    if method == "clad":
        for tag in tags:
            zc = np.load(out / f"feats_{tag}" / f"{name}.npz")
            keep = np.array([r in pos for r in zc["row_id"]])
            extra[tag] = (zc["feats"][keep].astype(np.float32),
                          np.array([pos[r] for r in zc["row_id"][keep]]))
    if method == "text":
        scale = float(z["logit_scale"])
        text = {k[5:]: z[k] for k in z.files if k.startswith("text_")}

    rows, preds = [], []

    def record(variant, proto, srcs, tgt, prob, te):
        mt, pred = metrics(prob, y[te])
        rows.append({"method": method, "variant": variant, "backbone": name, "seed": seed,
                     "protocol": proto, "train_on": "+".join(srcs), "test_on": tgt,
                     "in_domain": int(tgt in srcs), **mt})
        preds.append(pd.DataFrame({"variant": variant, "backbone": name, "protocol": proto,
                                   "train_on": "+".join(srcs), "test_on": tgt, "row_id": row_id[te],
                                   "y_true": np.array(CORE)[y[te]], "y_pred": np.array(CORE)[pred],
                                   "conf": prob.max(1).round(4)}))

    for proto, srcs in PROTOCOLS:
        src = np.isin(ds, srcs)
        tr, va = src & (split == "train"), src & (split == "val")
        sc, clf, C, _ = fit_probe(X[tr], y[tr], X[va], y[va], seed)
        targets = DATASET_NAMES if proto == "single" else [d for d in DATASET_NAMES if d not in srcs]
        tests = {t: (ds == t) & (split == "test") for t in targets}

        if method == "tta":
            W = clf.coef_
            Zva = sc.transform(X[va])
            T_t3a = fit_temperature(t3a_logits(Zva, clf.decision_function(Zva), W), y[va])
            T_fn = fit_temperature(t3a_logits(standardize_self(X[va]),
                                              clf.decision_function(standardize_self(X[va])), W), y[va])
            for tgt, te in tests.items():
                Zs = sc.transform(X[te]); Zt = standardize_self(X[te])
                ls, lt = clf.decision_function(Zs), clf.decision_function(Zt)
                record("b0", proto, srcs, tgt, softmax(ls, 1), te)
                record("featnorm", proto, srcs, tgt, softmax(lt, 1), te)
                p_t3a = softmax(t3a_logits(Zs, ls, W) / T_t3a, 1)
                record("t3a", proto, srcs, tgt, p_t3a, te)
                record("featnorm_t3a", proto, srcs, tgt, softmax(t3a_logits(Zt, lt, W) / T_fn, 1), te)
                # ensemble of source probe and T3A (log-linear, equal weights, fixed a priori)
                p_ens = softmax(0.5 * np.log(softmax(ls, 1) + 1e-12) + 0.5 * np.log(p_t3a + 1e-12), 1)
                record("t3a_ens", proto, srcs, tgt, p_ens, te)

        elif method == "text":
            for tgt, te in tests.items():
                p_lp = clf.predict_proba(sc.transform(X[te]))
                record("lp", proto, srcs, tgt, p_lp, te)
                for sname, Wt in text.items():
                    p_zs = softmax(scale * X[te] @ Wt.T, 1)
                    record(f"zs_{sname}", proto, srcs, tgt, p_zs, te)
                    alphas = [(0.5, "")] + ([(0.25, "_a25"), (0.75, "_a75")] if sname == "definition" else [])
                    for a, suf in alphas:
                        lp = (1 - a) * np.log(p_lp + 1e-12) + a * np.log(p_zs + 1e-12)
                        record(f"ens_{sname}{suf}", proto, srcs, tgt, softmax(lp, 1), te)

        elif method == "clad":
            for tgt, te in tests.items():
                record("b0", proto, srcs, tgt, clf.predict_proba(sc.transform(X[te])), te)
            for tag, (Fc, idx) in extra.items():
                use = tr[idx]                                   # composites of source-train images only
                Xtr = np.vstack([X[tr], Fc[use]]); ytr = np.concatenate([y[tr], y[idx[use]]])
                sc2, clf2, _, _ = fit_probe(Xtr, ytr, X[va], y[va], seed)
                for tgt, te in tests.items():
                    record(f"clad_{tag}", proto, srcs, tgt, clf2.predict_proba(sc2.transform(X[te])), te)
    print(f"[{method}] {name} seed{seed} done ({time.time() - t0:.0f}s)", flush=True)
    return rows, pd.concat(preds, ignore_index=True)


def summarize(df):
    out = []
    for (bb, var), g in df.groupby(["backbone", "variant"]):
        per_seed = []
        for s, h in g.groupby("seed"):
            sg, lo = h[h.protocol == "single"], h[h.protocol == "lodo"]
            ind, ood = sg[sg.in_domain == 1].macro_f1.mean(), sg[sg.in_domain == 0].macro_f1.mean()
            t2r = sg[(sg.train_on == "TrashNet") & (sg.test_on == "RealWaste")].macro_f1.mean()
            per_seed.append({"in_domain_F1": ind, "ood_F1": ood, "gap": ind - ood,
                             "TrashNet->RealWaste_F1": t2r, "LODO_avg_F1": lo.macro_f1.mean(),
                             "ood_ECE": sg[sg.in_domain == 0].ece.mean()})
        p = pd.DataFrame(per_seed)
        row = {"backbone": bb, "variant": var, "n_seeds": len(p)}
        for c in p.columns:
            row[c] = f"{p[c].mean():.1f} ± {p[c].std(ddof=1) if len(p) > 1 else 0:.1f}"
            row[c + "_mean"] = p[c].mean()
        out.append(row)
    s = pd.DataFrame(out)
    ref = s[s.variant.isin(["b0", "lp"])].set_index("backbone")
    s["d_ood_vs_base"] = s.apply(lambda r: r.ood_F1_mean - ref.loc[r.backbone].ood_F1_mean
                                 if r.backbone in ref.index else np.nan, axis=1).round(2)
    s["d_lodo_vs_base"] = s.apply(lambda r: r.LODO_avg_F1_mean - ref.loc[r.backbone].LODO_avg_F1_mean
                                  if r.backbone in ref.index else np.nan, axis=1).round(2)
    return s.sort_values(["backbone", "ood_F1_mean"], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--method", required=True, choices=["tta", "text", "clad"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--backbones", nargs="+", default=None)
    ap.add_argument("--tags", nargs="+", default=["clad_c1", "clad_full"])
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    man = pd.read_csv(out / "manifest.csv")
    default_bb = {"tta": ["resnet50", "dinov2_b", "siglip_b16"], "text": ["clip_b16", "siglip_b16"],
                  "clad": ["resnet50", "dinov2_b", "siglip_b16"]}[args.method]
    names = args.backbones or default_bb
    res_root = out / "results_methods" / args.method

    jobs = [(n, s) for s in args.seeds for n in names]
    results = Parallel(n_jobs=min(args.jobs, len(jobs)))(
        delayed(run_one)(args.method, n, s, out, man, args.tags) for n, s in jobs)
    for s in args.seeds:
        d = res_root / f"seed{s}"; d.mkdir(parents=True, exist_ok=True)
        sel = [r for (n, ss), r in zip(jobs, results) if ss == s]
        pd.DataFrame([x for rs, _ in sel for x in rs]).to_csv(d / "probe_results.csv", index=False)
        pd.concat([p for _, p in sel]).to_csv(d / "predictions.csv.gz", index=False, compression="gzip")

    all_df = pd.concat([pd.read_csv(f) for f in sorted(res_root.glob("seed*/probe_results.csv"))])
    summ = summarize(all_df)
    summ.to_csv(res_root / "summary.csv", index=False)
    cols = ["backbone", "variant", "n_seeds", "in_domain_F1", "ood_F1", "gap",
            "TrashNet->RealWaste_F1", "LODO_avg_F1", "ood_ECE", "d_ood_vs_base", "d_lodo_vs_base"]
    print(f"\n=== {args.method}: macro-F1 %, mean ± std over seeds ===")
    print(summ[cols].to_string(index=False))


if __name__ == "__main__":
    main()
