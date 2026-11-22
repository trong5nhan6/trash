"""Step 3: baseline B0 = linear probe on frozen features. One seed per run.

Protocols
  single : train on one dataset's train split, test on the test split of all 4 datasets (4x4 matrix)
  lodo   : train on the other 3 datasets' train splits, test on the held-out dataset's test split
The seed controls the train/val/test split. C is chosen on the *source* val split only (no target leakage).

Usage:  python probe.py --out outputs --seed 0 [--backbones resnet50 dinov2_b ...]
Output: outputs/results/seed<k>/probe_results.csv   (one row per backbone x protocol x target)
        outputs/results/seed<k>/matrix_<metric>_<backbone>.csv  (4x4, rows=train, cols=test)
        outputs/results/seed<k>/table2a_summary.csv
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from common import CORE, CORE_TO_ID, DATASET_NAMES, make_split

C_GRID = [0.001, 0.01, 0.1, 1.0]


def ece_score(prob, y, n_bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def fit_probe(Xtr, ytr, Xva, yva, seed):
    """Standardize + multinomial logistic regression; pick C by macro-F1 on source val."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    return sc, best[2], best[1], best[0]


def evaluate(sc, clf, X, y):
    prob = clf.predict_proba(sc.transform(X))
    pred = prob.argmax(1)
    return {"acc": accuracy_score(y, pred) * 100,
            "macro_f1": f1_score(y, pred, average="macro", labels=range(len(CORE))) * 100,
            "ece": ece_score(prob, y) * 100, "n_test": len(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbones", nargs="*", default=None, help="default: all files in feats/")
    ap.add_argument("--jobs", type=int, default=4, help="backbones probed in parallel (CPU processes)")
    args = ap.parse_args()

    out = Path(args.out)
    res_dir = out / "results" / f"seed{args.seed}"; res_dir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(out / "manifest.csv")
    names = args.backbones or sorted(p.stem for p in (out / "feats").glob("*.npz"))

    # One worker per backbone (CPU); each returns its result rows.
    from joblib import Parallel, delayed
    per_bb = Parallel(n_jobs=min(args.jobs, len(names)), verbose=0)(
        delayed(run_backbone)(name, out, man, args.seed) for name in names)
    rows = [r for rs in per_bb for r in rs]
    finish(rows, names, res_dir, args.seed)


def run_backbone(name, out, man, seed):
    rows = []
    z = np.load(out / "feats" / f"{name}.npz")
    X = z["feats"].astype(np.float32)
    m = man.loc[z["row_id"]]
    ds = m.dataset.to_numpy()
    y = m.label.map(CORE_TO_ID).to_numpy()
    split = make_split(ds, y, seed=seed)

    protocols = [("single", [d]) for d in DATASET_NAMES] + \
                [("lodo", [s for s in DATASET_NAMES if s != d]) for d in DATASET_NAMES]
    for proto, srcs in protocols:
        t0 = time.time()
        src = np.isin(ds, srcs)
        tr, va = src & (split == "train"), src & (split == "val")
        sc, clf, C, val_f1 = fit_probe(X[tr], y[tr], X[va], y[va], seed)
        targets = DATASET_NAMES if proto == "single" else [d for d in DATASET_NAMES if d not in srcs]
        for tgt in targets:
            te = (ds == tgt) & (split == "test")
            r = {"backbone": name, "seed": seed, "protocol": proto,
                 "train_on": "+".join(srcs), "test_on": tgt,
                 "in_domain": int(tgt in srcs), "C": C, "val_f1": val_f1 * 100}
            r.update(evaluate(sc, clf, X[te], y[te]))
            rows.append(r)
        print(f"[{name}] {proto:6s} train={'+'.join(srcs):35s} C={C:<6} "
              f"val_f1={val_f1*100:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def finish(rows, names, res_dir, seed):
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "probe_results.csv", index=False)

    # 4x4 matrices (Fig. 2) per backbone
    single = df[df.protocol == "single"]
    for name in names:
        for metric in ["acc", "macro_f1"]:
            mat = single[single.backbone == name].pivot(index="train_on", columns="test_on", values=metric)
            mat.loc[DATASET_NAMES, DATASET_NAMES].round(2).to_csv(res_dir / f"matrix_{metric}_{name}.csv")

    # Table 2a summary
    summ = []
    for name in names:
        s = single[single.backbone == name]
        lodo = df[(df.protocol == "lodo") & (df.backbone == name)]
        ind, ood = s[s.in_domain == 1].macro_f1.mean(), s[s.in_domain == 0].macro_f1.mean()
        t2r = s[(s.train_on == "TrashNet") & (s.test_on == "RealWaste")].macro_f1.item()
        summ.append({"backbone": name, "in_domain_F1": ind, "ood_F1": ood, "gap": ind - ood,
                     "TrashNet->RealWaste_F1": t2r, "LODO_avg_F1": lodo.macro_f1.mean(),
                     "ood_ECE": s[s.in_domain == 0].ece.mean()})
    summ = pd.DataFrame(summ).round(2)
    summ.to_csv(res_dir / "table2a_summary.csv", index=False)
    print(f"\n=== Table 2a (macro-F1 in %, seed {seed}) ===")
    print(summ.to_string(index=False))
    for name in names:
        print(f"\n--- {name}: macro-F1 matrix (rows=train, cols=test) ---")
        print(pd.read_csv(res_dir / f"matrix_macro_f1_{name}.csv", index_col=0).to_string())


if __name__ == "__main__":
    main()
