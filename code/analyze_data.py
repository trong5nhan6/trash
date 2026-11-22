"""Data analysis that motivates the methods (numbers for the paper's analysis section).

A1 class priors per dataset            A2 'Name That Dataset' (linear probe predicts the source dataset)
A3 background shortcut: class predicted from border/acquisition descriptors only (needs nuisance.npz)
A4 feature geometry: same-class/different-dataset vs different-class/same-dataset centroid distance
Usage: python analyze_data.py --out outputs --backbones resnet50 dinov2_b siglip_b16 [--seed 0]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from common import CORE, DATASET_NAMES as DS, make_split
from probe import load_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--backbones", nargs="+", default=["resnet50", "dinov2_b", "siglip_b16"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); man = pd.read_csv(out / "manifest.csv")
    core = man[(man.keep == 1) & (man.core == 1)]
    pd.set_option("display.width", 200)
    print("A1 class proportions (%)\n", pd.crosstab(core.dataset, core.label, normalize="index").mul(100).round(1).loc[DS])

    for bb in args.backbones:
        X, rid = load_kept(np.load(out / "feats" / f"{bb}.npz"), man); m = man.loc[rid]
        ds, y = m.dataset.to_numpy(), m.label.to_numpy(); sp = make_split(ds, y, args.seed)
        tr, te = sp == "train", sp == "test"
        sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)
        clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Z[tr], ds[tr])
        acc = accuracy_score(ds[te], clf.predict(Z[te])) * 100
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        C = {(d, c): Xn[(ds == d) & (y == c)].mean(0) for d in DS for c in CORE}
        cd = lambda a, b: 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b)
        same_c = np.mean([cd(C[(a, c)], C[(b, c)]) for c in CORE for i, a in enumerate(DS) for b in DS[i + 1:]])
        same_d = np.mean([cd(C[(d, a)], C[(d, b)]) for d in DS for i, a in enumerate(CORE) for b in CORE[i + 1:]])
        spread = {c: np.mean([cd(C[(a, c)], C[(b, c)]) for i, a in enumerate(DS) for b in DS[i + 1:]]) for c in CORE}
        print(f"\n[{bb}] A2 name-that-dataset acc = {acc:.1f}% (chance 25)")
        print(f"[{bb}] A4 centroid distance: same class/diff dataset = {same_c:.3f} | diff class/same dataset = {same_d:.3f}")
        print(f"[{bb}] A4 per-class cross-dataset spread:", {k: round(v, 3) for k, v in spread.items()})

    nf = out / "nuisance.npz"
    if nf.exists():
        z = np.load(nf); pos = {r: i for i, r in enumerate(z["row_id"])}
        N = z["Z"][[pos[r] for r in core.index]]
        ds, y = core.dataset.to_numpy(), core.label.to_numpy(); sp = make_split(ds, y, args.seed)
        M = {}
        for s in DS:
            tr = (ds == s) & (sp == "train"); sc = StandardScaler().fit(N[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(N[tr]), y[tr])
            M[s] = {t: f1_score(y[(ds == t) & (sp == "test")], clf.predict(sc.transform(N[(ds == t) & (sp == "test")])),
                                average="macro") * 100 for t in DS}
        print("\nA3 class from border/acquisition descriptors only (macro-F1 %, chance 20; rows=train)\n",
              pd.DataFrame(M).T.loc[DS, DS].round(1))


if __name__ == "__main__":
    main()
