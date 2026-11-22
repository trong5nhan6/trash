"""Near-duplicate filtering with DINOv2 features (cosine >= threshold), applied after md5 dedup.

Visual inspection: >=0.98 identical image (watermark/resize), 0.95-0.98 mostly same photo,
0.92-0.95 different photos of similar objects -> default threshold 0.95.
Groups are connected components; inside a group the copy from the earliest-published dataset
(DEDUP_PRIORITY) is kept. Updates manifest.csv in place (keep=0, drop_reason=near_dup_of_<ds>).

Usage: python neardup.py --out outputs [--feats dinov2_b] [--thr 0.95]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from common import DEDUP_PRIORITY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--feats", default="dinov2_b")
    ap.add_argument("--thr", type=float, default=0.95)
    args = ap.parse_args()
    out = Path(args.out)
    man = pd.read_csv(out / "manifest.csv")
    if man.drop_reason.astype(str).str.startswith("near_dup").any():
        print("near-dup filtering already applied -> skip"); return

    z = np.load(out / "feats" / f"{args.feats}.npz")
    row_id = z["row_id"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.nn.functional.normalize(torch.tensor(z["feats"].astype(np.float32), device=dev), dim=1)
    n = len(X)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    n_pairs = 0
    for s in range(0, n, 2048):                     # chunked similarity to bound memory
        S = X[s:s + 2048] @ X.T
        ii, jj = (S >= args.thr).nonzero(as_tuple=True)
        ii = ii.cpu().numpy() + s; jj = jj.cpu().numpy()
        for a, b in zip(ii, jj):
            if a < b:
                n_pairs += 1
                ra, rb = find(a), find(b)
                if ra != rb: parent[ra] = rb

    pri = {d: i for i, d in enumerate(DEDUP_PRIORITY)}
    ds = man.loc[row_id, "dataset"].to_numpy()
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    dropped = []
    for members in groups.values():
        if len(members) < 2: continue
        members.sort(key=lambda i: (pri[ds[i]], i))
        keeper = ds[members[0]]
        for i in members[1:]:
            r = row_id[i]
            man.loc[r, ["keep", "drop_reason"]] = [0, f"near_dup_of_{keeper}"]
            dropped.append(ds[i])
    man.to_csv(out / "manifest.csv", index=False)
    print(f"pairs >= {args.thr}: {n_pairs} | dropped per dataset:",
          pd.Series(dropped).value_counts().to_dict())


if __name__ == "__main__":
    main()
