"""Step 1: scan the 4 datasets, verify images, remove exact duplicates, write manifest.csv.

Usage:  python prepare.py --root <folder containing TrashNet/ RealWaste/ TrashBox/ Garbage12/> --out <out_dir>
"""
import argparse, csv, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

from common import DATASETS, CORE, IMG_EXT, DEDUP_PRIORITY, resolve_root


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    root = resolve_root(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds, cfg in DATASETS.items():
        for d in cfg["dirs"]:
            for cls_dir in sorted(p for p in (root / d).iterdir() if p.is_dir()):
                for f in sorted(cls_dir.rglob("*")):
                    if f.suffix.lower() not in IMG_EXT:
                        continue
                    label = cfg["map"].get(cls_dir.name, "other")
                    r = {"path": f.relative_to(root).as_posix(), "dataset": ds, "domain": cfg["domain"],
                         "orig_class": cls_dir.name, "label": label, "core": int(label in CORE),
                         "width": -1, "height": -1, "md5": "", "keep": 1, "drop_reason": ""}
                    try:
                        with Image.open(f) as im:
                            im.verify()
                        with Image.open(f) as im:
                            r["width"], r["height"] = im.size
                        r["md5"] = md5(f)
                    except Exception as e:  # corrupted / unreadable
                        r["keep"], r["drop_reason"] = 0, f"unreadable: {type(e).__name__}"
                    rows.append(r)
        print(f"[scan] {ds}: {sum(r['dataset'] == ds for r in rows)} files")

    # ---- exact-duplicate removal (md5) ----
    by_hash = defaultdict(list)
    for i, r in enumerate(rows):
        if r["keep"]:
            by_hash[r["md5"]].append(i)
    cross_pairs, cross_label_agree = Counter(), Counter()
    cross_log = []
    for idxs in by_hash.values():
        if len(idxs) < 2:
            continue
        # 1) inside each dataset: conflicting labels -> drop all copies, else keep the first copy
        survivors = []
        for ds in DEDUP_PRIORITY:
            own = [i for i in idxs if rows[i]["dataset"] == ds]
            if not own:
                continue
            if len({rows[i]["label"] for i in own}) > 1:
                for i in own:
                    rows[i]["keep"], rows[i]["drop_reason"] = 0, "dup_conflicting_label"
                continue
            for i in own[1:]:
                rows[i]["keep"], rows[i]["drop_reason"] = 0, "dup_within_dataset"
            survivors.append(own[0])
        # 2) across datasets: keep the copy in the earliest-published dataset, drop later copies
        if len(survivors) > 1:
            orig = survivors[0]
            for i in survivors[1:]:
                key = f"{rows[orig]['dataset']}->{rows[i]['dataset']}"
                cross_pairs[key] += 1
                cross_label_agree[key] += int(rows[i]["label"] == rows[orig]["label"])
                rows[i]["keep"], rows[i]["drop_reason"] = 0, f"dup_of_{rows[orig]['dataset']}"
                cross_log.append({"kept": rows[orig]["path"], "kept_label": rows[orig]["orig_class"],
                                  "dropped": rows[i]["path"], "dropped_label": rows[i]["orig_class"]})

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- report (basis for Table 1) ----
    lines = ["dataset,label,n_raw,n_kept"]
    for ds in DATASETS:
        for lab in CORE + ["other"]:
            sel = [r for r in rows if r["dataset"] == ds and r["label"] == lab]
            lines.append(f"{ds},{lab},{len(sel)},{sum(r['keep'] for r in sel)}")
    (out / "table1_counts.csv").write_text("\n".join(lines), encoding="utf-8")

    drops = Counter((r["dataset"], r["drop_reason"]) for r in rows if not r["keep"])
    print("\n[drop] dataset / reason / n")
    for (ds, why), n in sorted(drops.items()):
        print(f"  {ds:10s} {why:24s} {n}")
    print("\n[cross-dataset exact duplicates]  original->copy : n  (same unified label)")
    for k, n in sorted(cross_pairs.items()):
        print(f"  {k:22s}: {n:5d}  ({cross_label_agree[k]} same label)")
    if cross_log:
        with open(out / "cross_dataset_duplicates.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cross_log[0]))
            w.writeheader(); w.writerows(cross_log)
    print("\n[core 5-class, kept]")
    for ds in DATASETS:
        c = Counter(r["label"] for r in rows if r["dataset"] == ds and r["keep"] and r["core"])
        print(f"  {ds:10s}", {k: c.get(k, 0) for k in CORE}, "total", sum(c.values()))
    print(f"\nWrote {out / 'manifest.csv'} and {out / 'table1_counts.csv'}")


if __name__ == "__main__":
    main()
