"""Method M1 step 1: foreground segmentation with rembg (U^2-Net), saved as RGBA cut-outs.

Segments every kept core image + up to --n-other images per dataset from the non-core classes
(e-waste, food, textile, ...). Those 'other' objects are used only as clutter distractors, so
no 5-class test image can leak into training composites.
Run one process per GPU:  --shard 0 --nshards 2  /  --shard 1 --nshards 2
Output: outputs/seg/fg/<row_id>.png (RGBA, cropped to object, longest side <= 256)
        outputs/seg/stats_<shard>.csv  (row_id, dataset, core, fg_frac, ok)
"""
import argparse, csv, time
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

MIN_FRAC, MAX_FRAC = 0.02, 0.95   # masks outside this range are treated as failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-other", type=int, default=600)
    ap.add_argument("--model", default="u2net")
    args = ap.parse_args()

    import onnxruntime as ort
    from rembg import new_session, remove
    print("onnxruntime providers:", ort.get_available_providers(), flush=True)
    sess = new_session(args.model)

    out = Path(args.out); fg_dir = out / "seg" / "fg"; fg_dir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(out / "manifest.csv")
    core = man[(man.keep == 1) & (man.core == 1)]
    other = man[(man.keep == 1) & (man.core == 0)]
    other = other.groupby("dataset", group_keys=False).apply(
        lambda d: d.sample(min(len(d), args.n_other), random_state=0))
    todo = pd.concat([core, other]).sort_index()
    todo = todo.iloc[args.shard::args.nshards]
    stats_f = out / "seg" / f"stats_{args.shard}.csv"
    done = set(pd.read_csv(stats_f).row_id) if stats_f.exists() else set()
    print(f"shard {args.shard}: {len(todo)} images ({len(done)} already done)", flush=True)

    t0 = time.time()
    with open(stats_f, "a", newline="") as f:
        w = csv.writer(f)
        if not done: w.writerow(["row_id", "dataset", "core", "fg_frac", "ok"])
        for n, (rid, r) in enumerate(todo.iterrows()):
            if rid in done: continue
            try:
                im = Image.open(Path(args.root) / r.path).convert("RGB")
                im.thumbnail((512, 512))
                mask = remove(im, session=sess, only_mask=True)
                a = np.asarray(mask, dtype=np.uint8)
                frac = float((a > 127).mean())
                ok = MIN_FRAC <= frac <= MAX_FRAC
                if ok:
                    ys, xs = np.where(a > 127)
                    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                    rgba = im.copy(); rgba.putalpha(mask)
                    rgba = rgba.crop(box); rgba.thumbnail((256, 256))
                    rgba.save(fg_dir / f"{rid}.png")
            except Exception as e:
                print("error", r.path, e, flush=True); frac, ok = -1.0, False
            w.writerow([rid, r.dataset, r.core, round(frac, 4), int(ok)])
            if n % 1000 == 0:
                f.flush(); print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"shard {args.shard} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
