"""Per-image nuisance descriptors (cheap, pixel statistics; no labels needed).

Used by method M7 (nuisance erasure): these describe *how* a photo was taken, not *what* it shows.
  border strip (10% of each side): mean RGB, std RGB, mean saturation, mean brightness, brightness std
  center-vs-border colour contrast, 4x4x4 colour histogram of the border (64 bins)
  acquisition: log(shorter side), aspect ratio
Output: outputs/nuisance.npz  (Z [N, 76] float32, row_id)

Usage: python nuisance.py --root <data> --out outputs
"""
import argparse, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = {}


def describe(rel):
    with Image.open(Path(ROOT["root"]) / rel) as im:
        W, H = im.size
        im = im.convert("RGB"); im.thumbnail((128, 128))
    a = np.asarray(im, float) / 255; h, w = a.shape[:2]; b = max(2, int(0.1 * min(h, w)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                             a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    center = a[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3)
    mx, mn = border.max(1), border.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    hist = np.histogramdd(border, bins=(4, 4, 4), range=[(0, 1)] * 3)[0].ravel() / len(border)
    return np.concatenate([border.mean(0), border.std(0),
                           [sat.mean(), border.mean(), border.std(),
                            np.abs(center.mean(0) - border.mean(0)).sum()],
                           hist, [np.log(min(W, H)), W / H]]).astype(np.float32)


def init(r): ROOT["root"] = r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "nuisance.npz").exists():
        print("nuisance.npz exists -> skip"); return
    man = pd.read_csv(out / "manifest.csv")
    sel = man[(man.keep == 1) & (man.core == 1)]
    with Pool(os.cpu_count(), initializer=init, initargs=(args.root,)) as p:
        Z = np.stack(p.map(describe, sel.path.tolist(), chunksize=64))
    np.savez(out / "nuisance.npz", Z=Z, row_id=sel.index.to_numpy())
    print(f"nuisance descriptors: {Z.shape}")


if __name__ == "__main__":
    main()
