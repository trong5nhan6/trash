"""Cache kept core images with shorter side = 256 px (same relative paths) so fine-tuning is not
limited by JPEG decoding of large images on Kaggle's 4 CPU cores.

Usage: python preresize.py --root <data> --out outputs --dst /tmp/small
"""
import argparse, os
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from PIL import Image

ARGS = {}


def work(rel):
    src, dst = Path(ARGS["root"]) / rel, Path(ARGS["dst"]) / rel
    if dst.exists(): return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        s = ARGS["side"] / min(im.size)
        if s < 1: im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
        im.save(dst, quality=95)


def init(a): ARGS.update(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--dst", default="/tmp/small")
    ap.add_argument("--side", type=int, default=256)
    args = ap.parse_args()
    man = pd.read_csv(Path(args.out) / "manifest.csv")
    paths = man[(man.keep == 1) & (man.core == 1)].path.tolist()
    with Pool(os.cpu_count(), initializer=init, initargs=(vars(args),)) as p:
        for i, _ in enumerate(p.imap_unordered(work, paths, chunksize=64)):
            if i % 5000 == 0: print(f"  {i}/{len(paths)}", flush=True)
    print(f"cached {len(paths)} images -> {args.dst}")


if __name__ == "__main__":
    main()
