"""Method M1 step 2: CLAD-Aug composites (Clutter- And Degradation-aware recomposition).

Variants
  c1   : object cut-out (random flip/rotate/scale/position) on a *procedural* background.
         ~ ForAug-style foreground/background recombination -> strong baseline.
  full : c1 + C2 clutter (1-3 'other'-class objects of the same dataset behind the object,
         sometimes one small occluder in front) + C3 degradation (crumple/elastic warp,
         dirt stains, colour/lighting change, cast shadow).
No real background or object from any 5-class test image is used: backgrounds are procedural
and distractors come only from non-core classes -> no target leakage under any protocol.

Usage: python composite.py --out outputs --img-out /tmp/clad --variants c1 full --k 3
Output: <img-out>/<variant>/<row_id>_<k>.jpg and <img-out>/list_<variant>.csv (path,row_id,k)
"""
import argparse, glob, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates

S = 256
G = {}  # per-worker globals


def _init(fg_dir, pools):
    G["fg_dir"], G["pools"] = fg_dir, pools


def load_fg(rid):
    return Image.open(os.path.join(G["fg_dir"], f"{rid}.png")).convert("RGBA")


# ---------------- procedural backgrounds ----------------
def muted(rng, lo=30, hi=230):
    """Random colour pulled towards grey (natural-looking, low saturation)."""
    c = rng.uniform(lo, hi, 3)
    g = c.mean()
    return g + rng.uniform(.15, .6) * (c - g)


def procedural_bg(rng):
    kind = rng.integers(4)
    if kind == 0:   # smooth two-colour gradient + grain
        c1, c2 = muted(rng), muted(rng)
        th = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:S, 0:S] / S
        t = (np.cos(th) * xx + np.sin(th) * yy); t = (t - t.min()) / (np.ptp(t) + 1e-6)
        img = c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None]
        img += rng.normal(0, 6, img.shape)
    elif kind == 1:  # multi-scale noise, earthy / concrete palette (ground-like)
        base = rng.choice([[120, 110, 100], [90, 80, 60], [150, 150, 145], [70, 70, 70], [140, 120, 90]])
        n = sum(w * np.kron(rng.normal(0, 1, (S // s, S // s)), np.ones((s, s)))
                for s, w in [(4, .2), (8, .3), (16, .3), (32, .2)])
        n = gaussian_filter(n, 2)
        img = np.asarray(base, float)[None, None] + 35 * n[..., None] + rng.normal(0, 8, (S, S, 3))
    elif kind == 2:  # synthetic clutter: many blurred, muted random shapes
        im = Image.new("RGB", (S, S), tuple(int(v) for v in muted(rng, 60, 200)))
        d = ImageDraw.Draw(im)
        for _ in range(rng.integers(15, 45)):
            x0, y0 = rng.integers(-40, S, 2); w, h = rng.integers(10, 90, 2)
            col = tuple(int(v) for v in muted(rng))
            (d.ellipse if rng.random() < .5 else d.rectangle)([x0, y0, x0 + w, y0 + h], fill=col)
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(2, 4)))
        arr = np.asarray(im, float) + rng.normal(0, 6, (S, S, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    else:            # flat colour + light noise
        img = muted(rng)[None, None] + rng.normal(0, 5, (S, S, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# ---------------- C3 degradations (object appearance) ----------------
def elastic(rgba, rng, alpha=None, sigma=6):
    a = np.asarray(rgba).astype(np.float32); h, w = a.shape[:2]
    alpha = alpha or rng.uniform(4, 10)
    dx = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha * sigma
    dy = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha * sigma
    yy, xx = np.mgrid[0:h, 0:w]
    coords = [np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]
    out = np.stack([map_coordinates(a[..., c], coords, order=1) for c in range(4)], -1)
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def degrade(rgba, rng):
    if rng.random() < .5:
        rgba = elastic(rgba, rng)                      # crumpled / deformed object
    rgb, al = rgba.convert("RGB"), rgba.getchannel("A")
    if rng.random() < .8:                              # lighting / colour change
        rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(.6, 1.2))
        rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(.7, 1.2))
        rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(.6, 1.2))
    if rng.random() < .5:                              # dirt / stains
        h, w = rgb.size[1], rgb.size[0]
        blob = gaussian_filter(rng.random((h, w)), rng.uniform(2, 6))
        blob = np.clip((blob - np.quantile(blob, rng.uniform(.6, .85))) * 8, 0, 1)[..., None]
        dirt = np.array([rng.uniform(60, 110), rng.uniform(50, 85), rng.uniform(30, 60)])
        arr = np.asarray(rgb, float)
        arr = arr * (1 - rng.uniform(.25, .5) * blob) + dirt * rng.uniform(.25, .5) * blob
        rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    rgb.putalpha(al)
    return rgb


def shadow(canvas, rng):
    yy, xx = np.mgrid[0:S, 0:S] / S
    th = rng.uniform(0, 2 * np.pi)
    t = np.cos(th) * (xx - .5) + np.sin(th) * (yy - .5)
    m = 1 - rng.uniform(.15, .4) * (t > rng.uniform(-.2, .2))
    m = gaussian_filter(m, 6)[..., None]
    return Image.fromarray(np.clip(np.asarray(canvas, float) * m, 0, 255).astype(np.uint8))


# ---------------- placement ----------------
def place(canvas, rgba, rng, size_frac, center_range=(.3, .7), center=None):
    rgba = rgba.copy()
    if rng.random() < .5: rgba = rgba.transpose(Image.FLIP_LEFT_RIGHT)
    rgba = rgba.rotate(rng.uniform(-30, 30), expand=True, resample=Image.BILINEAR)
    sc = size_frac * S / max(rgba.size)
    rgba = rgba.resize((max(1, int(rgba.size[0] * sc)), max(1, int(rgba.size[1] * sc))), Image.BILINEAR)
    cx, cy = center if center is not None else rng.uniform(*center_range, 2) * S
    x, y = int(cx - rgba.size[0] / 2), int(cy - rgba.size[1] / 2)
    canvas.paste(rgba, (x, y), rgba)
    return (x, y, x + rgba.size[0], y + rgba.size[1])


def make(task):
    rid, ds, k, variant, out_path = task
    rng = np.random.default_rng([int(rid), int(k), 0 if variant == "c1" else 1])
    canvas = procedural_bg(rng)
    obj = load_fg(rid)
    pool = G["pools"].get(ds, [])
    full = variant == "full"
    if full and pool:                                   # C2: clutter behind the object
        for d in rng.choice(pool, size=min(len(pool), rng.integers(1, 4)), replace=False):
            place(canvas, degrade(load_fg(d), rng), rng, rng.uniform(.25, .5), (0, 1))
    if full:
        obj = degrade(obj, rng)                         # C3: object degradation
    box = place(canvas, obj, rng, rng.uniform(.45, .85))
    if full and pool and rng.random() < .3:             # C2: small partial occluder on the object's edge
        x0, y0, x1, y1 = box
        side = rng.integers(4)
        cx = [x0, x1, rng.uniform(x0, x1), rng.uniform(x0, x1)][side]
        cy = [rng.uniform(y0, y1), rng.uniform(y0, y1), y0, y1][side]
        place(canvas, load_fg(rng.choice(pool)), rng, rng.uniform(.15, .3), center=(cx, cy))
    if full and rng.random() < .3:
        canvas = shadow(canvas, rng)
    canvas.convert("RGB").save(out_path, quality=90)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--img-out", default="/tmp/clad")
    ap.add_argument("--variants", nargs="+", default=["c1", "full"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    out = Path(args.out)
    st = pd.concat([pd.read_csv(f) for f in glob.glob(str(out / "seg" / "stats_*.csv"))])
    st = st.drop_duplicates("row_id")
    ok = st[st.ok == 1]
    pools = {d: g.row_id.tolist() for d, g in ok[ok.core == 0].groupby("dataset")}
    core_ok = ok[ok.core == 1]
    print("mask success rate (core):", st[st.core == 1].groupby("dataset").ok.mean().round(3).to_dict())
    print("distractor pool sizes:", {d: len(v) for d, v in pools.items()})

    for v in args.variants:
        vdir = Path(args.img_out) / v; vdir.mkdir(parents=True, exist_ok=True)
        tasks = [(r.row_id, r.dataset, k, v, str(vdir / f"{r.row_id}_{k}.jpg"))
                 for r in core_ok.itertuples() for k in range(args.k)]
        todo = [t for t in tasks if not os.path.exists(t[4])]
        with Pool(args.workers, initializer=_init, initargs=(str(out / "seg" / "fg"), pools)) as p:
            for i, _ in enumerate(p.imap_unordered(make, todo, chunksize=64)):
                if i % 10000 == 0: print(f"  {v}: {i}/{len(todo)}", flush=True)
        pd.DataFrame({"path": [os.path.relpath(t[4], args.img_out) for t in tasks],
                      "row_id": [t[0] for t in tasks], "k": [t[2] for t in tasks]}
                     ).to_csv(Path(args.img_out) / f"list_{v}.csv", index=False)
        print(f"{v}: {len(tasks)} composites -> {vdir}")


if __name__ == "__main__":
    main()
