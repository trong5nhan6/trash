"""XAI analysis of the linear-probe baseline (frozen backbones), CPU-friendly.

(A) Region masking: features are recomputed with the central box or the border ring replaced by the
    image mean (0 after normalisation); the B0 probe trained on original TrashNet images is applied.
    'object only' keeps the central 60% box, 'context only' keeps the border ring.
(B) Occlusion sensitivity (Zeiler & Fergus): a 6x6 grid of grey patches; attribution of a cell = drop of
    the predicted-class probability when it is occluded. We report the share of positive attribution that
    falls in the border ring, for correct and wrong OOD predictions (TrashNet -> RealWaste), and save maps.
Usage: python xai.py --root <data> --out <analysis dir with manifest.csv and feats/> [--n-mask 300 --n-occ 60]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from common import CORE, CORE_TO_ID, DATASET_NAMES as DS, make_split
from probe import fit_probe, load_kept
from extract import build

S = 224
C0, C1 = int(0.2 * S), int(0.8 * S)          # central 60% box


def masks():
    m = torch.zeros(S, S, dtype=torch.bool); m[C0:C1, C0:C1] = True
    return m                                  # True = centre


@torch.no_grad()
def encode(model, x):
    return model(x).float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbones", nargs="+", default=["resnet50", "dinov2_b", "siglip_b16"])
    ap.add_argument("--source", default="TrashNet")
    ap.add_argument("--ood", default="RealWaste")
    ap.add_argument("--n-mask", type=int, default=300, help="test images per dataset for (A)")
    ap.add_argument("--n-occ", type=int, default=60, help="correct and wrong OOD images each for (B)")
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    O = Path(a.out); (O / "xai").mkdir(exist_ok=True)
    man = pd.read_csv(O / "manifest.csv")
    centre = masks()
    rng = np.random.default_rng(a.seed)
    res_a, res_b = [], []
    for bb in a.backbones:
        t0 = time.time()
        X, rid = load_kept(np.load(O / "feats" / f"{bb}.npz"), man)
        m = man.loc[rid]; ds = m.dataset.to_numpy(); y = m.label.map(CORE_TO_ID).to_numpy()
        sp = make_split(ds, y, a.seed)
        tr, va = (ds == a.source) & (sp == "train"), (ds == a.source) & (sp == "val")
        sc, clf, _, _ = fit_probe(X[tr], y[tr], X[va], y[va], a.seed)
        prob = lambda F: clf.predict_proba(sc.transform(F))
        model, tf = build(bb); model.eval()
        load = lambda i: tf(Image.open(Path(a.root) / m.path.iloc[i]).convert("RGB"))

        # ---------- (A) region masking
        for d in DS:
            idx = np.where((ds == d) & (sp == "test"))[0]
            idx = rng.choice(idx, min(a.n_mask, len(idx)), replace=False)
            xs = torch.stack([load(i) for i in idx])
            obj = xs.clone(); obj[:, :, ~centre] = 0          # keep object (centre)
            ctx = xs.clone(); ctx[:, :, centre] = 0           # keep context (border)
            for cond, batch in [("original", xs), ("object only", obj), ("context only", ctx)]:
                F = np.concatenate([encode(model, batch[k:k + 32]) for k in range(0, len(batch), 32)])
                pr = prob(F).argmax(1)
                from sklearn.metrics import f1_score
                res_a.append({"backbone": bb, "test_on": d, "in_domain": int(d == a.source), "condition": cond,
                              "macro_f1": f1_score(y[idx], pr, average="macro", labels=range(len(CORE))) * 100,
                              "acc": (pr == y[idx]).mean() * 100, "n": len(idx)})
        print(f"[{bb}] (A) done {time.time() - t0:.0f}s", flush=True)

        # ---------- (B) occlusion sensitivity on OOD images (source -> ood)
        te = np.where((ds == a.ood) & (sp == "test"))[0]
        p0 = prob(X[te]); pred = p0.argmax(1); ok = pred == y[te]
        sel = [(i, True) for i in rng.choice(te[ok], min(a.n_occ, ok.sum()), replace=False)] + \
              [(i, False) for i in rng.choice(te[~ok], min(a.n_occ, (~ok).sum()), replace=False)]
        g = a.grid; cell = S // g
        ring = np.ones((g, g), bool); ring[1:g - 1, 1:g - 1] = False        # border cells
        maps = {}
        for i, correct in sel:
            x = load(i)
            base = prob(encode(model, x[None]))[0]; c = base.argmax()
            occ = x[None].repeat(g * g, 1, 1, 1)
            for r in range(g):
                for q in range(g):
                    occ[r * g + q, :, r * cell:(r + 1) * cell, q * cell:(q + 1) * cell] = 0
            pc = prob(np.concatenate([encode(model, occ[k:k + 36]) for k in range(0, len(occ), 36)]))[:, c]
            att = np.clip(base[c] - pc, 0, None).reshape(g, g)
            share = att[ring].sum() / max(att.sum(), 1e-9)
            res_b.append({"backbone": bb, "row_id": int(rid[i]), "correct": int(correct),
                          "true": CORE[y[i]], "pred": CORE[c], "conf": float(base[c]), "border_share": float(share)})
            maps[int(rid[i])] = att.tolist()
        json.dump(maps, open(O / "xai" / f"occlusion_maps_{bb}.json", "w"))
        print(f"[{bb}] (B) done {time.time() - t0:.0f}s", flush=True)
        del model

    A = pd.DataFrame(res_a); A.to_csv(O / "xai" / "region_masking.csv", index=False)
    B = pd.DataFrame(res_b); B.to_csv(O / "xai" / "occlusion.csv", index=False)
    print(A.pivot_table(index=["backbone", "condition"], columns="in_domain", values="macro_f1").round(1))
    print(B.groupby(["backbone", "correct"]).border_share.agg(["mean", "median", "count"]).round(3))


if __name__ == "__main__":
    main()
