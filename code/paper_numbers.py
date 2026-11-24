"""Collect every number used in the paper from the result folders (CPU only).

Usage: python paper_numbers.py --results ../results --out ../paperwrite/analysis
Writes CSV files into --out and prints a compact report.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from common import CORE, DATASET_NAMES as DS

BB = ["resnet50", "effnet_b0", "convnext_t", "dinov2_b", "dinov2_l", "clip_b16", "siglip_b16"]
NAMES = {"resnet50": "ResNet-50", "effnet_b0": "EfficientNet-B0", "convnext_t": "ConvNeXt-T",
         "dinov2_b": "DINOv2-B/14", "dinov2_l": "DINOv2-L/14", "clip_b16": "CLIP ViT-B/16",
         "siglip_b16": "SigLIP ViT-B/16"}


def read_runs(pattern):
    return pd.concat(pd.read_csv(f) for f in glob.glob(pattern) if not f.endswith(".gz"))


def per_seed(df):
    """protocol-level metrics -> one row per (backbone, variant, seed)."""
    rows = []
    for (bb, v, s), g in df.groupby(["backbone", "variant", "seed"]):
        sg, lo = g[g.protocol == "single"], g[g.protocol == "lodo"]
        rows.append({"backbone": bb, "variant": v, "seed": s,
                     "ID": sg[sg.in_domain == 1].macro_f1.mean(), "OOD": sg[sg.in_domain == 0].macro_f1.mean(),
                     "T2R": sg[(sg.train_on == "TrashNet") & (sg.test_on == "RealWaste")].macro_f1.mean(),
                     "LODO": lo.macro_f1.mean(), "ECE": sg[sg.in_domain == 0].ece.mean()})
    r = pd.DataFrame(rows); r["gap"] = r.ID - r.OOD
    return r


def mean_std(r):
    g = r.groupby(["backbone", "variant"])
    m, s = g[["ID", "OOD", "gap", "T2R", "LODO", "ECE"]].mean(), g[["ID", "OOD", "gap", "T2R", "LODO", "ECE"]].std()
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../paperwrite/analysis")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    # ---------------- data audit (Table 1)
    man = pd.read_csv(R / "kaggle_gated_methods/res/manifest.csv")
    rows = []
    for d in DS:
        m = man[man.dataset == d]; c = m[m.core == 1]
        k = c[c.keep == 1]
        rows.append({"dataset": d, "files": len(m), "core_raw": len(c),
                     "exact_dup_within": int((c.drop_reason == "dup_within_dataset").sum() + (c.drop_reason == "dup_conflicting_label").sum()),
                     "copy_of_TrashNet": int((c.drop_reason == "dup_of_TrashNet").sum()),
                     "near_dup": int(c.drop_reason.astype(str).str.startswith("near_dup").sum()),
                     "unreadable": int(c.drop_reason.astype(str).str.startswith("unreadable").sum()),
                     **{cl: int((k.label == cl).sum()) for cl in CORE}, "kept": len(k),
                     "median_short_side": int(np.median(np.minimum(k.width, k.height)))})
    t1 = pd.DataFrame(rows); t1.to_csv(O / "t1_data.csv", index=False)
    allm = man.copy()
    print("== T1 data audit\n", t1.to_string(index=False))
    print("all-class cross copies:", allm.drop_reason.value_counts().to_dict())

    # ---------------- baseline + frozen CG (gated run, 7 backbones)
    g = read_runs(str(R / "kaggle_gated_methods/res/results_methods/gated/seed*/probe_results.csv"))
    gs = per_seed(g); gm, gsd = mean_std(gs)
    b0 = gm.xs("b0", level="variant").loc[BB]
    b0sd = gsd.xs("b0", level="variant").loc[BB]
    t2 = b0.round(2).astype(str) + " ± " + b0sd.round(2).astype(str)
    t2.to_csv(O / "t2_baseline.csv"); print("\n== T2 baseline (B0, frozen LP)\n", t2.to_string())
    # 4x4 matrices of B0
    sg = g[(g.variant == "b0") & (g.protocol == "single")]
    for bb in BB:
        M = sg[sg.backbone == bb].groupby(["train_on", "test_on"]).macro_f1.mean().unstack().loc[DS, DS]
        M.round(1).to_csv(O / f"matrix_b0_{bb}.csv")
    print("\nB0 4x4 SigLIP\n", pd.read_csv(O / "matrix_b0_siglip_b16.csv", index_col=0))
    print("B0 4x4 ResNet\n", pd.read_csv(O / "matrix_b0_resnet50.csv", index_col=0))

    # confidence analysis (B0 predictions)
    gp = pd.concat(pd.read_csv(f) for f in glob.glob(str(R / "kaggle_gated_methods/res/results_methods/gated/seed*/predictions.csv.gz")))
    q = gp[(gp.variant == "b0") & (gp.protocol == "single")].copy()
    q["ind"] = q.train_on == q.test_on; q["ok"] = q.y_true == q.y_pred; q["hi"] = q.conf >= 0.9
    conf = q.groupby(["backbone", "ind"]).apply(lambda d: pd.Series({
        "share_hi": d.hi.mean() * 100, "acc_hi": d[d.hi].ok.mean() * 100, "acc_lo": d[~d.hi].ok.mean() * 100,
        "err_in_lo": (~d.ok & ~d.hi).sum() / max(1, (~d.ok).sum()) * 100}), include_groups=False)
    conf.round(1).to_csv(O / "conf_analysis.csv"); print("\n== confidence (B0)\n", conf.round(1).to_string())
    c2 = q[(q.train_on == "TrashNet") & (q.test_on == "RealWaste") & (q.y_true == "cardboard")]
    cb = c2.groupby("backbone").apply(lambda d: (d.y_pred == "paper").mean() * 100, include_groups=False).round(1)
    print("cardboard->paper (TrashNet->RealWaste, B0):", cb.to_dict())

    # ---------------- LP-FT + CG (main)
    lf = read_runs(str(R / "kaggle_lpft_cg/merged/results_methods/lpftcg/parts/*.csv"))
    ls_ = per_seed(lf); lm, lsd = mean_std(ls_)
    main_vars = ["lp", "ft", "ft_cg_hard", "ft_cg_soft"]
    rows = []
    for bb in BB:
        for v in main_vars:
            r = lm.loc[(bb, v)]; s = lsd.loc[(bb, v)]
            rows.append({"backbone": NAMES[bb], "variant": v, **{k: f"{r[k]:.1f}±{s[k]:.1f}" for k in ["ID", "OOD", "LODO"]},
                         "gap": round(r.gap, 1), "ECE": round(r.ECE, 1)})
    t3 = pd.DataFrame(rows); t3.to_csv(O / "t3_main.csv", index=False)
    mean_bb = lm.groupby(level="variant")[["ID", "OOD", "gap", "LODO", "ECE"]].mean().loc[main_vars].round(2)
    print("\n== T3 main (per backbone)\n", t3.to_string(index=False)); print("mean over 7 backbones\n", mean_bb)

    # ---------------- ablation (deltas vs own reference, mean over 7 backbones)
    def deltas(m, ref):
        d = m.sub(m.xs(ref, level="variant"), level="backbone")
        return d.groupby(level="variant")[["ID", "OOD", "LODO", "ECE"]].agg(["mean", "min", "max"])
    abl_f = deltas(gm, "b0"); abl_t = deltas(lm, "lp")
    abl = pd.concat({"frozen": abl_f, "finetuned": abl_t}).round(2)
    abl.to_csv(O / "t4_ablation.csv"); print("\n== T4 ablation (Δ vs B0/lp)\n", abl.to_string())

    # ---------------- negative results (3 backbones) from transductive + tta + rich + lpft(wise)
    neg = []
    def add(pattern, variants, ref, label):
        df = read_runs(pattern); m, _ = mean_std(per_seed(df))
        d = m.sub(m.xs(ref, level="variant"), level="backbone")
        for v in variants:
            x = d.xs(v, level="variant")[["ID", "OOD", "LODO"]]
            neg.append({"method": label, "variant": v, "n_bb": len(x),
                        **{k: round(x[k].mean(), 2) for k in ["ID", "OOD", "LODO"]}})
    T = R / "kaggle_transductive_methods/res/results_methods"
    add(str(T / "prior/seed*/probe_results.csv"), ["bcts_em"], "b0", "label-shift EM (BCTS)")
    add(str(T / "subclass/seed*/probe_results.csv"), ["subK4"], "b0", "sub-class probe (K=4)")
    add(str(T / "erase/seed*/probe_results.csv"), ["erase_nuis", "erase_ds"], "b0", "LEACE erasure")
    add(str(T / "reweight/seed*/probe_results.csv"), ["rw"], "b0", "background propensity reweighting")
    add(str(T / "tta/seed*/probe_results.csv"), ["featnorm"], "b0", "target feature re-standardisation")
    add(str(R / "kaggle_more_methods_2xT4/res/results_methods/rich/seed*/probe_results.csv"), ["rich"], "b0", "richer features")
    t5 = pd.DataFrame(neg); t5.to_csv(O / "t5_negative.csv", index=False); print("\n== T5 negative\n", t5.to_string(index=False))
    lp_old = read_runs(str(R / "merged_lpft/results_methods/lpft/parts/*.csv"))
    wm, _ = mean_std(per_seed(lp_old)); wd = wm.sub(wm.xs("lp", level="variant"), level="backbone").xs("wise", level="variant")
    print("WiSE alpha=0.5 vs lp:\n", wd[["ID", "OOD", "LODO"]].round(2))

    # ---------------- statistics (paired over backbone x seed x pair)
    st = []
    unit_ood = lf[(lf.protocol == "single") & (lf.in_domain == 0)].pivot_table(
        index=["backbone", "seed", "train_on", "test_on"], columns="variant", values="macro_f1")
    unit_id = lf[(lf.protocol == "single") & (lf.in_domain == 1)].pivot_table(
        index=["backbone", "seed", "train_on"], columns="variant", values="macro_f1")
    unit_lodo = lf[lf.protocol == "lodo"].pivot_table(index=["backbone", "seed", "test_on"], columns="variant", values="macro_f1")
    for v in ["ft", "ft_cg_hard", "ft_cg_soft"]:
        for name, U in [("OOD", unit_ood), ("in-domain", unit_id), ("LODO", unit_lodo)]:
            d = U[v] - U["lp"]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            st.append({"variant": v, "set": name, "n": len(d), "mean_diff": round(d.mean(), 2),
                       "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "p": p})
    t6 = pd.DataFrame(st); t6.to_csv(O / "t6_stats.csv", index=False); print("\n== T6 Wilcoxon vs lp\n", t6.to_string(index=False))

    # ---------------- per-class OOD recall + confusions, lp vs ft_cg_hard
    lp_pred = pd.concat(pd.read_csv(f) for f in glob.glob(str(R / "kaggle_lpft_cg/merged/results_methods/lpftcg/parts/*_pred.csv.gz")))
    z = lp_pred[(lp_pred.protocol == "single") & (lp_pred.train_on != lp_pred.test_on) & lp_pred.variant.isin(["lp", "ft", "ft_cg_hard"])]
    rec = z.assign(ok=z.y_true == z.y_pred).groupby(["variant", "backbone", "y_true"]).ok.mean().unstack()[CORE] * 100
    rc = rec.groupby(level="variant").mean().round(1); rc.to_csv(O / "t7_perclass_recall.csv")
    print("\n== T7 per-class OOD recall (mean over backbones)\n", rc)
    conf_pairs = {}
    for v in ["lp", "ft", "ft_cg_hard"]:
        w = z[z.variant == v]
        conf_pairs[v] = {"cardboard->paper": (w[w.y_true == "cardboard"].y_pred == "paper").mean() * 100,
                         "paper->cardboard": (w[w.y_true == "paper"].y_pred == "cardboard").mean() * 100,
                         "glass->plastic": (w[w.y_true == "glass"].y_pred == "plastic").mean() * 100,
                         "plastic->paper": (w[w.y_true == "plastic"].y_pred == "paper").mean() * 100}
    cp = pd.DataFrame(conf_pairs).round(1); cp.to_csv(O / "t7_confusions.csv"); print(cp)
    kept = lf.dropna(subset=["kept_frac"]).groupby(["backbone", "in_domain"]).kept_frac.mean().unstack().round(3)
    kept.to_csv(O / "gate_kept.csv"); print("\n== gate kept (ft_cg_hard)\n", kept)
    tau = lf.dropna(subset=["tau"]).groupby("backbone").tau.agg(["mean", "min", "max"]).round(2); print(tau)


if __name__ == "__main__":
    main()
