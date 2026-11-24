"""Paper figures (static, print). Run after paper_numbers.py.

fig_heatmaps.pdf : B0 cross-dataset macro-F1 matrices (ResNet-50, SigLIP), sequential blue ramp
fig_tradeoff.pdf : mean change vs B0 in in-domain and OOD macro-F1 for every variant (7 backbones)
Usage: python make_figures.py --analysis ../paperwrite/analysis --out ../paperwrite/paper/figures
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES = {"frozen": "#2a78d6", "fine-tuned": "#eb6834"}     # validated: CVD dE 24.7, normal 33.6
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "pdf.fonttype": 42})


def heatmaps(A, O):
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    DS = ["TrashNet", "RealWaste", "TrashBox", "Garbage12"]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.35), constrained_layout=True)
    for ax, (bb, title) in zip(axes, [("resnet50", "ResNet-50"), ("siglip_b16", "SigLIP ViT-B/16")]):
        M = pd.read_csv(A / f"matrix_b0_{bb}.csv", index_col=0).loc[DS, DS].to_numpy()
        im = ax.imshow(M, cmap=cmap, vmin=40, vmax=100)
        for i in range(4):
            for j in range(4):
                v = M[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v >= 78 else INK, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(DS, rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(DS if ax is axes[0] else [""] * 4, fontsize=7)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("test dataset", fontsize=7)
        if ax is axes[0]: ax.set_ylabel("training dataset", fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("macro-F1 (%)", fontsize=7, color=INK2); cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)
    fig.savefig(O / "fig_heatmaps.pdf"); fig.savefig(O / "fig_heatmaps.png", dpi=200)


def tradeoff(A, O):
    t = pd.read_csv(A / "t4_ablation.csv", header=[0, 1], index_col=[0, 1])
    pts = {  # (group, variant) -> label
        ("frozen", "shot_lprop_auto"): "SHOT→LP", ("frozen", "t3a_ens"): "T3A-ens",
        ("frozen", "cg_soft"): "CG-soft", ("frozen", "cg_hard"): "CG-hard",
        ("finetuned", "ft"): "LP-FT", ("finetuned", "ft_shot_lprop_auto"): "LP-FT+SHOT→LP",
        ("finetuned", "ft_t3a_ens"): "LP-FT+T3A-ens", ("finetuned", "ft_cg_soft"): "LP-FT+CG-soft",
        ("finetuned", "ft_cg_hard"): "LP-FT+CG-hard"}
    offs = {"SHOT→LP": (4, -9), "T3A-ens": (-5, -10), "CG-soft": (4, 5), "CG-hard": (5, -6),
            "LP-FT": (-4, 5), "LP-FT+SHOT→LP": (-4, 5), "LP-FT+T3A-ens": (5, -4),
            "LP-FT+CG-soft": (6, -1), "LP-FT+CG-hard": (6, -5)}
    fig, ax = plt.subplots(figsize=(4.8, 2.7), constrained_layout=True)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0); ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
    ax.grid(color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    for grp, name, marker in [("frozen", "frozen features", "o"), ("finetuned", "LP-FT features", "s")]:
        xs, ys = [], []
        for (g, v), lab in pts.items():
            if g != grp: continue
            x, y = t.loc[(g, v), ("ID", "mean")], t.loc[(g, v), ("OOD", "mean")]
            xs.append(x); ys.append(y)
            ha = "right" if offs[lab][0] < 0 else "left"
            ax.annotate(lab, (x, y), xytext=offs[lab], textcoords="offset points", fontsize=6.3,
                        color=INK2, ha=ha)
        ax.scatter(xs, ys, s=34, marker=marker, color=SERIES["frozen" if grp == "frozen" else "fine-tuned"],
                   edgecolor="#fcfcfb", linewidth=1.0, label=name, zorder=3)
    ax.scatter([0], [0], s=34, marker="D", color=INK2, edgecolor="#fcfcfb", linewidth=1.0, zorder=3, label="B0 (linear probe)")
    ax.set_xlabel("Δ in-domain macro-F1 (points)"); ax.set_ylabel("Δ OOD macro-F1 (points)")
    ax.set_xlim(-2.6, 1.9); ax.set_ylim(-0.4, 5.4)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, handletextpad=0.3)
    fig.savefig(O / "fig_tradeoff.pdf"); fig.savefig(O / "fig_tradeoff.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="../paperwrite/analysis")
    ap.add_argument("--out", default="../paperwrite/paper/figures")
    a = ap.parse_args()
    A, O = Path(a.analysis), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    heatmaps(A, O); tradeoff(A, O); print("figures ->", O)
