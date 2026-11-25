"""Paper figures in the figures4papers style (see fig_style.py).

fig_pipeline.pdf     : LP-FT+CG pipeline with example images
fig_heatmaps.pdf     : B0 cross-dataset macro-F1 (ResNet-50, SigLIP)
fig_xai_masking.pdf  : region masking (full / object only / context only), three backbones
fig_xai.pdf          : occlusion-sensitivity examples (correct and wrong OOD predictions)
Usage: python make_figures_v2.py --analysis ../paperwrite/analysis --out ../paperwrite/paper/figures --root ../datasets
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from fig_style import PALETTE, FigureStyle, apply_publication_style, make_grouped_bar, make_heatmap, finalize_figure

DS = ["TrashNet", "RealWaste", "TrashBox", "Garbage12"]
P = PALETTE


def load_crop(root, path, size=224):
    im = Image.open(Path(root) / path).convert("RGB")
    s = size / min(im.size); im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
    L, T = (im.size[0] - size) // 2, (im.size[1] - size) // 2
    return im.crop((L, T, L + size, T + size))


# ------------------------------------------------------------------ pipeline
def box(ax, x0, y0, x1, y1, text, fc, ec, bold_first=True, fs=15):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=ec, lw=2.2, zorder=2))
    lines = text.split("\n")
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.18 if len(lines) > 1 else 0), lines[0], ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold_first else "normal", zorder=3)
    if len(lines) > 1:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 0.28, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=fs - 2.5, color="#333333", zorder=3, linespacing=1.15)


def arrow(ax, p, q, dashed=False, color="#444444", rad=0.0, label=None, lpos=0.5, loff=(0, 0.17)):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=20, lw=2.0, color=color, zorder=1,
                        linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        x, y = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=12.5, color="#333333", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.5))


def pipeline(A, O, root):
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=0))
    man = pd.read_csv(A / "manifest.csv")
    k = man[(man.keep == 1) & (man.core == 1)]
    src = k[(k.dataset == "TrashNet") & (k.label == "cardboard")].iloc[5].path
    occ = pd.read_csv(A / "xai" / "occlusion.csv")
    tgt = man.loc[occ[(occ.backbone == "siglip_b16") & (occ.true == "cardboard") & (occ.correct == 0)].row_id.iloc[0], "path"]
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.4); ax.axis("off")
    # lanes
    for y0, y1, lab, fc in [(3.55, 6.35, "Training on labelled source data", "#EEF3FA"),
                            (0.05, 3.25, "Test time (unlabelled target)", "#EEF6F7")]:
        ax.add_patch(FancyBboxPatch((0.05, y0), 15.9, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=fc, ec="none", zorder=0))
        ax.text(0.2, y1 - 0.22, lab, fontsize=14, fontweight="bold", color=P["blue_main"] if y0 > 3 else P["teal"], va="top")
    # thumbnails
    for path, (x0, y0), cap in [(src, (0.25, 3.75), "TrashNet"), (tgt, (0.25, 0.25), "RealWaste")]:
        ax.imshow(load_crop(root, path), extent=(x0, x0 + 1.75, y0, y0 + 1.75), zorder=2)
        ax.text(x0 + 0.875, y0 + 1.9, cap, ha="center", fontsize=12.5, color="#333333")
    blue_fc, blue_ec = "#D6E4F4", P["blue_main"]
    teal_fc, teal_ec = "#D9ECEE", P["teal"]
    # training lane
    box(ax, 2.55, 4.05, 5.15, 5.35, "Linear probe", blue_fc, blue_ec)
    box(ax, 5.8, 4.05, 9.0, 5.35, "LP-FT", blue_fc, blue_ec)
    box(ax, 9.65, 4.05, 12.25, 5.35, "New probe $f$", blue_fc, blue_ec)
    box(ax, 12.9, 4.05, 15.75, 5.35, "Source validation", "#ECECEC", "#7A7A7A")
    arrow(ax, (2.0, 4.7), (2.55, 4.7))
    arrow(ax, (5.15, 4.7), (5.8, 4.7), label="init", loff=(0, 0.25))
    arrow(ax, (9.0, 4.7), (9.65, 4.7))
    arrow(ax, (12.25, 4.7), (12.9, 4.7), dashed=True)
    # test lane
    box(ax, 2.55, 0.55, 5.35, 1.95, "Features, $p_0$", teal_fc, teal_ec)
    box(ax, 6.15, 0.55, 9.7, 1.95, "Cluster propagation", teal_fc, teal_ec)
    box(ax, 10.6, 0.55, 13.3, 1.95, "Confidence gate", "#FFF3B0", "#B8920A")
    ax.text(14.95, 1.25, "Prediction", ha="center", va="center", fontsize=15, fontweight="bold")
    arrow(ax, (2.0, 1.25), (2.55, 1.25))
    arrow(ax, (5.35, 1.25), (6.15, 1.25), label="$z$, $p_0$", loff=(0, 0.25))
    arrow(ax, (9.7, 1.25), (10.6, 1.25), label="$p_{\\mathrm{adapt}}$", loff=(0, 0.27))
    arrow(ax, (13.3, 1.25), (14.05, 1.25))
    # bypass of p0 and c to the gate, below the propagation box
    ax.plot([3.95, 3.95, 11.9], [0.55, 0.3, 0.3], color="#444444", lw=2.0, zorder=1)
    arrow(ax, (11.9, 0.3), (11.9, 0.55))
    ax.text(7.9, 0.14, "$p_0$, $c$", ha="center", va="center", fontsize=12, color="#333333")

    # cross-lane links drawn as right-angle (orthogonal) connectors
    def elbow(pts, dashed=False, label=None, lxy=None):
        ls = "--" if dashed else "-"
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color="#444444", lw=2.0, ls=ls, zorder=1)
        arrow(ax, pts[-2], pts[-1], dashed=dashed)
        if label:
            ax.text(*lxy, label, ha="center", va="center", fontsize=12.5, color="#333333", zorder=4,
                    bbox=dict(fc="white", ec="none", pad=0.5))
    # the fine-tuned backbone and its probe are applied to the target images
    elbow([(10.95, 4.05), (10.95, 3.4), (4.9, 3.4), (4.9, 1.95)], label="backbone + $f$", lxy=(7.9, 3.4))
    # hyper-parameters chosen on source validation
    elbow([(13.3, 4.05), (13.3, 2.6), (7.9, 2.6), (7.9, 1.95)], dashed=True, label="$k$, $\\alpha$", lxy=(10.6, 2.6))
    elbow([(14.6, 4.05), (14.6, 2.3), (12.5, 2.3), (12.5, 1.95)], dashed=True, label="$\\tau$", lxy=(14.6, 3.05))
    finalize_figure(fig, O / "fig_pipeline")


# ------------------------------------------------------------------ heatmaps
def heatmaps(A, O):
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=0))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), constrained_layout=True)
    order = ["TrashNet", "RealWaste", "TrashBox", "Garbage12"]
    # wider lightness range: light up to ~70, dark only near 100, so the 67-99 SigLIP cells stay distinguishable
    cmap = LinearSegmentedColormap.from_list("blues_wide", [(0.0, "#FFFFFF"), (0.25, "#E4EEF8"), (0.5, "#B9D1EC"),
                                                            (0.75, "#6E9DD3"), (1.0, P["blue_main"])])
    for ax, (bb, title) in zip(axes, [("resnet50", "ResNet-50"), ("siglip_b16", "SigLIP ViT-B/16")]):
        M = pd.read_csv(A / f"matrix_b0_{bb}.csv", index_col=0).loc[order, order].to_numpy()
        im = make_heatmap(ax, M, x_labels=order, y_labels=order if ax is axes[0] else [""] * 4,
                          annotate=True, vmin=40, vmax=100, bold_diag=True, cmap=cmap, dark_from=0.78)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("test dataset")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        if ax is axes[0]: ax.set_ylabel("training dataset")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02); cb.set_label("macro-F1 (%)"); cb.outline.set_visible(False)
    finalize_figure(fig, O / "fig_heatmaps")


# ------------------------------------------------------------------ XAI: region masking
def xai_masking(A, O):
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2.0))
    R = pd.read_csv(A / "xai" / "region_masking.csv")
    order = ["TrashNet", "Garbage12", "TrashBox", "RealWaste"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True, constrained_layout=True)
    for ax, (bb, name) in zip(axes, [("resnet50", "ResNet-50"), ("dinov2_b", "DINOv2-B"), ("siglip_b16", "SigLIP")]):
        d = R[R.backbone == bb].pivot(index="test_on", columns="condition", values="macro_f1").loc[order]
        make_grouped_bar(ax, ["TrashNet\n(in-domain)", "Garbage12", "TrashBox", "RealWaste"],
                         [d["original"], d["object only"], d["context only"]], ["full image", "object only", "context only"],
                         ylabel="macro-F1 (%)" if ax is axes[0] else "",
                         colors=[P["blue_main"], P["green_3"], P["red_strong"]])
        ax.set_title(name, fontweight="bold"); ax.set_ylim(0, 100); ax.axhline(20, color="#999999", lw=1.2, ls=":")
        ax.tick_params(axis="x", labelsize=13)
    axes[0].text(3.45, 22, "chance", fontsize=12, color="#777777", ha="right")
    axes[-1].legend(loc="upper right", fontsize=13)
    finalize_figure(fig, O / "fig_xai_masking")


# ------------------------------------------------------------------ XAI: occlusion examples
def xai_occlusion(A, O, root, n_each=3):
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=0))
    man = pd.read_csv(A / "manifest.csv"); occ = pd.read_csv(A / "xai" / "occlusion.csv")
    heat = LinearSegmentedColormap.from_list("h", ["#ffffff00", "#B6434280", "#B64342EE"])
    rows = [("siglip_b16", "SigLIP"), ("dinov2_b", "DINOv2-B")]
    fig, axes = plt.subplots(len(rows), 2 * n_each, figsize=(16, 6.0), constrained_layout=True)
    for r, (bb, name) in enumerate(rows):
        maps = json.load(open(A / "xai" / f"occlusion_maps_{bb}.json"))
        d = occ[occ.backbone == bb]
        pick = pd.concat([d[d.correct == 1].sort_values("conf", ascending=False).head(n_each),
                          d[d.correct == 0].sort_values("conf", ascending=False).head(n_each)])
        for c, (_, e) in enumerate(pick.iterrows()):
            ax = axes[r, c]
            ax.imshow(load_crop(root, man.loc[e.row_id, "path"]).convert("L"), cmap="gray", vmin=0, vmax=255)
            m = np.array(maps[str(e.row_id)]); m = m / max(m.max(), 1e-9)
            ax.imshow(np.kron(m, np.ones((38, 38)))[:224, :224], cmap=heat, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor(P["green_3"] if e.correct else P["red_strong"]); s.set_linewidth(5)
            ax.set_title(f"{e.true} → {e.pred}", fontsize=13.5)
            if c == 0: ax.set_ylabel(name, fontweight="bold")
    fig.text(0.25, -0.02, "correct predictions", ha="center", fontsize=14, color=P["green_3"], fontweight="bold")
    fig.text(0.75, -0.02, "wrong predictions", ha="center", fontsize=14, color=P["red_strong"], fontweight="bold")
    finalize_figure(fig, O / "fig_xai")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="../paperwrite/analysis")
    ap.add_argument("--out", default="../paperwrite/paper/figures")
    ap.add_argument("--root", default="../datasets")
    a = ap.parse_args()
    A, O = Path(a.analysis), Path(a.out)
    pipeline(A, O, a.root); heatmaps(A, O); xai_masking(A, O); xai_occlusion(A, O, a.root)
    print("figures ->", O)
