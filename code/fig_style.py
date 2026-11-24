"""Publication style following figures4papers (github.com/ChenLiu-1996/figures4papers, CC BY-NC 4.0):
PALETTE, FigureStyle, apply_publication_style, make_grouped_bar, make_heatmap, finalize_figure
(re-implemented from its documented API, scientific-figure-making/references/api.md)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = [PALETTE[k] for k in ["blue_main", "green_3", "red_strong", "teal", "violet", "neutral"]]


@dataclass
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle = FigureStyle()):
    plt.rcParams.update({
        "font.size": style.font_size, "font.family": "sans-serif", "font.sans-serif": list(style.font_family),
        "axes.linewidth": style.axes_linewidth, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": style.font_size, "axes.titlesize": style.font_size,
        "xtick.labelsize": style.font_size - 2, "ytick.labelsize": style.font_size - 2,
        "xtick.major.width": style.axes_linewidth * 0.8, "ytick.major.width": style.axes_linewidth * 0.8,
        "legend.frameon": False, "legend.fontsize": style.font_size - 2,
        "text.usetex": style.use_tex, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def make_grouped_bar(ax, categories, series, labels, ylabel="Value", colors=None, annotate=False, fmt="{:.1f}"):
    series = [np.asarray(s, float) for s in series]
    colors = colors or DEFAULT_COLORS
    n, x = len(series), np.arange(len(categories))
    w = 0.8 / n
    bars = None
    for i, (s, lab) in enumerate(zip(series, labels)):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, s, w * 0.92, label=lab, color=colors[i % len(colors)],
                      edgecolor="white", linewidth=1.0)
        if annotate:
            for b, v in zip(bars, s):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, fmt.format(v), ha="center",
                        va="bottom", fontsize=plt.rcParams["font.size"] - 5)
    ax.set_xticks(x); ax.set_xticklabels(categories); ax.set_ylabel(ylabel)
    return bars


def make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap=None, cbar_label=None, annotate=False,
                 vmin=None, vmax=None, fmt="{:.1f}", bold_diag=False):
    cmap = cmap or LinearSegmentedColormap.from_list("blues", ["#FFFFFF", PALETTE["blue_secondary"], PALETTE["blue_main"]])
    m = np.asarray(matrix, float)
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    if x_labels is not None: ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(x_labels)
    if y_labels is not None: ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(y_labels)
    if annotate:
        lo, hi = (vmin if vmin is not None else m.min()), (vmax if vmax is not None else m.max())
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                dark = (m[i, j] - lo) / max(hi - lo, 1e-9) > 0.55
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", color="white" if dark else "black",
                        fontweight="bold" if bold_diag and i == j else "normal",
                        fontsize=plt.rcParams["font.size"] - 3)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    return im


def finalize_figure(fig, out_path, formats=("pdf", "png"), dpi=300, close=True, pad=0.05):
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(out.with_suffix(f".{f}"), dpi=dpi, bbox_inches="tight", pad_inches=pad)
    if close: plt.close(fig)
