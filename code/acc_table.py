"""Accuracy version of the main-results table (same runs, variants and averaging as Table 2).

Usage: python acc_table.py --results ../results --out ../paperwrite/analysis
Writes t3_main_acc.csv and prints the LaTeX body rows.
"""
import argparse
from pathlib import Path
import pandas as pd
from paper_numbers import BB, NAMES, read_runs

VARS = ["lp", "ft", "ft_cg_hard", "ft_cg_soft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../paperwrite/analysis")
    a = ap.parse_args()
    lf = read_runs(str(Path(a.results) / "kaggle_lpft_cg/merged/results_methods/lpftcg/parts/*.csv"))
    lf = lf[lf.variant.isin(VARS)]
    rows = []
    for (bb, v, s), g in lf.groupby(["backbone", "variant", "seed"]):
        sg, lo = g[g.protocol == "single"], g[g.protocol == "lodo"]
        rows.append({"backbone": bb, "variant": v, "seed": s, "ID": sg[sg.in_domain == 1].acc.mean(),
                     "OOD": sg[sg.in_domain == 0].acc.mean(), "LODO": lo.acc.mean()})
    r = pd.DataFrame(rows)
    m = r.groupby(["backbone", "variant"])[["ID", "OOD", "LODO"]].mean()
    m.reset_index().to_csv(Path(a.out) / "t3_main_acc.csv", index=False)

    def line(name, vals):  # vals[col][variant]
        cells = []
        for col in ["ID", "OOD", "LODO"]:
            x = [round(vals[col][v], 1) for v in VARS]
            best = max(x)
            cells.append(" & ".join(f"\\textbf{{{y:.1f}}}" if y == best else f"{y:.1f}" for y in x))
        return f"{name:<15} & " + " && ".join(cells) + " \\\\"

    for bb in BB:
        print(line(NAMES[bb], {c: {v: m.loc[(bb, v), c] for v in VARS} for c in ["ID", "OOD", "LODO"]}))
    mean = m.groupby(level="variant").mean()
    print(line("Mean", {c: {v: mean.loc[v, c] for v in VARS} for c in ["ID", "OOD", "LODO"]}))
    gap = (mean.ID - mean.OOD).loc[VARS].round(1)
    print("gap", gap.to_dict())
    d = mean.loc["ft_cg_hard"] - mean.loc["lp"]; ds = mean.loc["ft_cg_soft"] - mean.loc["lp"]
    print("hard - lp", d.round(2).to_dict(), " soft - lp", ds.round(2).to_dict(),
          " ft - lp", (mean.loc["ft"] - mean.loc["lp"]).round(2).to_dict())


if __name__ == "__main__":
    main()
