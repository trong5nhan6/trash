"""Merge per-run result files of a method and write summary.csv (mean ± std over seeds).

lpft writes one file per (backbone, seed, protocol) under results_methods/lpft/parts/;
the other methods already write results_methods/<method>/seed<k>/probe_results.csv.
Also builds results_methods/leaderboard.csv: one row per (method, backbone, variant).

Usage: python summarize.py --out outputs [--methods tta text clad rich ens lpft]
"""
import argparse
from pathlib import Path
import pandas as pd
from run_methods import summarize

COLS = ["backbone", "variant", "n_seeds", "in_domain_F1", "ood_F1", "gap", "TrashNet->RealWaste_F1",
        "LODO_avg_F1", "ood_ECE", "d_ood_vs_base", "d_lodo_vs_base"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--methods", nargs="+",
                    default=["tta", "text", "clad", "rich", "ens", "lpft", "gated", "lpftcg"])
    args = ap.parse_args()
    root = Path(args.out) / "results_methods"
    board = []
    for meth in args.methods:
        d = root / meth
        has_parts = (d / "parts").is_dir()      # finetune.py writes one file per (backbone, seed, protocol)
        if has_parts:
            files = [f for f in (d / "parts").glob("*.csv") if not f.name.endswith("_pred.csv.gz")]
        else:
            files = list(d.glob("seed*/probe_results.csv"))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files])
        if has_parts:   # only keep (backbone, seed) with all 8 protocols finished
            done = df.groupby(["backbone", "seed"]).train_on.nunique()
            ok = done[done == 8].index
            df = df.set_index(["backbone", "seed"]).loc[ok].reset_index() if len(ok) else df.iloc[0:0]
            if df.empty:
                print(f"{meth}: no complete (backbone, seed) yet"); continue
        s = summarize(df); s.insert(0, "method", meth)
        s.to_csv(d / "summary.csv", index=False)
        board.append(s)
        print(f"\n=== {meth} ===\n" + s[COLS].to_string(index=False))
    if board:
        b = pd.concat(board)
        b.to_csv(root / "leaderboard.csv", index=False)
        print("\n=== leaderboard (sorted by OOD macro-F1) ===")
        print(b.sort_values("ood_F1_mean", ascending=False)[["method"] + COLS].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
