"""Paired comparisons on the dynamic car (8 test circuits; same bootstrap and exact Wilcoxon test as analyze.py),
against the extracted references (rmse) and the true centerlines (rmse_true).

Usage: python experiments/dynamic_stats.py
Output: results/final/robustness/dynamic_summary.csv, dynamic_comparisons.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
from analyze import bootstrap_ci, wilcoxon_exact  # noqa: E402
from robustness_eval import summarise  # noqa: E402

R = ROOT / "results" / "final" / "robustness"
PAIRS = [("fixed_retuned", "stanley_retuned"), ("fixed_retuned", "rlpp_retrained"), ("fixed_retuned_highg", "fixed_retuned"),
         ("fixed_retuned_highg", "stanley_retuned"), ("fixed_retuned_highg", "rlpp_retrained"),
         ("fixed_retuned", "fixed_transfer"), ("stanley_retuned", "stanley_transfer"), ("rlpp_retrained", "rlpp_transfer")]


def main():
    summ, comp = [], []
    for exp in ("dynamic_dt", "dynamic_dr"):
        d = pd.read_csv(R / f"{exp}.csv")
        s = summarise(d).reset_index()
        s.insert(0, "experiment", exp)
        summ.append(s)
        d["completed"] = d["completed"].astype(str).str.lower().eq("true").astype(float)
        prs = d.groupby(["controller", "train_seed", "ref"]).mean(numeric_only=True).reset_index()
        prs["family"] = "circuit"
        for a, b in PAIRS:
            for metric in ("rmse", "rmse_true"):
                point, lo, hi, per_ref = bootstrap_ci(prs, a, b, metric)
                p, _ = wilcoxon_exact(per_ref)
                comp.append({"experiment": exp, "A": a, "B": b, "metric": metric, "diff_cm": 100 * point, "lo": 100 * lo,
                             "hi": 100 * hi, "A_better_on": int((per_ref < 0).sum()), "p": p})
    pd.concat(summ).to_csv(R / "dynamic_summary.csv", index=False)
    c = pd.DataFrame(comp)
    c.to_csv(R / "dynamic_comparisons.csv", index=False)
    with pd.option_context("display.width", 200):
        print(c.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
