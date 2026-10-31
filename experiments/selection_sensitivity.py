"""Sensitivity of model selection to the selection score (validation / tuning logs only).

The selection score is mean Geom@0.10 (a fraction in [0, 1]) minus mean RMSE in metres, averaged with
equal weight over circuits and synthetic references (valeval.val_score). This script asks, for every
learned run and every PP grid, whether selecting by the same-weighted mean RMSE alone would have
chosen a different checkpoint / configuration, and how much validation RMSE the two choices differ by.

Usage: python experiments/selection_sensitivity.py
Output: results/final/verification/selection_sensitivity.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS  # noqa: E402
from evaluate import SEEDS  # noqa: E402

RUN_DIRS = [("bead", m) for m in ("dagger", "dagger_bpp", "ppo", "dagger_ppo", "ppo_acc", "dagger_ppo_acc")] + \
           [("rlpp", m) for m in ("joint", "ld_only", "joint_matched")]


def rmse_mean(r):
    return 0.5 * (r["rmse_circuits"] + r["rmse_synth"])


def main():
    out = {"runs": {}, "pp_grids": {}}
    for kind, mode in RUN_DIRS:
        for s in SEEDS:
            f = RUNS / kind / mode / f"seed{s}" / "val_log.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            if "actor_frozen" in d:
                d = d[~d.actor_frozen.astype(str).str.lower().eq("true")]
            d = d.reset_index(drop=True)
            i_score, i_rmse = int(d.score.idxmax()), int(d.apply(rmse_mean, axis=1).idxmin())
            out["runs"][f"{kind}/{mode}/seed{s}"] = {
                "same_choice": i_score == i_rmse,
                "val_rmse_cm_by_score": 100 * rmse_mean(d.loc[i_score]),
                "val_rmse_cm_by_rmse": 100 * rmse_mean(d.loc[i_rmse])}
    tune = json.loads((RUNS / "pp_tuning.json").read_text())
    for key in ("fixed_native", "adaptive_native", "fixed_matched", "adaptive_matched", "beadpp_matched"):
        rows = [r for r in tune[key]["grid"] if r["completed_all"] == 1.0]
        by_s, by_r = max(rows, key=lambda r: r["score"]), min(rows, key=rmse_mean)
        out["pp_grids"][key] = {"same_choice": by_s["params"] == by_r["params"],
                                "params_by_score": by_s["params"], "params_by_rmse": by_r["params"],
                                "rmse_cm_by_score": 100 * rmse_mean(by_s), "rmse_cm_by_rmse": 100 * rmse_mean(by_r)}
    n = list(out["runs"].values()) + list(out["pp_grids"].values())
    out["n_same"], out["n_total"] = sum(x["same_choice"] for x in n), len(n)
    diffs = [abs(x.get("val_rmse_cm_by_score", x.get("rmse_cm_by_score")) -
                 x.get("val_rmse_cm_by_rmse", x.get("rmse_cm_by_rmse"))) for x in n]
    out["max_rmse_diff_cm"] = max(diffs) if diffs else 0.0
    nz = sorted(x for x, y in zip(diffs, n) if not y["same_choice"])
    out["median_rmse_diff_cm_when_different"] = float(pd.Series(nz).median()) if nz else 0.0
    path = ROOT / "results" / "final" / "verification" / "selection_sensitivity.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_same", "n_total", "max_rmse_diff_cm")}), "->", path)


if __name__ == "__main__":
    main()
