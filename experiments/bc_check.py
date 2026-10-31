"""Behaviour-cloning weight check for DAgger->PPO fine-tuning (VALIDATION only).

Compares lambda in {0.1, 1, 10} on training seeds 0-2 (lambda = 1 are the main runs in
runs/bead/dagger_ppo, the others runs/bead/dagger_ppo_bc{0.1,10}). For every run the checkpoint
is the one with the best validation score (as used everywhere); lambda is selected by the mean of
these validation scores over the three seeds. No test data are used.

Usage: python experiments/bc_check.py
Output: results/final/verification/bc_check.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS  # noqa: E402

LAMBDAS = {"0.1": "dagger_ppo_bc0.1", "1": "dagger_ppo", "10": "dagger_ppo_bc10"}
SEEDS = [0, 1, 2]


def best_row(path: Path) -> pd.Series:
    d = pd.read_csv(path)
    if "actor_frozen" in d:          # PPO logs only: rows during the critic warm-up are not candidates
        d = d[~d.actor_frozen.astype(str).str.lower().eq("true")]
    return d.loc[d.score.idxmax()]


def main():
    out = {"seeds": SEEDS, "lambdas": {}}
    for lam, mode in LAMBDAS.items():
        rows = [best_row(RUNS / "bead" / mode / f"seed{s}" / "val_log.csv") for s in SEEDS]
        out["lambdas"][lam] = {
            "dir": f"runs/bead/{mode}",
            "val_score_mean": float(np.mean([r.score for r in rows])),
            "val_rmse_circuits_cm_mean": float(np.mean([100 * r.rmse_circuits for r in rows])),
            "val_rmse_circuits_cm_per_seed": [float(100 * r.rmse_circuits) for r in rows],
            "best_timesteps": [int(r.timesteps) for r in rows]}
    out["selected_lambda"] = max(out["lambdas"], key=lambda k: out["lambdas"][k]["val_score_mean"])
    dag = [best_row(RUNS / "bead" / "dagger" / f"seed{s}" / "val_log.csv") for s in SEEDS]
    out["dagger_val_rmse_circuits_cm_mean"] = float(np.mean([100 * r.rmse_circuits for r in dag]))
    path = ROOT / "results" / "final" / "verification" / "bc_check.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
