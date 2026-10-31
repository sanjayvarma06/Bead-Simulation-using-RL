"""Effect of the open loop of the Austin test reference (no new rollouts).

The extracted Austin reference ends 4.6 cm short of its start (Table 1). Every stored Austin test
trajectory is rescored with the same metric code against (a) the reference as used in the paper and
(b) the same reference closed by appending its first point; the largest change of any metric over
all controllers, training seeds and episodes is reported.

Usage: python experiments/closure_check.py
Output: results/final/closure_check.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_path  # noqa: E402
from metrics import trajectory_metrics  # noqa: E402

FINAL = ROOT / "results" / "final"
KEYS = ["rmse", "p95", "max_err", "geom010", "ordered_cov", "completed"]


def main():
    pts, brk = load_path("circuit_Austin")
    closed = np.vstack([pts, pts[:1]])
    df = pd.read_csv(FINAL / "episodes.csv")
    df = df[df.ref == "circuit_Austin"]
    worst = {k: 0.0 for k in KEYS}
    flips = 0
    per_ctrl = {}
    for (c, s), g in df.groupby(["controller", "train_seed"]):
        T = np.load(FINAL / "traj" / f"{c}__s{'None' if s == -1 else s}.npz")
        for ev in g.eval_seed:
            tr = T[f"circuit_Austin__{ev}"].astype(float)
            a = trajectory_metrics(pts, tr, brk)
            b = trajectory_metrics(closed, tr, brk)
            for k in KEYS:
                worst[k] = max(worst[k], abs(float(a[k]) - float(b[k])))
            flips += int(a["completed"] != b["completed"])
            per_ctrl.setdefault(c, []).append(100 * (b["rmse"] - a["rmse"]))
    out = {"closure_gap_m": float(np.linalg.norm(pts[-1] - pts[0])), "episodes": int(len(df)),
           "max_abs_change": worst, "completion_flips": flips,
           "max_rmse_change_cm_per_controller": {c: float(np.max(np.abs(v))) for c, v in per_ctrl.items()}}
    (FINAL / "closure_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("closure_gap_m", "episodes", "max_abs_change", "completion_flips")}, indent=1))


if __name__ == "__main__":
    main()
