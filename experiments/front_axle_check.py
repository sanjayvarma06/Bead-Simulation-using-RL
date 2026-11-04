"""Rear- versus front-axle tracking error of the car controllers (no new rollouts).

All car controllers are scored at the car's reference point, the rear axle (the state of the
kinematic bicycle). Stanley regulates the FRONT axle onto the path, Pure Pursuit the rear axle; in a
corner of radius R the two points are offset laterally by about W^2 / (2R). This script recomputes
the RMSE at the front axle from the stored test trajectories: for the kinematic bicycle the rear
axle moves along the heading, so front = rear + W * (unit direction of motion).

Usage: python experiments/front_axle_check.py
Output: results/final/front_axle.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, EVAL_SEEDS, load_path, split_ids  # noqa: E402
from geometry_utils import points_to_polyline_distance  # noqa: E402

FINAL = ROOT / "results" / "final"
W = 0.33
NAMES = ["pp_fixed_tuned", "pp_fixed_tuned_matched", "stanley_tuned", "stanley_tuned_matched"]


def main():
    rows = []
    for name in NAMES:
        T = np.load(FINAL / "traj" / f"{name}__sNone.npz")
        for ref in split_ids("test", "f1tenth_circuit"):
            pts, brk = load_path(ref)
            for ev in EVAL_SEEDS:
                rear = T[f"{ref}__{ev}"].astype(float)
                v = np.diff(rear, axis=0)
                n = np.linalg.norm(v, axis=1, keepdims=True)
                keep = n[:, 0] > 1e-9
                front = rear[1:][keep] + W * v[keep] / n[keep]
                er = points_to_polyline_distance(rear, pts, path_breaks=set(brk))
                ef = points_to_polyline_distance(front, pts, path_breaks=set(brk))
                rows.append({"controller": name, "ref": ref, "eval_seed": ev,
                             "rmse_rear": float(np.sqrt(np.mean(er ** 2))), "rmse_front": float(np.sqrt(np.mean(ef ** 2)))})
    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "front_axle.csv", index=False)
    s = df.groupby(["controller", "ref"]).mean(numeric_only=True).groupby("controller").mean()
    print((100 * s[["rmse_rear", "rmse_front"]]).round(2))


if __name__ == "__main__":
    main()
