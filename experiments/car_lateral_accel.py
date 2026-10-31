"""Lateral acceleration demanded of the kinematic car by tuned Fixed PP on the test circuits.

The kinematic bicycle model has no tyre forces, so nothing limits a_lat = v * yaw_rate = v^2 tan(delta) / W.
This script measures it exactly from the simulator (speed from the integrated positions, steering angle
from the controller) for the first evaluation episode of every test circuit, at native and matched speed,
to show how far the simulated manoeuvres exceed what tyre friction would allow (about mu g ~ 10 m/s^2).

Usage: python experiments/car_lateral_accel.py
Output: results/final/verification/car_lateral_accel.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, EVAL_SEEDS, load_path, run_pp, split_ids  # noqa: E402

WHEELBASE = 0.33


def main():
    tune = json.loads((RUNS / "pp_tuning.json").read_text())
    out = {}
    for key in ("fixed_native", "fixed_matched"):
        kw = {k: v for k, v in tune[key]["kwargs"].items() if k != "mode"}
        acc = []
        for ref in split_ids("test", "f1tenth_circuit"):
            pts, brk = load_path(ref)
            m, ex = run_pp("fixed", pts, brk, EVAL_SEEDS[0], **kw)
            traj, delta = ex["traj"], ex["D"]
            v = np.linalg.norm(np.diff(traj, axis=0), axis=1) / 0.016      # exact: position integrates v dt
            n = min(len(v), len(delta))
            acc.append(np.abs(v[:n] ** 2 * np.tan(delta[:n]) / WHEELBASE))
        a = np.concatenate(acc)
        out[key] = {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                    "p99": float(np.percentile(a, 99)), "max": float(a.max()),
                    "frac_above_10": float(np.mean(a > 10.0))}
        print(key, json.dumps(out[key]))
    path = ROOT / "results" / "final" / "verification" / "car_lateral_accel.json"
    path.write_text(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
