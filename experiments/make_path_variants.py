"""Reference variants for two robustness checks.

dense: the 8 test circuits re-extracted from the SAME rasterized images with the unchanged front end,
       but 1000 instead of 350 waypoints (datasets/path_cache_dense).
real:  all 23 F1TENTH centerlines at TRUE scale (metres, from the source CSV, no image), closed and
       resampled at 0.15 m arc-length spacing, about the arena references' median spacing, so that
       the lap counter (radius 0.18 m, at most 4 waypoints per step) keeps its meaning
       (datasets/path_cache_real).

Usage: python experiments/make_path_variants.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CACHE, split_ids, load_manifest  # noqa: E402
from environment import BeadEnvironment  # noqa: E402

REAL_SPACING = 0.15


def main():
    dense = CACHE.parent / "path_cache_dense"
    real = CACHE.parent / "path_cache_real"
    dense.mkdir(exist_ok=True)
    real.mkdir(exist_ok=True)
    src = {e["id"]: e["source"] for e in load_manifest()["entries"]}
    BeadEnvironment.WAYPOINT_BUDGET = 1000
    info = {"dense": {}, "real": {}}
    for pid in split_ids("test", "f1tenth_circuit"):
        env = BeadEnvironment(image_path=str(ROOT / src[pid]), seed=0, strict=True)
        p = env.path_points
        np.savez_compressed(dense / f"{pid}.npz", points=p, breaks=np.asarray(sorted(env.path_breaks), int))
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        info["dense"][pid] = {"n": len(p), "median_spacing_m": float(np.median(d))}
    BeadEnvironment.WAYPOINT_BUDGET = 350
    for split in ("train", "val", "test"):
        for pid in split_ids(split, "f1tenth_circuit"):
            csv = ROOT / "datasets/f1tenth_circuits/centerlines" / f"{pid.replace('circuit_', '')}_centerline.csv"
            xy = pd.read_csv(csv, comment="#", header=None).iloc[:, :2].values.astype(float)
            xy = np.vstack([xy, xy[:1]])                      # closed loop
            seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            L = float(seg.sum())
            n = int(round(L / REAL_SPACING)) + 1
            p = BeadEnvironment._resample(xy, n)
            np.savez_compressed(real / f"{pid}.npz", points=p, breaks=np.zeros(0, int))
            info["real"][pid] = {"split": split, "n": len(p), "length_m": round(L, 1),
                                 "extent_m": [round(float(np.ptp(p[:, 0])), 1), round(float(np.ptp(p[:, 1])), 1)]}
    (CACHE.parent / "path_variants.json").write_text(json.dumps(info, indent=1))
    r = info["real"]
    print("dense:", {k[8:]: v["n"] for k, v in info["dense"].items()})
    print("real lengths [m]:", min(v["length_m"] for v in r.values()), "-", max(v["length_m"] for v in r.values()),
          "points", min(v["n"] for v in r.values()), "-", max(v["n"] for v in r.values()))


if __name__ == "__main__":
    main()
