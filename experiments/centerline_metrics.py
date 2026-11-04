"""Tracking metrics against the TRUE circuit centerlines (no new rollouts).

The paper's metrics are measured against the reference extracted from the rasterized image. Here
the stored test trajectories (results/final/traj) are re-scored against the exact image of the
source F1TENTH centerline in arena coordinates (prepare_data.centerline_in_arena), with the same
metric code (metrics.trajectory_metrics): RMSE, P95 and maximum of the distance from each trajectory
sample to the centerline polyline, and Geom@0.10 over the centerline points.
Also reports the waypoint spacing of every extracted circuit reference.

Usage: python experiments/centerline_metrics.py
Output: results/final/centerline/episodes.csv, results/final/centerline/spacing.csv
"""
from __future__ import annotations
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_path, split_ids  # noqa: E402

FINAL = ROOT / "results" / "final"
OUT = FINAL / "centerline"


def true_centerline(ref: str) -> np.ndarray:
    from prepare_data import centerline_in_arena, CIRCUIT_DIR
    return centerline_in_arena(CIRCUIT_DIR / "centerlines" / f"{ref.replace('circuit_', '')}_centerline.csv")


def score(args):
    name, seed, ref, ev = args
    from metrics import trajectory_metrics
    traj = np.load(FINAL / "traj" / f"{name}__s{seed}.npz")[f"{ref}__{ev}"].astype(float)
    m = trajectory_metrics(true_centerline(ref), traj)
    return {"controller": name, "train_seed": -1 if seed is None else seed, "ref": ref, "eval_seed": ev,
            "rmse_true": m["rmse"], "p95_true": m["p95"], "max_true": m["max_err"], "geom010_true": m["geom010"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FINAL / "episodes.csv")
    df = df[df.family == "circuit"]
    jobs = [(r.controller, None if r.train_seed == -1 else int(r.train_seed), r.ref, int(r.eval_seed))
            for r in df[["controller", "train_seed", "ref", "eval_seed"]].drop_duplicates().itertuples()]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(score, jobs, chunksize=8))
    out = pd.DataFrame(rows).sort_values(["controller", "train_seed", "ref", "eval_seed"])
    out.to_csv(OUT / "episodes.csv", index=False)
    sp = []
    for ref in split_ids("train", "f1tenth_circuit") + split_ids("val", "f1tenth_circuit") + split_ids("test", "f1tenth_circuit"):
        pts, _ = load_path(ref)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        sp.append({"ref": ref, "n_points": len(pts), "spacing_median_m": float(np.median(d)),
                   "spacing_min_m": float(d.min()), "spacing_max_m": float(d.max())})
    pd.DataFrame(sp).to_csv(OUT / "spacing.csv", index=False)
    analyse(out)
    print(len(out), "episodes ->", OUT)


def analyse(out: pd.DataFrame):
    """Summary per controller and the comparisons of analyze.py, on the true-centerline RMSE."""
    from analyze import PRIMARY_TESTS, EXPLORATORY, bootstrap_ci, wilcoxon_exact
    prs = out.groupby(["controller", "train_seed", "ref"]).mean(numeric_only=True).reset_index()
    prs["family"] = "circuit"
    summ = prs.groupby(["controller", "train_seed"]).mean(numeric_only=True).groupby("controller").mean()
    summ[["rmse_true", "p95_true", "max_true", "geom010_true"]].to_csv(OUT / "summary.csv")
    rows = []
    for a, b in PRIMARY_TESTS + EXPLORATORY:
        if not ({a, b} <= set(prs.controller)):
            continue
        point, lo, hi, per_ref = bootstrap_ci(prs, a, b, "rmse_true")
        p, _ = wilcoxon_exact(per_ref)
        rows.append({"A": a, "B": b, "mean_diff_A_minus_B": point, "ci95_lo": lo, "ci95_hi": hi,
                     "A_better_on": int((per_ref < 0).sum()), "n_refs": len(per_ref), "wilcoxon_p": p,
                     "primary": (a, b) in PRIMARY_TESTS})
    pd.DataFrame(rows).to_csv(OUT / "comparisons.csv", index=False)


def analyse_only():
    analyse(pd.read_csv(OUT / "episodes.csv"))


if __name__ == "__main__":
    analyse_only() if "--analyse" in sys.argv else main()
