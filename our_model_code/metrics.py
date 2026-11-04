"""Trajectory metrics shared by every controller.

All quantities are computed post hoc from the executed trajectory X = (x_0, ..., x_T)
(positions in metres, x_0 = start position) and the ordered reference P = (p_0, ..., p_{N-1}).

  d_ref(p)  = min distance from reference point p to the trajectory polyline X
  e(x)      = min distance from trajectory sample x to the reference polyline P,
              excluding segments that bridge two disjoint components (path breaks)

  Geom@tau            = mean_p 1[d_ref(p) <= tau]          (reference-point coverage, order-free)
  RMSE / P95 / max    = statistics of e(x_t) over t = 0..T (tracking error)
  strict progress     = j*/(N-1), where the pointer j starts at 0 and, at every sample x_t,
                        advances to the largest c in (j, j+max_skip] (c not a break index)
                        with ||x_t - p_c|| <= rho. This is exactly the rule the bead
                        environment uses internally (its termination criterion), now applied
                        to every controller. One long excursion stalls it permanently.
  ordered coverage    = K/N, where K is the largest number of reference points
                        i_1 < i_2 < ... < i_K that are each visited (||x_t - p_i|| <= rho) at
                        non-decreasing times t_1 <= ... <= t_K. Skipped geometry and
                        out-of-order traversal lower it; a temporary excursion does not
                        forfeit the rest of the path. Computed exactly by dynamic programming.
  completed           = ordered coverage >= COMPLETION_THRESHOLD
"""
from __future__ import annotations
import numpy as np
from geometry_utils import points_to_polyline_distance

SEQ_RADIUS = 0.18          # = BeadEnvironment.COV_RADIUS
SEQ_MAX_SKIP = 4
COMPLETION_THRESHOLD = 0.95
OFF_PATH_TOL = 0.10


def ordered_coverage(path, traj, rho: float = SEQ_RADIUS) -> float:
    path = np.asarray(path, float)
    traj = np.asarray(traj, float)
    n = len(path)
    if n == 0 or len(traj) == 0:
        return 0.0
    INF = np.iinfo(np.int64).max
    best = np.full(n + 1, INF, dtype=np.int64)   # best[k] = earliest end time of a chain of length k
    best[0] = -1
    top = 0
    for i in range(n):
        d2 = (traj[:, 0] - path[i, 0]) ** 2 + (traj[:, 1] - path[i, 1]) ** 2
        visits = np.nonzero(d2 <= rho * rho)[0]
        if len(visits) == 0:
            continue
        prev = best[:top + 1]
        idx = np.searchsorted(visits, np.maximum(prev, 0), side="left")
        ok = idx < len(visits)
        cand = np.full(top + 1, INF, dtype=np.int64)
        cand[ok] = visits[idx[ok]]
        best[1:top + 2] = np.minimum(best[1:top + 2], cand)
        if best[top + 1] < INF:
            top += 1
    return top / n


def sequential_coverage(path, traj, breaks=(), rho: float = SEQ_RADIUS,
                        max_skip: int = SEQ_MAX_SKIP) -> float:
    path = np.asarray(path, float)
    traj = np.asarray(traj, float)
    n = len(path)
    if n < 2 or len(traj) == 0:
        return 0.0
    brk = set(int(b) for b in breaks)
    j = 0
    for x in traj:
        hi = min(n - 1, j + max_skip)
        if hi == j:
            continue
        cand = np.arange(j + 1, hi + 1)
        d = np.hypot(path[cand, 0] - x[0], path[cand, 1] - x[1])
        ok = [c for c, dc in zip(cand, d) if dc <= rho and int(c) not in brk]
        if ok:
            j = int(max(ok))
    return j / (n - 1)


def trajectory_metrics(path, traj, breaks=(), dt: float = 0.016, boundary_hits: int = 0) -> dict:
    path = np.asarray(path, float)
    traj = np.asarray(traj, float)
    if len(path) == 0:
        raise ValueError("empty reference path")
    if len(traj) == 0:
        raise ValueError("empty trajectory")
    d_ref = points_to_polyline_distance(path, traj)
    e = points_to_polyline_distance(traj, path, path_breaks=set(breaks))
    steps = len(traj) - 1
    time_s = steps * dt
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1) if steps > 0 else np.zeros(0)
    dist = float(seg.sum())
    if steps >= 2:
        vel = np.diff(traj, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        rms_acc = float(np.sqrt(np.mean(np.sum(acc ** 2, axis=1))))
    else:
        rms_acc = 0.0
    seq = sequential_coverage(path, traj, breaks)
    ordc = ordered_coverage(path, traj)
    return {
        "geom005": float(np.mean(d_ref <= 0.05)),
        "geom010": float(np.mean(d_ref <= 0.10)),
        "geom015": float(np.mean(d_ref <= 0.15)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mean_err": float(np.mean(e)),
        "p95": float(np.percentile(e, 95)),
        "max_err": float(np.max(e)),
        "off_path_frac": float(np.mean(e > OFF_PATH_TOL)),
        "sequential": float(seq),
        "ordered_cov": float(ordc),
        "completed": bool(ordc >= COMPLETION_THRESHOLD),
        "steps": int(steps),
        "time_s": float(time_s),
        "distance_m": dist,
        "mean_speed": float(dist / time_s) if time_s > 0 else 0.0,
        "peak_speed": float(seg.max() / dt) if steps > 0 else 0.0,
        "rms_accel": rms_acc,
        "boundary_hits": int(boundary_hits),
    }
