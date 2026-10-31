"""Parallel deterministic evaluation of a saved policy on a list of reference ids.

Used for validation-based checkpoint selection (never on test ids) and by evaluate.py.
"""
from __future__ import annotations
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_MODEL_CACHE: dict = {}


def _worker(args):
    kind, model_path, pid, seed, kw = args
    import torch
    torch.set_num_threads(1)
    from common import load_path, run_bead, run_pp, actor_mean, vecnorm_obs_fn
    from stable_baselines3 import PPO
    pts, brk = load_path(pid, kw.get("path_set"))
    bead_kw = {k: kw[k] for k in ("pos_noise", "accel", "arena") if k in kw}
    if kind == "bead_geo":                 # geometric controllers on the bead plant (new object per episode)
        from common import ExpertK, BeadPursuit
        policy = (ExpertK(kw["k"], kw.get("anchor", "counter")) if kw["ctrl"] == "expert_k"
                  else BeadPursuit(kw["L_d"], kw["g"], kw["v_ref"]))
        m, extra = run_bead(policy, pts, brk, seed, **bead_kw)
    elif kind == "bead":
        if model_path == "expert":
            policy = "expert"
        else:
            key = (model_path, Path(model_path).stat().st_mtime)
            if key not in _MODEL_CACHE:
                _MODEL_CACHE.clear()
                _MODEL_CACHE[key] = PPO.load(model_path, device="cpu")
            model = _MODEL_CACHE[key]
            policy = lambda o: actor_mean(model, o)  # noqa: E731
        m, extra = run_bead(policy, pts, brk, seed, **bead_kw)
    else:
        model = norm = None
        if model_path is not None:
            key = (model_path, Path(model_path).stat().st_mtime)
            if key not in _MODEL_CACHE:
                _MODEL_CACHE.clear()
                _MODEL_CACHE[key] = PPO.load(model_path, device="cpu")
            model = _MODEL_CACHE[key]
            if kw.get("vecnorm"):
                norm = vecnorm_obs_fn(Path(kw["vecnorm"]))
        pp_kw = {k: v for k, v in kw.items() if k in ("fixed_L", "fixed_g", "v_cap", "adapt_a", "adapt_b", "stanley_k", "stanley_ks", "dynamics",
                                                       "rwf_ke", "rwf_kpsi", "pos_noise", "max_steer", "arena", "L_max", "g_max")}
        m, extra = run_pp(kw["mode"], pts, brk, seed, model=model, obs_norm=norm, **pp_kw)
    m["id"] = pid
    m["seed"] = seed
    return m, (extra if kw.get("keep_traj") else None)


class Evaluator:
    def __init__(self, workers: int = 8):
        self.pool = ProcessPoolExecutor(max_workers=workers)

    def run(self, kind: str, model_path, ids, seeds, **kw):
        jobs = [(kind, str(model_path) if model_path not in (None, "expert") else model_path, pid, s, kw)
                for pid in ids for s in seeds]
        return list(self.pool.map(_worker, jobs, chunksize=1))

    def close(self):
        self.pool.shutdown()


def val_score(rows) -> dict:
    """Selection score on validation references: mean Geom@0.10 minus mean RMSE [m],
    with circuits and synthetic references weighted equally."""
    import numpy as np
    circ = [r for r in rows if r["id"].startswith("circuit_")]
    syn = [r for r in rows if not r["id"].startswith("circuit_")]
    def part(rs):
        return float(np.mean([r["geom010"] for r in rs]) - np.mean([r["rmse"] for r in rs])) if rs else None
    parts = [p for p in (part(circ), part(syn)) if p is not None]
    return {
        "score": float(np.mean(parts)),
        "geom010_circuits": float(np.mean([r["geom010"] for r in circ])) if circ else float("nan"),
        "rmse_circuits": float(np.mean([r["rmse"] for r in circ])) if circ else float("nan"),
        "completed_circuits": float(np.mean([r["completed"] for r in circ])) if circ else float("nan"),
        "geom010_synth": float(np.mean([r["geom010"] for r in syn])) if syn else float("nan"),
        "rmse_synth": float(np.mean([r["rmse"] for r in syn])) if syn else float("nan"),
        "time_circuits": float(np.mean([r["time_s"] for r in circ])) if circ else float("nan"),
    }
