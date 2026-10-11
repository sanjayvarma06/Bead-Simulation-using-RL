"""Shared paths, data access and rollout helpers for the corrected experiments."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for sub in ("our_model_code", "research_paper_code"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from environment import BeadEnvironment  # noqa: E402
from bead_gym_env import BeadTraceEnv  # noqa: E402
from metrics import trajectory_metrics  # noqa: E402

CACHE = ROOT / "datasets" / "path_cache"
MANIFEST = ROOT / "datasets" / "split_manifest.json"
RUNS = ROOT / "runs"
EVAL_SEEDS = [50000 + 137 * k for k in range(5)]       # same start-perturbation seeds as the paper


# ----------------------------------------------------------------------------- data
def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def load_path(pid: str):
    d = np.load(CACHE / f"{pid}.npz")
    return d["points"].astype(float), tuple(int(b) for b in d["breaks"])


def split_ids(split: str, family: str | None = None) -> list[str]:
    """Ids of successfully extracted references in a split (optionally one family)."""
    out = []
    for e in load_manifest()["entries"]:
        if e["split"] == split and e["extraction_ok"] and (family is None or e["family"] == family):
            out.append(e["id"])
    return out


def load_paths(ids):
    return [load_path(i) for i in ids]


# ----------------------------------------------------------------------------- bead controllers
def expert_action(core: BeadEnvironment) -> np.ndarray:
    """Geometric expert of Eq. (4): unit vector towards waypoint w_t + 1."""
    if len(core.path_points) == 0:
        return np.zeros(2, np.float32)
    wi = min(core.waypoint_idx, len(core.path_points) - 1)
    ti = min(wi + 1, len(core.path_points) - 1)
    vec = core.path_points[ti] - np.array([core.pos_x, core.pos_y])
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return np.zeros(2, np.float32)
    return np.clip(vec / n, -1.0, 1.0).astype(np.float32)


def actor_mean(model, obs: np.ndarray) -> np.ndarray:
    """Deterministic action = mean of the Gaussian policy (no exploration noise)."""
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        feats = model.policy.extract_features(x)
        latent = model.policy.mlp_extractor.forward_actor(feats)
        return model.policy.action_net(latent)[0].cpu().numpy().astype(np.float32)


def run_bead(policy, points, breaks, seed: int, keep_actions: bool = False):
    """Roll out a bead controller. `policy` is 'expert' or a callable obs -> action."""
    env = BeadTraceEnv(paths=[(points, breaks)], seed=seed, strict=True)
    obs, _ = env.reset(seed=seed)
    core = env._env
    traj = [(core.pos_x, core.pos_y)]
    acts, comps = [], {}
    hits = 0
    done = trunc = False
    info: dict = {}
    total_r = 0.0
    while not (done or trunc):
        a = expert_action(core) if policy == "expert" else policy(obs)
        a = np.clip(np.asarray(a, np.float32), -1.0, 1.0)
        acts.append(a)
        obs, r, done, trunc, info = env.step(a)
        total_r += r
        hits += int(info["hit_boundary"])
        for k, v in info["reward_components"].items():
            comps[k] = comps.get(k, 0.0) + float(v)
        traj.append((core.pos_x, core.pos_y))
    traj = np.asarray(traj)
    acts = np.asarray(acts)
    m = trajectory_metrics(points, traj, breaks, dt=core.DT, boundary_hits=hits)
    m.update({
        "env_reward": float(total_r),
        "env_sequential": float(info.get("coverage", 0.0)),
        "terminated": bool(done),
        "stalled": bool(info.get("stalled", False)),
        "action_change": float(np.mean(np.linalg.norm(np.diff(acts, axis=0), axis=1))) if len(acts) > 1 else 0.0,
    })
    m.update({f"rc_{k}": v for k, v in comps.items() if k != "total"})
    extra = {"traj": traj}
    if keep_actions:
        extra["actions"] = acts
    return m, extra


# ----------------------------------------------------------------------------- pure pursuit controllers
def run_pp(mode: str, points, breaks, seed: int, model=None, obs_norm=None,
           fixed_L: float = 0.55, fixed_g: float = 1.05, v_cap=None, adapt_a: float = 0.35,
           adapt_b: float = 0.25):
    """Roll out a PP-family controller. `obs_norm` is a callable applied to observations
    before they reach an RL-PP policy (VecNormalize statistics from training)."""
    from rl_pure_pursuit_env import RLPurePursuitEnv
    env = RLPurePursuitEnv(mode=mode, seed=seed, paths=[(points, breaks)], fixed_L=fixed_L,
                           fixed_g=fixed_g, v_cap=v_cap, strict=True,
                           adapt_a=adapt_a, adapt_b=adapt_b)
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    L, G, D = [], [], []
    hits = 0
    info: dict = {}
    while not (done or trunc):
        if model is not None:
            o = obs_norm(obs) if obs_norm is not None else obs
            a = model.predict(o, deterministic=True)[0]
        else:
            a = np.zeros(2, np.float32)
        obs, r, done, trunc, info = env.step(a)
        L.append(info["lookahead_distance"]); G.append(info["steering_gain"]); D.append(info["steering_angle"])
        hits += int(info["hit_boundary"])
    traj = np.asarray(env.trajectory)
    m = trajectory_metrics(points, traj, breaks, dt=env._env.DT, boundary_hits=hits)
    m.update({
        "terminated": bool(done),
        "action_change": float(np.mean(np.abs(np.diff(D)))) if len(D) > 1 else 0.0,
        "Ld_mean": float(np.mean(L)), "Ld_std": float(np.std(L)),
        "g_mean": float(np.mean(G)), "g_std": float(np.std(G)),
    })
    return m, {"traj": traj, "L": np.asarray(L), "G": np.asarray(G), "D": np.asarray(D)}


def vecnorm_obs_fn(stats_path: Path):
    """Return obs -> normalised obs using saved VecNormalize running statistics."""
    import pickle
    with open(stats_path, "rb") as f:
        st = pickle.load(f)
    mean, var, clip, eps = st["mean"], st["var"], st["clip_obs"], st["epsilon"]
    return lambda o: np.clip((o - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
