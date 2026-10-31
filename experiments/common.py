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


def load_path(pid: str, path_set: str | None = None):
    """Reference of `pid` from the main cache, or from datasets/path_cache_<path_set> (e.g. "real":
    true-scale centerlines, "dense": 1000-point re-extraction)."""
    d = np.load((CACHE if not path_set else CACHE.parent / f"path_cache_{path_set}") / f"{pid}.npz")
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
def expert_action(core: BeadEnvironment, k: int = 1) -> np.ndarray:
    """Geometric expert of Eq. (4): unit vector towards waypoint w_t + k (k = 1 in the paper)."""
    if len(core.path_points) == 0:
        return np.zeros(2, np.float32)
    wi = min(core.waypoint_idx, len(core.path_points) - 1)
    ti = min(wi + k, len(core.path_points) - 1)
    vec = core.path_points[ti] - np.array([core.pos_x, core.pos_y])
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return np.zeros(2, np.float32)
    return np.clip(vec / n, -1.0, 1.0).astype(np.float32)


class ExpertK:
    """Expert of Eq. (4) aiming k waypoints ahead instead of one.

    anchor='counter': k indices past the environment's waypoint counter w_t (Eq. 4 as defined).
    anchor='nearest': k indices past the nearest reference point, tracked with the forward
    window of the PP controllers, so that the target cannot lock when w_t stops advancing.
    """

    def __init__(self, k: int, anchor: str = "counter"):
        self.k, self.anchor, self.idx = int(k), anchor, 0

    def core_action(self, core: BeadEnvironment) -> np.ndarray:
        if self.anchor == "counter":
            return expert_action(core, self.k)
        pts = core.path_points
        N = len(pts)
        if N == 0:
            return np.zeros(2, np.float32)
        pos = np.array([core.pos_x, core.pos_y])
        lo, hi = max(0, self.idx - 3), min(N, self.idx + 22)
        self.idx = lo + int(np.argmin(np.hypot(pts[lo:hi, 0] - pos[0], pts[lo:hi, 1] - pos[1])))
        vec = pts[min(self.idx + self.k, N - 1)] - pos
        n = float(np.linalg.norm(vec))
        return (vec / n).astype(np.float32) if n > 1e-8 else np.zeros(2, np.float32)


class BeadPursuit:
    """The Pure Pursuit law of the car (Eq. 9), driven on the bead plant.

    Nearest-point window and lookahead search are those of RLPurePursuitEnv / PurePursuitController.
    The heading is the direction of the bead's velocity; the PP curvature gamma = g 2 y' / L_d^2 is
    realised as a lateral acceleration v^2 gamma, and a longitudinal acceleration holds the speed at
    v_ref (exact one-step inverse of the damped dynamics, Eq. 1). Steering has priority: when the
    per-axis acceleration bound is reached, the longitudinal part is reduced first.
    """

    def __init__(self, L_d: float, g: float, v_ref):
        """v_ref: constant reference speed [m/s], or "profile" for the car's curvature-based speed
        profile (RLPurePursuitEnv: v = clip(4.2/sqrt(1+2.5|kappa|), 1.8, 5.0) with braking and
        acceleration passes), so that the bead can be run at the car's native pace."""
        from pure_pursuit import PurePursuitController
        self.L_d, self.g = float(L_d), float(g)
        self.v_ref = v_ref if v_ref == "profile" else float(v_ref)
        self.profile = None
        self.pp = PurePursuitController()
        self.idx = 0

    def _speed(self, pts) -> float:
        if self.v_ref != "profile":
            return self.v_ref
        if self.profile is None:
            from pure_pursuit import compute_path_curvature, compute_friction_speed_profile
            closed = float(np.linalg.norm(pts[0] - pts[-1])) < 0.5
            k = compute_path_curvature(pts, is_closed=closed)
            self.profile = compute_friction_speed_profile(k, path_points=pts, is_closed=closed, v_base=4.2,
                                                          v_min=1.8, v_max=5.0, alpha=2.5, a_brake=3.5, a_accel=2.5)
        return float(self.profile[self.idx])

    def core_action(self, core: BeadEnvironment) -> np.ndarray:
        pts = core.path_points
        N = len(pts)
        if N == 0:
            return np.zeros(2, np.float32)
        pos = np.array([core.pos_x, core.pos_y])
        lo, hi = max(0, self.idx - 3), min(N, self.idx + 22)
        self.idx = lo + int(np.argmin(np.hypot(pts[lo:hi, 0] - pos[0], pts[lo:hi, 1] - pos[1])))
        look, _ = self.pp.find_lookahead_point(pos, pts, self.idx, self.L_d, is_closed=False)
        vel = np.array([core.vel_x, core.vel_y])
        v = float(np.linalg.norm(vel))
        to_look = np.asarray(look, float) - pos
        if v < 0.05:                                   # at rest: no heading yet, accelerate towards the lookahead point
            n = float(np.linalg.norm(to_look))
            return (to_look / n).astype(np.float32) if n > 1e-8 else np.zeros(2, np.float32)
        t_hat = vel / v
        n_hat = np.array([-t_hat[1], t_hat[0]])
        y_body = float(to_look @ n_hat)
        gamma = self.g * 2.0 * y_body / self.L_d ** 2
        a_n = v * v * gamma * n_hat
        k = core.FRICTION * core.DT
        a_t = (self._speed(pts) / (1.0 - k) - v) / core.DT * t_hat
        u_n, u_t = a_n / core.ACCEL, a_t / core.ACCEL
        if np.max(np.abs(u_n)) >= 1.0:                 # lateral demand alone saturates: keep its direction
            return (u_n / np.max(np.abs(u_n))).astype(np.float32)
        lo_s, hi_s = 0.0, 1.0                          # largest share s of the longitudinal part that fits the box
        if np.max(np.abs(u_n + u_t)) > 1.0:
            for _ in range(30):
                mid = 0.5 * (lo_s + hi_s)
                lo_s, hi_s = (mid, hi_s) if np.max(np.abs(u_n + mid * u_t)) <= 1.0 else (lo_s, mid)
            hi_s = lo_s
        return np.clip(u_n + hi_s * u_t, -1.0, 1.0).astype(np.float32)


def actor_mean(model, obs: np.ndarray) -> np.ndarray:
    """Deterministic action = mean of the Gaussian policy (no exploration noise)."""
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        feats = model.policy.extract_features(x)
        latent = model.policy.mlp_extractor.forward_actor(feats)
        return model.policy.action_net(latent)[0].cpu().numpy().astype(np.float32)


def run_bead(policy, points, breaks, seed: int, keep_actions: bool = False, reward_mode: str = "base",
             pos_noise: float = 0.0, accel=None, arena: bool = True):
    """Roll out a bead controller. `policy` is 'expert', an object with `core_action(core)`
    (geometric controllers that read the simulator state), or a callable obs -> action.
    Robustness options: `pos_noise` [m] adds Gaussian noise to the position the controller sees
    (observation and geometric control; the bead moves from its true position); `accel` overrides
    the acceleration bound a_max (faster bead); `arena=False` removes the wall (real-scale circuits)."""
    env = BeadTraceEnv(paths=[(points, breaks)], seed=seed, strict=True, reward_mode=reward_mode)
    core = env._env
    if accel is not None:
        core.ACCEL = float(accel)                # instance override of the class constant
    if not arena:
        core.BOUNDARY = 1e9
    obs, _ = env.reset(seed=seed)
    noise_rng = np.random.default_rng(seed + 7919)
    traj = [(core.pos_x, core.pos_y)]
    acts, comps = [], {}
    hits = 0
    done = trunc = False
    info: dict = {}
    total_r = 0.0
    while not (done or trunc):
        true_xy = (core.pos_x, core.pos_y)
        if pos_noise > 0:
            n = noise_rng.normal(0.0, pos_noise, 2)
            core.pos_x, core.pos_y = true_xy[0] + n[0], true_xy[1] + n[1]
            obs = np.clip(core._observe(), -1.0, 1.0).astype(np.float32)
        if hasattr(policy, "core_action"):
            a = policy.core_action(core)
        else:
            a = expert_action(core) if policy == "expert" else policy(obs)
        core.pos_x, core.pos_y = true_xy
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
           adapt_b: float = 0.25, stanley_k: float = 2.0, stanley_ks: float = 1.0, dynamics: str = "kinematic",
           rwf_ke: float = 4.0, rwf_kpsi: float = 2.0, pos_noise: float = 0.0, max_steer: float = 1.10,
           arena: bool = True, L_max: float = 1.20, g_max: float = 1.40):
    """Roll out a PP-family (or Stanley) controller on the kinematic or dynamic car. `obs_norm` is a callable applied to observations
    before they reach an RL-PP policy (VecNormalize statistics from training)."""
    from rl_pure_pursuit_env import RLPurePursuitEnv
    env = RLPurePursuitEnv(mode=mode, seed=seed, paths=[(points, breaks)], fixed_L=fixed_L,
                           fixed_g=fixed_g, v_cap=v_cap, strict=True,
                           adapt_a=adapt_a, adapt_b=adapt_b, stanley_k=stanley_k, stanley_ks=stanley_ks,
                           dynamics=dynamics, rwf_ke=rwf_ke, rwf_kpsi=rwf_kpsi, pos_noise=pos_noise,
                           max_steer=max_steer, arena=arena, L_max=L_max, g_max=g_max)
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
