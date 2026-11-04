"""Stanley baseline and dynamic single-track model."""
import numpy as np
import pytest
from rl_pure_pursuit_env import RLPurePursuitEnv
from dynamic_single_track import step as st_step, rhs


def straight(n=200):
    x = np.linspace(-9.0, 9.0, n)
    return [(np.stack([x, np.zeros(n)], 1), ())]


def circle(r=4.0, n=300):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(np.stack([r * np.cos(t), r * np.sin(t)], 1), ())]


def test_stanley_steers_back_towards_path():
    env = RLPurePursuitEnv(mode="stanley", paths=straight(), stanley_k=2.0, stanley_ks=1.0, strict=True)
    env.reset(seed=0)
    env.heading = 0.0
    env._env.pos_x, env._env.pos_y = -5.0, 0.3            # left of a path running along +x
    assert env._stanley_steering(np.array([-5.0, 0.3])) < 0      # steer right
    env._env.pos_y = -0.3
    assert env._stanley_steering(np.array([-5.0, -0.3])) > 0     # steer left
    assert env._stanley_steering(np.array([-5.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_stanley_tracks_a_circle():
    env = RLPurePursuitEnv(mode="stanley", paths=circle(), stanley_k=3.0, stanley_ks=0.5, v_cap=1.256, strict=True)
    env.reset(seed=0)
    errs, done, trunc = [], False, False
    while not (done or trunc):
        _, _, done, trunc, info = env.step(np.zeros(2))
        errs.append(info["tracking_error"])
    assert done and np.sqrt(np.mean(np.square(errs[100:]))) < 0.02


def test_dynamic_model_matches_kinematic_at_low_lateral_acceleration():
    """Constant small steering at moderate speed: yaw rate approaches the kinematic value."""
    v, d = 1.256, 0.05
    x = np.array([0, 0, d, v, 0, 0, 0], float)
    for _ in range(400):
        x = st_step(x, d, 0.0, 0.016)
    assert x[5] == pytest.approx(v / 0.33 * np.tan(d), rel=0.05)


def test_dynamic_rhs_kinematic_branch_below_switch_speed():
    f = rhs(np.array([0, 0, 0.1, 0.3, 0.0, 0.0, 0.0]), 0.0, 0.0)
    assert f[4] == pytest.approx(0.3 / (0.15875 + 0.17145) * np.tan(0.1))


def test_steering_rate_is_limited():
    x = np.array([0, 0, 0.0, 1.0, 0, 0, 0], float)
    x = st_step(x, 1.0, 0.0, 0.016)
    assert x[2] == pytest.approx(3.2 * 0.016, rel=1e-6)


def test_default_kinematic_rollouts_unchanged():
    """A stored test trajectory of tuned Fixed PP is reproduced bit-for-bit after the edits."""
    import json
    from pathlib import Path
    from common import ROOT, load_path, run_pp
    final = ROOT / "results" / "final"
    ctrl = [c for c in json.loads((final / "controllers.json").read_text()) if c["name"] == "pp_fixed_tuned"][0]
    kw = {k: v for k, v in ctrl["kw"].items() if k != "mode"}
    pts, brk = load_path("circuit_Montreal")
    _, ex = run_pp("fixed", pts, brk, 50000, **kw)
    old = np.load(final / "traj" / "pp_fixed_tuned__sNone.npz")["circuit_Montreal__50000"]
    assert np.array_equal(ex["traj"].astype(np.float32), old)
