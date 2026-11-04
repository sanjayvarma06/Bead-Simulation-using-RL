"""Rear-wheel feedback, position noise, rear-axle position of the dynamic car, fast bead, real scale."""
import numpy as np
import pytest
from rl_pure_pursuit_env import RLPurePursuitEnv


def straight(n=200):
    x = np.linspace(-9.0, 9.0, n)
    return [(np.stack([x, np.zeros(n)], 1), ())]


def circle(r=4.0, n=300):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(np.stack([r * np.cos(t), r * np.sin(t)], 1), ())]


def test_rwf_steers_back_towards_path():
    env = RLPurePursuitEnv(mode="rwf", paths=straight(), strict=True)
    env.reset(seed=0)
    env.heading, env.current_idx = 0.0, 50
    assert env._rwf_steering(np.array([env._env.path_points[50, 0], 0.3])) < 0    # left of path -> steer right
    assert env._rwf_steering(np.array([env._env.path_points[50, 0], -0.3])) > 0
    assert env._rwf_steering(np.array([env._env.path_points[50, 0], 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_rwf_feedforward_tracks_circle():
    env = RLPurePursuitEnv(mode="rwf", paths=circle(), rwf_ke=16, rwf_kpsi=4, v_cap=1.256, strict=True)
    env.reset(seed=0)
    errs, done, trunc = [], False, False
    while not (done or trunc):
        _, _, done, trunc, info = env.step(np.zeros(2))
        errs.append(info["tracking_error"])
    assert done and np.sqrt(np.mean(np.square(errs[100:]))) < 0.01


def test_position_noise_is_seeded_and_does_not_move_the_car():
    from common import run_pp
    pts, brk = circle()[0]
    a, ea = run_pp("fixed", pts, brk, 3, fixed_L=0.3, fixed_g=1.0, pos_noise=0.02)
    b, eb = run_pp("fixed", pts, brk, 3, fixed_L=0.3, fixed_g=1.0, pos_noise=0.02)
    c, ec = run_pp("fixed", pts, brk, 3, fixed_L=0.3, fixed_g=1.0)
    assert np.array_equal(ea["traj"], eb["traj"]) and not np.array_equal(ea["traj"], ec["traj"])
    assert np.max(np.linalg.norm(np.diff(ea["traj"], axis=0), axis=1)) < 0.1     # no teleporting


def test_dynamic_position_is_rear_axle():
    env = RLPurePursuitEnv(mode="fixed", paths=straight(), dynamics="dynamic_norate", v_cap=1.256, strict=True)
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.zeros(2))
    x = env.dyn_state
    rear = x[:2] - 0.17145 * np.array([np.cos(x[4]), np.sin(x[4])])
    assert np.allclose(rear, [env._env.pos_x, env._env.pos_y])


def test_bead_noise_fast_and_real_options():
    from common import run_bead, BeadPursuit, load_path
    pts, brk = circle()[0]
    m0, e0 = run_bead("expert", pts, brk, 1)
    m1, e1 = run_bead("expert", pts, brk, 1, pos_noise=0.02)
    m2, _ = run_bead("expert", pts, brk, 1, accel=25.0)
    assert not np.array_equal(e0["traj"], e1["traj"]) and m2["mean_speed"] > 2 * m0["mean_speed"]
    m3, _ = run_bead(BeadPursuit(0.15, 1.0, "profile"), pts, brk, 1, accel=25.0)
    assert m3["completed"] and m3["mean_speed"] > 2.0
    real, rb = load_path("circuit_IMS", "real")
    assert np.ptp(real[:, 0]) > 20                     # true scale, beyond the 20 m arena
    m4, e4 = run_bead("expert", real, rb, 1, arena=False)
    assert m4["completed"] and np.abs(e4["traj"]).max() > 10
