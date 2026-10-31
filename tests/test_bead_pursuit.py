"""Geometric controllers on the bead plant (experiments/bead_pursuit.py)."""
import numpy as np
import pytest
from common import ExpertK, BeadPursuit, expert_action, run_bead


def circle(n=350, r=4.0):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def test_expert_k1_counter_is_eq4():
    pts = circle()
    a, _ = run_bead("expert", pts, (), seed=3)
    b, _ = run_bead(ExpertK(1, "counter"), pts, (), seed=3)
    assert a == b


def test_bead_pp_holds_v_ref_on_straight():
    x = np.linspace(-8, 8, 300)
    pts = np.stack([x, np.zeros_like(x)], 1)
    m, extra = run_bead(BeadPursuit(0.3, 1.0, 1.0), pts, (), seed=0)
    assert m["completed"]
    assert m["mean_speed"] == pytest.approx(1.0, abs=0.05)
    assert m["rmse"] < 0.01


def test_bead_pp_lateral_accel_matches_pp_curvature():
    """Velocity along +x, lookahead point to the left: command = v^2 gamma n + holding term."""
    from environment import BeadEnvironment
    x = np.linspace(0, 6, 200)
    pts = np.stack([x, 0.2 + 0 * x], 1)          # reference 0.2 m to the left of the bead
    e = BeadEnvironment(paths=[(pts, ())], seed=0)
    e.reset(seed=0)
    e.pos_x, e.pos_y, e.vel_x, e.vel_y = 0.0, 0.0, 1.0, 0.0
    c = BeadPursuit(L_d=0.5, g=1.0, v_ref=1.0)
    u = c.core_action(e)
    look_y = 0.2                                  # y' of the lookahead point in the velocity frame
    gamma = 2 * look_y / 0.5 ** 2
    assert u[1] == pytest.approx(min(1.0, 1.0 * gamma / e.ACCEL), rel=1e-3)
    assert u[0] > 0                               # forward term compensates damping


def test_dagger_episode_with_beadpp_teacher_labels_and_drives_like_beadpp():
    from train_bead import _dagger_episode
    pts = circle()
    teacher = {"L_d": 0.3, "g": 1.0, "v_ref": 1.0}
    O, A = _dagger_episode((None, pts, (), 1.0, 4, True, teacher))
    m, _ = run_bead(BeadPursuit(**teacher), pts, (), seed=4)
    assert len(O) == m["steps"]                    # teacher-only episode = the Bead-PP rollout
    O2, A2 = _dagger_episode((None, pts, (), 1.0, 4, True, None))
    assert np.allclose(np.linalg.norm(A2, axis=1), 1.0, atol=1e-5)   # default: expert labels


def test_expert_nearest_anchor_advances_without_counter():
    pts = circle()
    m, _ = run_bead(ExpertK(8, "nearest"), pts, (), seed=0)
    assert m["completed"]
