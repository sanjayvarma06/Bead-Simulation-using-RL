"""Bead dynamics, reward signs, seeding and termination."""
import numpy as np
import pytest
from environment import BeadEnvironment, PathExtractionError
from bead_gym_env import BeadTraceEnv


def straight(n=200, length=20.0):
    x = np.linspace(-length / 2 + 0.5, length / 2 - 0.5, n)
    return np.stack([x, np.zeros(n)], 1)


def env_on(points, seed=0):
    return BeadEnvironment(paths=[(points, ())], seed=seed)


def test_terminal_speed_matches_eq2():
    e = env_on(straight())
    e.reset(seed=0)
    e.pos_x, e.pos_y = -9.0, 0.0
    for _ in range(300):
        e._physics(np.array([1.0, 0.0], np.float32))
    v_star = 10.0 * 0.016 * (1 - 7 * 0.016) / (7 * 0.016)
    assert e.vel_x == pytest.approx(v_star, rel=1e-4)
    assert BeadEnvironment.terminal_speed() == pytest.approx(v_star)


def test_diagonal_speed_is_sqrt2_vstar():
    e = env_on(straight())
    e.reset(seed=0)
    e.pos_x = e.pos_y = -9.0
    for _ in range(300):
        e._physics(np.array([1.0, 1.0], np.float32))
    assert np.hypot(e.vel_x, e.vel_y) == pytest.approx(np.sqrt(2) * BeadEnvironment.terminal_speed(), rel=1e-4)


def test_actions_are_clipped():
    a, b = env_on(straight()), env_on(straight())
    a.reset(seed=1); b.reset(seed=1)
    a.step(np.array([5.0, -7.0])); b.step(np.array([1.0, -1.0]))
    assert (a.pos_x, a.pos_y, a.vel_x, a.vel_y) == (b.pos_x, b.pos_y, b.vel_x, b.vel_y)


def test_wall_contact_reflects_and_flags():
    e = env_on(straight())
    e.reset(seed=0)
    e.pos_x, e.vel_x = e.BOUNDARY - 1e-3, 1.0
    e._physics(np.array([1.0, 0.0], np.float32))
    assert e.pos_x == e.BOUNDARY and e.vel_x < 0 and e.hit_boundary
    _, r, _, _, info = e.step(np.array([1.0, 0.0]))
    assert info["reward_components"]["boundary"] < 0


def test_reward_signs():
    e = env_on(straight())
    e.reset(seed=0)
    e.pos_x, e.pos_y = e.path_points[0]
    _, _, _, _, info_on = e.step(np.array([1.0, 0.0]))
    c = info_on["reward_components"]
    assert c["precision"] > 0.2 and c["tracking_error"] <= 0 and c["step_cost"] < 0
    assert c["tangent"] > 0          # moving forward along +x tangent
    e2 = env_on(straight()); e2.reset(seed=0)
    e2.pos_y = 1.0
    _, _, _, _, info_off = e2.step(np.array([-1.0, 0.0]))
    c2 = info_off["reward_components"]
    assert c2["off_curve"] < 0 and c2["precision"] < 1e-6 and c2["tangent"] < 0


def test_waypoint_reward_when_advancing():
    e = env_on(straight())
    e.reset(seed=0)
    e.pos_x, e.pos_y = e.path_points[3]
    e.vel_x = e.vel_y = 0.0
    _, _, _, _, info = e.step(np.zeros(2))
    assert info["advanced_count"] > 0 and info["reward_components"]["waypoint"] > 0


def test_seeding_is_local_and_reproducible():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = BeadTraceEnv(paths=[(straight(), ())], seed=5)
    after = np.random.get_state()[1]
    assert np.array_equal(before, after), "env construction must not touch the global RNG"
    o1, _ = a.reset(seed=7)
    o2, _ = a.reset(seed=7)
    o3, _ = a.reset(seed=8)
    assert np.array_equal(o1, o2) and not np.array_equal(o1, o3)


def test_path_pool_sampling_is_seeded():
    pool = [(straight() + [0, k], ()) for k in range(5)]
    e = BeadEnvironment(paths=pool, seed=0)
    seq1 = []
    for s in range(10):
        e.reset(seed=s); seq1.append(e.path_points[0, 1])
    seq2 = []
    for s in range(10):
        e.reset(seed=s); seq2.append(e.path_points[0, 1])
    assert seq1 == seq2 and len(set(seq1)) > 1


def test_step_cap_scales_with_length():
    short = env_on(straight(50, 5.0))
    assert short.max_steps == BeadEnvironment.MAX_STEPS
    loop = np.stack([9 * np.cos(np.linspace(0, 6 * np.pi, 600)), 9 * np.sin(np.linspace(0, 6 * np.pi, 600))], 1)
    long_env = env_on(loop)
    L = np.sum(np.linalg.norm(np.diff(loop, axis=0), axis=1))
    assert long_env.max_steps >= 1.5 * L / (BeadEnvironment.terminal_speed() * BeadEnvironment.DT) - 1
    assert long_env.max_steps > BeadEnvironment.MAX_STEPS


def test_completion_terminates_with_bonus():
    e = env_on(straight(60, 6.0))
    e.reset(seed=0)
    done = False
    for _ in range(5000):
        wi = min(e.waypoint_idx + 1, len(e.path_points) - 1)
        v = e.path_points[wi] - [e.pos_x, e.pos_y]
        _, r, done, trunc, info = e.step(v / (np.linalg.norm(v) + 1e-9))
        if done or trunc:
            break
    assert done and not trunc and info["reward_components"]["completion"] == 50.0


def test_strict_mode_raises(tmp_path):
    with pytest.raises(PathExtractionError):
        BeadEnvironment(image_path=str(tmp_path / "missing.png"), strict=True)
    import cv2
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((200, 200), 255, np.uint8))
    with pytest.raises(PathExtractionError):
        BeadEnvironment(image_path=str(blank), strict=True)


def test_legacy_fallback_is_flagged(tmp_path):
    e = BeadEnvironment(image_path=str(tmp_path / "missing.png"))
    assert e.path_source == "default_circle"
