"""Metric definitions checked on toy trajectories with known answers."""
import numpy as np
import pytest
from metrics import trajectory_metrics, sequential_coverage


def line(n=101, length=10.0):
    x = np.linspace(0, length, n)
    return np.stack([x, np.zeros(n)], 1)


def test_exact_tracking_is_perfect():
    p = line()
    m = trajectory_metrics(p, p.copy(), dt=0.1)
    assert m["geom005"] == 1.0 and m["geom010"] == 1.0
    assert m["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["sequential"] == 1.0 and m["completed"]
    assert m["mean_speed"] == pytest.approx(1.0)       # 0.1 m per 0.1 s


def test_constant_lateral_offset():
    p = line()
    t = p + np.array([0.0, 0.07])
    m = trajectory_metrics(p, t)
    assert m["rmse"] == pytest.approx(0.07, abs=1e-9)
    assert m["max_err"] == pytest.approx(0.07, abs=1e-9)
    assert m["geom005"] == 0.0 and m["geom010"] == 1.0
    assert m["off_path_frac"] == 0.0


def test_half_traversal():
    p = line()
    m = trajectory_metrics(p, p[:51])
    # the last sample (x=5.0) is within rho=0.18 of the next waypoint (x=5.1)
    assert m["sequential"] == pytest.approx(0.51)
    assert m["geom010"] == pytest.approx(51 / 101, abs=0.02)
    assert not m["completed"]


def test_reverse_traversal_covers_geometry_but_not_sequence():
    """Reference-point coverage ignores order; sequential coverage does not."""
    p = line()
    m = trajectory_metrics(p, p[::-1])
    assert m["geom010"] == 1.0
    # only the waypoints within rho of the start region can be claimed (index <= 5)
    assert m["sequential"] <= 0.05


def test_shortcut_is_penalised_by_coverage_not_rmse_alone():
    # Square reference; trajectory cuts straight from first to last corner.
    side = np.linspace(0, 4, 41)
    sq = np.concatenate([np.stack([side, 0 * side], 1), np.stack([4 + 0 * side, side], 1)[1:],
                         np.stack([side[::-1], 4 + 0 * side], 1)[1:]])
    cut = np.stack([np.linspace(0, 0, 50), np.linspace(0, 4, 50)], 1)  # straight up the missing 4th side
    m = trajectory_metrics(sq, cut)
    assert m["geom010"] < 0.2
    assert m["sequential"] < 0.05


def test_path_breaks_are_not_counted_as_reference():
    a = line(11, 1.0)
    b = line(11, 1.0) + np.array([5.0, 0.0])
    p = np.vstack([a, b])
    bridge = np.stack([np.linspace(1.0, 5.0, 41), np.zeros(41)], 1)   # lies on the bridging segment
    with_break = trajectory_metrics(p, bridge, breaks=(11,))
    without = trajectory_metrics(p, bridge)
    assert without["max_err"] == pytest.approx(0.0, abs=1e-9)
    assert with_break["max_err"] == pytest.approx(2.0, abs=1e-6)


def test_sequential_pointer_cannot_skip_more_than_window():
    p = line(101, 10.0)                       # 0.1 m spacing
    jump = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    # the start sample may claim waypoint 1 (0.1 m away); the 5 m jump claims nothing
    assert sequential_coverage(p, jump) == 0.01


def test_acceleration_of_uniform_circle():
    r, w, dt = 2.0, 0.5, 0.01
    t = np.arange(0, 2 * np.pi / w, dt)
    traj = np.stack([r * np.cos(w * t), r * np.sin(w * t)], 1)
    m = trajectory_metrics(traj, traj, dt=dt)
    assert m["rms_accel"] == pytest.approx(r * w * w, rel=1e-3)


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        trajectory_metrics(np.zeros((0, 2)), line())
    with pytest.raises(ValueError):
        trajectory_metrics(line(), np.zeros((0, 2)))


def test_ordered_coverage_basic_cases():
    from metrics import ordered_coverage
    p = line()
    assert ordered_coverage(p, p) == 1.0
    assert ordered_coverage(p, p[::-1]) <= 0.05          # out of order
    assert ordered_coverage(p, p[:51]) == pytest.approx(52 / 101, abs=0.011)


def test_excursion_stalls_strict_progress_but_not_ordered_coverage():
    """Temporary 0.5 m excursion over the middle 20 % of a line."""
    p = line()
    t = p.copy()
    t[40:61, 1] = 0.5
    m = trajectory_metrics(p, t)
    assert m["sequential"] < 0.45                        # pointer stalls at the excursion
    assert m["ordered_cov"] == pytest.approx(80 / 101, abs=0.03)
    assert m["geom010"] == pytest.approx(80 / 101, abs=0.03)


def test_ordered_coverage_matches_bruteforce_on_random_cases():
    from itertools import combinations
    from metrics import ordered_coverage
    rng = np.random.default_rng(0)
    for _ in range(30):
        p = rng.uniform(0, 1, (6, 2))
        t = rng.uniform(0, 1, (8, 2))
        rho = 0.3
        V = [set(np.nonzero(np.hypot(*(t - q).T) <= rho)[0]) for q in p]
        best = 0
        for k in range(1, 7):
            for idx in combinations(range(6), k):
                # feasible if times can be chosen non-decreasing
                cur, ok = -1, True
                for i in idx:
                    nxt = [v for v in sorted(V[i]) if v >= cur]
                    if not nxt:
                        ok = False; break
                    cur = nxt[0]
                if ok:
                    best = max(best, k)
        assert ordered_coverage(p, t, rho=rho) == pytest.approx(best / 6)
