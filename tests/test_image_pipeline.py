"""Image -> ordered reference path on synthetic drawings with known geometry."""
import cv2
import numpy as np
import pytest
from environment import BeadEnvironment

SIZE = 400


def draw(tmp_path, name, fn):
    img = np.full((SIZE, SIZE), 255, np.uint8)
    fn(img)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return BeadEnvironment(image_path=str(p), strict=True, seed=0)


def scale():
    return 2 * BeadEnvironment.BOUNDARY / (SIZE - 1)


def assert_ordered(pts):
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert step.max() < 3 * np.median(step), "consecutive waypoints jump: path is not ordered"


def test_circle_is_closed_ordered_and_scaled(tmp_path):
    e = draw(tmp_path, "c.png", lambda im: cv2.circle(im, (200, 200), 120, 0, 5))
    p = e.path_points
    r = np.hypot(p[:, 0] - 0.5 * scale() * 0, p[:, 1])
    centre = ((SIZE - 1) / 2 - 200) * scale()
    r = np.hypot(p[:, 0] + centre, p[:, 1] - centre)
    assert np.abs(r - 120 * scale()).max() < 3 * scale()
    assert np.linalg.norm(p[0] - p[-1]) < 2 * scale()
    assert len(e.path_breaks) == 0
    assert_ordered(p)


def test_open_line_endpoints_and_axis_orientation(tmp_path):
    # Line from image top-left (50,50) to bottom-right (350,350): arena y must decrease.
    e = draw(tmp_path, "l.png", lambda im: cv2.line(im, (50, 50), (350, 350), 0, 5))
    p = e.path_points
    ends = sorted([p[0], p[-1]], key=lambda q: q[0])
    assert ends[0][1] > 0 > ends[1][1], "image y-axis must be inverted to a right-handed frame"
    assert np.linalg.norm(ends[0] - ends[1]) == pytest.approx(300 * np.sqrt(2) * scale(), rel=0.03)
    assert_ordered(p)


def test_waypoints_uniform_in_arc_length(tmp_path):
    e = draw(tmp_path, "s.png", lambda im: cv2.rectangle(im, (80, 80), (320, 320), 0, 5))
    step = np.linalg.norm(np.diff(e.path_points, axis=0), axis=1)
    assert step.std() / step.mean() < 0.35
    assert len(e.path_points) == pytest.approx(350, abs=5)


def test_two_strokes_produce_a_path_break(tmp_path):
    def f(im):
        cv2.line(im, (40, 100), (360, 100), 0, 5)
        cv2.line(im, (40, 300), (200, 300), 0, 5)
    e = draw(tmp_path, "two.png", f)
    assert len(e.path_breaks) == 1
    b = next(iter(e.path_breaks))
    assert b > len(e.path_points) / 2          # longer component first


def test_light_on_dark_polarity(tmp_path):
    img = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(img, (200, 200), 120, 255, 5)
    p = tmp_path / "inv.png"
    cv2.imwrite(str(p), img)
    e = BeadEnvironment(image_path=str(p), strict=True)
    centre = ((SIZE - 1) / 2 - 200) * scale()
    r = np.hypot(e.path_points[:, 0] + centre, e.path_points[:, 1] - centre)
    assert np.abs(r - 120 * scale()).max() < 3 * scale()
