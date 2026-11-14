import chess
import numpy as np
import pytest

from magneticturk.geometry.transforms import (
    CameraIntrinsics,
    Pose,
    look_at,
    project,
    square_center,
    square_centers,
)

SQUARE = 28.0


@pytest.fixture
def intrinsics():
    return CameraIntrinsics.from_fov(66, (1280, 960))


def test_square_centres():
    assert np.allclose(square_center(chess.A1, SQUARE), [14, 14, 0])
    assert np.allclose(square_center(chess.H1, SQUARE), [210, 14, 0])
    assert np.allclose(square_center(chess.A8, SQUARE), [14, 210, 0])
    centres = square_centers(SQUARE)
    assert centres.shape == (64, 3)
    assert np.allclose(centres[chess.E4], square_center(chess.E4, SQUARE))


def test_pose_inverse_round_trip():
    pose = look_at(np.array([50.0, -80.0, 250.0]), np.array([112.0, 112.0, 0.0]))
    points = np.random.default_rng(0).normal(0, 100, (10, 3))
    assert np.allclose(pose.inverse().apply(pose.apply(points)), points)


def test_look_at_rotation_is_proper():
    pose = look_at(np.array([0.0, 0.0, 300.0]), np.array([112.0, 112.0, 0.0]))
    r = pose.rotation
    assert np.allclose(r.T @ r, np.eye(3))
    assert np.isclose(np.linalg.det(r), 1)


def test_straight_down_view_keeps_rank_8_at_the_top(intrinsics):
    pose = look_at(np.array([112.0, 112.0, 300.0]), np.array([112.0, 112.0, 0.0]))
    pixels, depth = project(square_centers(SQUARE), pose, intrinsics)
    assert np.all(depth > 0)
    assert pixels[chess.H1, 0] > pixels[chess.A1, 0]  # files run left to right
    assert pixels[chess.A8, 1] < pixels[chess.A1, 1]  # rank 8 is higher in the image


def test_target_projects_to_image_centre(intrinsics):
    target = np.array([60.0, 150.0, 0.0])
    pose = look_at(np.array([200.0, -50.0, 220.0]), target)
    pixels, depth = project(target, pose, intrinsics)
    assert np.allclose(pixels[0], [intrinsics.cx, intrinsics.cy])
    assert np.isclose(depth[0], np.linalg.norm(target - pose.position))


def test_points_behind_the_camera(intrinsics):
    pose = look_at(np.array([0.0, 0.0, 100.0]), np.array([0.0, 0.0, 0.0]))
    _, depth = project(np.array([0.0, 0.0, 200.0]), pose, intrinsics)
    assert depth[0] < 0


def test_look_at_rejects_zero_length_view():
    with pytest.raises(ValueError):
        look_at(np.zeros(3), np.zeros(3))


def test_intrinsics_from_fov():
    intrinsics = CameraIntrinsics.from_fov(90, (1000, 800))
    assert np.isclose(intrinsics.fx, 500)
    assert (intrinsics.cx, intrinsics.cy) == (500, 400)


def test_pose_identity():
    pose = Pose(np.eye(3), np.zeros(3))
    assert np.allclose(pose.apply(np.array([1.0, 2.0, 3.0])), [1, 2, 3])
