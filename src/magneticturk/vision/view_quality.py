"""Predict how well a camera pose sees each square, before (or without) taking a picture.

quality = in view x angle x size x occlusion, each in [0, 1], per square:
- in view: the square's look point projects inside the image, away from the edges;
- angle: steep views are better; flatter than min_elevation_deg counts as useless;
- size: the square must appear at least full_quality_square_px wide for full quality;
- occlusion: a taller piece standing between the camera and the square may hide it.

The view planner calls this for candidate poses. vision/occupancy.py (stage 4) uses the
same prediction for real images and multiplies in what only the image shows (glare).
Occluders are the pieces of the position passed in: the robot knows where they stand
from the game history, even though it never recognises them.
"""

from __future__ import annotations

import math

import chess
import numpy as np

from magneticturk.config import PieceConfig, ViewQualityConfig
from magneticturk.geometry.transforms import CameraIntrinsics, Pose, project, square_centers


def view_quality(
    camera_pose: Pose,
    intrinsics: CameraIntrinsics,
    board: chess.Board,
    square_size_mm: float,
    pieces: PieceConfig,
    config: ViewQualityConfig,
) -> np.ndarray:
    """(64,) quality in [0, 1] for each square seen from camera_pose."""
    look = square_centers(square_size_mm)
    look[:, 2] = config.look_height_mm
    return (
        _in_view(look, camera_pose, intrinsics, config)
        * _angle_quality(look, camera_pose, config)
        * _size_quality(look, camera_pose, intrinsics, square_size_mm, config)
        * np.where(
            occluded_squares(look, camera_pose.position, board, square_size_mm, pieces),
            config.occluded_quality,
            1.0,
        )
    )


def occluded_squares(
    look: np.ndarray,
    eye: np.ndarray,
    board: chess.Board,
    square_size_mm: float,
    pieces: PieceConfig,
) -> np.ndarray:
    """(64,) True where the line of sight from a square's look point to the eye passes
    through another piece, modelled as an upright cylinder of the piece's height."""
    occupied = [s for s in chess.SQUARES if board.piece_type_at(s) is not None]
    if not occupied:
        return np.zeros(64, dtype=bool)
    centres = square_centers(square_size_mm)[occupied, :2]  # (N, 2)
    heights = np.array([pieces.heights_mm[board.piece_type_at(s)] for s in occupied])

    direction = eye - look  # (64, 3), from look point (t = 0) to eye (t = 1)
    # The ray is below a piece's top only while t < t_top.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_top = (heights[None, :] - look[:, 2:3]) / direction[:, 2:3]  # (64, N)
    t_top = np.where(direction[:, 2:3] > 0, np.clip(t_top, 0, 1), 1.0)

    # Closest approach of the ray (in plan view) to each piece's axis, for t in [0, t_top].
    d_xy = direction[:, None, :2]  # (64, 1, 2)
    offset = centres[None, :, :] - look[:, None, :2]  # (64, N, 2)
    d_len2 = np.maximum((d_xy**2).sum(axis=2), np.finfo(float).tiny)  # (64, 1)
    t_closest = np.clip((offset * d_xy).sum(axis=2) / d_len2, 0, t_top)
    miss = offset - t_closest[:, :, None] * d_xy
    blocked = (miss**2).sum(axis=2) < pieces.base_radius_mm**2  # (64, N)

    blocked[np.array(occupied)[None, :] == np.arange(64)[:, None]] = False  # not by itself
    return blocked.any(axis=1)


def _in_view(
    look: np.ndarray, pose: Pose, intrinsics: CameraIntrinsics, config: ViewQualityConfig
) -> np.ndarray:
    pixels, depth = project(look, pose, intrinsics)
    m = config.image_margin_px
    with np.errstate(invalid="ignore"):
        inside = (
            (pixels[:, 0] >= m)
            & (pixels[:, 0] <= intrinsics.width - m)
            & (pixels[:, 1] >= m)
            & (pixels[:, 1] <= intrinsics.height - m)
        )
    return (depth > 0) & inside


def _angle_quality(look: np.ndarray, pose: Pose, config: ViewQualityConfig) -> np.ndarray:
    ray = pose.position - look
    sin_elevation = ray[:, 2] / np.linalg.norm(ray, axis=1)
    sin_min = math.sin(math.radians(config.min_elevation_deg))
    return np.clip((sin_elevation - sin_min) / (1 - sin_min), 0, 1)


def _size_quality(
    look: np.ndarray,
    pose: Pose,
    intrinsics: CameraIntrinsics,
    square_size_mm: float,
    config: ViewQualityConfig,
) -> np.ndarray:
    _, depth = project(look, pose, intrinsics)
    with np.errstate(divide="ignore"):
        square_px = np.where(depth > 0, intrinsics.fx * square_size_mm / depth, 0)
    return np.clip(square_px / config.full_quality_square_px, 0, 1)
