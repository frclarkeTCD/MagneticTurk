"""The ONLY place coordinate frames are converted.

Frames: board (origin at a1 corner, mm), camera, robot base, gripper (tool centre point).
Provides: square name <-> board-frame position, board <-> base (from calibration),
camera <-> gripper (hand-eye calibration), and image pixel <-> board via marker pose.

Board frame: x along the files (a -> h), y along the ranks (1 -> 8), z up, in mm.
Camera frame follows OpenCV: x right, y down, z forward along the optical axis.
So far: square positions, camera poses in the board frame and pinhole projection.
Board <-> base and hand-eye come with calibration (stages 8 and 9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np


def square_center(square: chess.Square, square_size_mm: float) -> np.ndarray:
    """Centre of a square on the board surface, (x, y, 0) in the board frame."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    return np.array([(file + 0.5) * square_size_mm, (rank + 0.5) * square_size_mm, 0.0])


def square_centers(square_size_mm: float) -> np.ndarray:
    """(64, 3) centres of every square, indexed by python-chess square."""
    return np.array([square_center(s, square_size_mm) for s in chess.SQUARES])


@dataclass(frozen=True, eq=False)
class Pose:
    """Rigid transform from a child frame into a parent frame: p_parent = R @ p_child + t.

    For a camera pose in the board frame, t is the camera position on the board and the
    columns of R are the camera's x, y, z axes expressed in board coordinates.
    """

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    @property
    def position(self) -> np.ndarray:
        return self.translation

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Child-frame points (N, 3) or (3,) -> parent frame."""
        return np.asarray(points) @ self.rotation.T + self.translation

    def inverse(self) -> Pose:
        r_inv = self.rotation.T
        return Pose(r_inv, -r_inv @ self.translation)


def look_at(eye: np.ndarray, target: np.ndarray) -> Pose:
    """Camera pose in the board frame, at `eye`, pointing at `target`.

    The image's "up" is kept towards +y (rank 8) where possible, so views are not rolled.
    """
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target - eye
    if np.linalg.norm(forward) == 0:
        raise ValueError("Camera eye and target are the same point")
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up_hint) > 0.99:  # looking along y: use x as the hint instead
        up_hint = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return Pose(np.column_stack([right, down, forward]), eye)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, hfov_deg: float, resolution: tuple[int, int]) -> CameraIntrinsics:
        """Ideal pinhole with square pixels and the principal point at the centre."""
        width, height = resolution
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(fx=f, fy=f, cx=width / 2, cy=height / 2, width=width, height=height)


def project(
    points_board: np.ndarray, camera_pose: Pose, intrinsics: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Board-frame points (N, 3) -> pixel coordinates (N, 2) and depth along the axis (N,).

    Points with depth <= 0 are behind the camera; their pixels are meaningless.
    """
    camera_points = camera_pose.inverse().apply(np.atleast_2d(points_board))
    depth = camera_points[:, 2]
    safe = np.where(depth > 0, depth, np.nan)
    u = intrinsics.fx * camera_points[:, 0] / safe + intrinsics.cx
    v = intrinsics.fy * camera_points[:, 1] / safe + intrinsics.cy
    return np.column_stack([u, v]), depth
