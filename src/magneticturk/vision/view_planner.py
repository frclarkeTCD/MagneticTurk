"""Choose the next-best camera viewpoint.

Given current beliefs and the remaining candidate moves, score a set of reachable
candidate camera poses by how well they would separate the candidates. Return the best pose.

Score of a pose = sum over squares of (bits a clear look would give) x (predicted view
quality), minus a small cost per mm of camera travel. The bits come from move inference
(square_entropy_bits), so the planner looks where the leading moves disagree, and view
quality makes it prefer close, steep views that are not blocked by tall pieces.

Candidates: a grid of camera positions over and around the board, at each configured
height, each aimed at the board centre or at the squares that matter most. Positions
close to an earlier view this turn are skipped: repeating a view repeats its glare and
occlusion. Whether the arm can reach a pose is decided by the `reachable` callback,
which robot/kinematics.py (stage 8) will provide; until then every pose is allowed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import chess
import numpy as np

from magneticturk.config import PieceConfig, ViewPlannerConfig, ViewQualityConfig
from magneticturk.geometry.transforms import CameraIntrinsics, Pose, look_at, square_centers
from magneticturk.vision.move_inference import InferenceResult, Status
from magneticturk.vision.view_quality import view_quality


def first_view(config: ViewPlannerConfig) -> Pose:
    return look_at(np.array(config.first_view_eye_mm), np.array(config.first_view_target_mm))


def candidate_poses(
    config: ViewPlannerConfig, square_size_mm: float, square_entropy_bits: np.ndarray
) -> list[Pose]:
    board_size = 8 * square_size_mm
    centre = np.array([board_size / 2, board_size / 2, 0.0])
    targets = [centre]
    if square_entropy_bits.sum() > 0:
        weights = square_entropy_bits / square_entropy_bits.sum()
        targets.append(weights @ square_centers(square_size_mm))

    m = config.candidate_margin_mm
    step = config.candidate_spacing_mm
    grid = np.arange(-m, board_size + m + step / 2, step)  # includes the far edge
    poses = []
    for height in config.candidate_heights_mm:
        for x in grid:
            for y in grid:
                eye = np.array([x, y, height])
                poses.extend(look_at(eye, target) for target in targets)
    return poses


def plan_next_view(
    board: chess.Board,
    inference: InferenceResult,
    previous_views: Sequence[Pose],
    intrinsics: CameraIntrinsics,
    square_size_mm: float,
    pieces: PieceConfig,
    quality: ViewQualityConfig,
    config: ViewPlannerConfig,
    reachable: Callable[[Pose], bool] = lambda pose: True,
) -> Pose | None:
    """The most useful next camera pose, or None if the move is already settled or no
    candidate would tell us anything.

    `board` is the position before the player's move: its pieces are the likely occluders.
    """
    if inference.status is not Status.NEED_MORE_VIEWS:
        return None
    entropy = inference.square_entropy_bits
    seen = np.array([p.position for p in previous_views]).reshape(-1, 3)
    current = seen[-1] if len(seen) else None

    best_pose, best_score = None, 0.0
    for pose in candidate_poses(config, square_size_mm, entropy):
        if len(seen) and np.min(np.linalg.norm(seen - pose.position, axis=1)) < (
            config.min_view_separation_mm
        ):
            continue
        if not reachable(pose):
            continue
        q = view_quality(pose, intrinsics, board, square_size_mm, pieces, quality)
        score = float(entropy @ q)
        if current is not None:
            score -= config.travel_cost_per_mm * float(np.linalg.norm(pose.position - current))
        if score > best_score:
            best_pose, best_score = pose, score
    return best_pose
