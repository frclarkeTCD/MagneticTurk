"""Simulated camera views at the evidence level, for tests and --fake runs.

Stands in for "capture an image, then run vision/occupancy.py" until stage 4 exists.
Each simulated view:
- reads every square the camera sees, with a signal that weakens for steep or distant
  views, plus Gaussian noise. Most of the noise belongs to the camera position (glare,
  shadows) and comes back if the camera returns there, as it would on the real board;
- gets occlusion wrong the way a real camera does: a square hidden behind a taller
  piece tends to look occupied, whatever is really there;
- reports the quality the robot would predict, from the position it *believes* is on
  the board (it cannot know the true one: that is what it is trying to find out).
"""

from __future__ import annotations

import chess
import numpy as np

from magneticturk.config import PieceConfig, SimulationConfig, ViewQualityConfig
from magneticturk.geometry.transforms import CameraIntrinsics, Pose, square_centers
from magneticturk.vision.fusion import ViewEvidence
from magneticturk.vision.view_quality import occluded_squares, view_quality


class ViewSimulator:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        square_size_mm: float,
        pieces: PieceConfig,
        quality: ViewQualityConfig,
        settings: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.intrinsics = intrinsics
        self.square_size_mm = square_size_mm
        self.pieces = pieces
        self.quality = quality
        self.rng = rng
        self.settings = settings
        self._pose_noise: dict[tuple[float, ...], np.ndarray] = {}

    def view(self, pose: Pose, true_board: chess.Board, known_board: chess.Board) -> ViewEvidence:
        s = self.settings
        truth = np.array([true_board.piece_at(sq) is not None for sq in chess.SQUARES])
        geometric = view_quality(
            pose, self.intrinsics, _empty_board(), self.square_size_mm, self.pieces, self.quality
        )
        look = square_centers(self.square_size_mm)
        look[:, 2] = self.quality.look_height_mm
        hidden = occluded_squares(look, pose.position, true_board, self.square_size_mm, self.pieces)

        signal = np.where(truth, s.signal_log_odds, -s.signal_log_odds) * geometric
        signal = np.where(hidden, s.occluded_bias_log_odds, signal)
        c = s.repeat_correlation
        fresh = self.rng.normal(0, 1, 64)
        noise = np.sqrt(c) * self._noise_at(pose) + np.sqrt(1 - c) * fresh
        reading = signal + s.noise_log_odds * noise

        reported_quality = view_quality(
            pose, self.intrinsics, known_board, self.square_size_mm, self.pieces, self.quality
        )
        seen = geometric > 0
        probability = np.where(seen, 1 / (1 + np.exp(-reading)), 0.5)
        return ViewEvidence(probability=probability, quality=np.where(seen, reported_quality, 0))

    def _noise_at(self, pose: Pose) -> np.ndarray:
        """Unit noise that stays the same for every view from (about) the same pose."""
        key = tuple(np.round(np.concatenate([pose.position, pose.rotation[:, 2] * 100])))
        if key not in self._pose_noise:
            self._pose_noise[key] = self.rng.normal(0, 1, 64)
        return self._pose_noise[key]


def _empty_board() -> chess.Board:
    return chess.Board(None)
