"""Combine occupancy evidence across views.

Keeps a per-square belief in log-odds; each new view adds weighted evidence.
Uncertain squares are those whose belief is near 0.5.

A view's evidence for square s is  quality_s * logit(p_s),  where p_s is the view's
P(occupied) and quality_s in [0, 1] says how far to trust it (0 = not seen). Each view's
logit is capped at max_view_log_odds, so no single view can outvote several others.
The belief starts at 0 (no evidence) for every board reading; move inference supplies
the prior through the previous position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magneticturk.config import FusionConfig


@dataclass(frozen=True, eq=False)
class ViewEvidence:
    """What one view says about the board. The output of vision/occupancy.py."""

    probability: np.ndarray  # (64,) P(occupied) per square; ignored where quality is 0
    quality: np.ndarray  # (64,) trust in [0, 1]

    def __post_init__(self) -> None:
        for name in ("probability", "quality"):
            value = getattr(self, name)
            if value.shape != (64,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be 64 finite numbers")
            if np.any(value < 0) or np.any(value > 1):
                raise ValueError(f"{name} must be in [0, 1]")


class BoardBelief:
    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self.log_odds = np.zeros(64)
        self.views = 0

    def add(self, view: ViewEvidence) -> None:
        cap = self.config.max_view_log_odds
        with np.errstate(divide="ignore"):
            logit = np.log(view.probability) - np.log1p(-view.probability)
        self.log_odds += view.quality * np.clip(logit, -cap, cap)
        self.views += 1

    @property
    def probability(self) -> np.ndarray:
        return 1 / (1 + np.exp(-self.log_odds))
