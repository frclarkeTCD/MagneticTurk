"""Identify the player's move from occupancy beliefs. No piece recognition needed.

Given the previous position (python-chess Board) and per-square beliefs:
enumerate legal moves, compute each move's predicted occupancy change, score each
hypothesis's likelihood, and return the winner once it beats the runner-up by the
configured ratio. Otherwise report which squares would best tell the leaders apart.

Beliefs are log-odds of occupancy, one per square (index = python-chess square, a1 = 0),
as produced by vision/fusion.py. 0 means "no evidence" (p = 0.5): such squares favour
no hypothesis, so a partial view never biases the result.

Hypotheses:
- every legal move, grouped by the occupancy it leaves behind. Moves that leave exactly
  the same squares occupied cannot be told apart by occupancy, so they share one
  hypothesis and the player picks between them on the display. This happens for
  promotions (which piece?) and when one piece could capture either of two pieces
  (the same squares end up occupied either way);
- "no move": the board is unchanged (e.g. the clock was tapped before moving).

Each hypothesis predicts occupied (1) or empty (0) for every square. With evidence L_s,
    log P(evidence | H) = sum_s log sigmoid(+L_s) if H says occupied, else log sigmoid(-L_s)
and hypotheses start equally likely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import chess
import numpy as np

from magneticturk.config import BoardReadingConfig

_EPS = 1e-12  # keeps log2 finite; numerical guard, not a tuning value


class Status(Enum):
    DECIDED = auto()  # one hypothesis clearly fits (it may still hold several moves)
    NO_CHANGE = auto()  # the board clearly has not changed
    NEED_MORE_VIEWS = auto()  # look again at informative_squares
    NO_FIT = auto()  # strong evidence contradicts even the best move: ask the player


@dataclass(frozen=True, eq=False)
class Hypothesis:
    moves: tuple[chess.Move, ...]  # () = no move; several = indistinguishable by occupancy
    occupancy: np.ndarray  # predicted occupancy after the move, bool, shape (64,)
    log_likelihood: float

    @property
    def needs_choice(self) -> bool:
        """The player must say which of these moves they made (on the display)."""
        return len(self.moves) > 1

    @property
    def is_promotion(self) -> bool:
        """Same from and to squares, differing only in the promotion piece."""
        return self.needs_choice and len({(m.from_square, m.to_square) for m in self.moves}) == 1


@dataclass(frozen=True, eq=False)
class InferenceResult:
    status: Status
    ranked: tuple[Hypothesis, ...]  # best first
    log_ratio: float  # log(likelihood of best / runner-up); inf if there is only one
    informative_squares: tuple[chess.Square, ...]  # most useful to look at first
    contradicted_squares: tuple[chess.Square, ...]  # evidence strongly disagrees with best

    @property
    def best(self) -> Hypothesis:
        return self.ranked[0]


def occupancy_array(board: chess.Board) -> np.ndarray:
    return _bitboard_to_array(board.occupied)


def _bitboard_to_array(bitboard: int) -> np.ndarray:
    return np.array([(bitboard >> s) & 1 for s in chess.SQUARES], dtype=bool)


def hypotheses_for(board: chess.Board) -> list[tuple[tuple[chess.Move, ...], np.ndarray]]:
    """(moves, predicted occupancy) for "no move" and each group of legal moves that
    leave the same squares occupied."""
    groups: dict[int, list[chess.Move]] = {}  # occupancy bitboard -> moves
    for move in board.legal_moves:
        board.push(move)
        groups.setdefault(board.occupied, []).append(move)
        board.pop()
    result = [((), occupancy_array(board))]
    result += [(tuple(moves), _bitboard_to_array(bb)) for bb, moves in groups.items()]
    return result


def infer_move(
    board: chess.Board, log_odds: np.ndarray, config: BoardReadingConfig
) -> InferenceResult:
    """Rank every hypothesis for what changed since `board` and decide if one clearly wins."""
    log_odds = np.asarray(log_odds, dtype=float)
    if log_odds.shape != (64,):
        raise ValueError(f"Expected 64 log-odds, got shape {log_odds.shape}")
    if not np.all(np.isfinite(log_odds)):
        raise ValueError("Log-odds must be finite")

    candidates = hypotheses_for(board.copy(stack=False))
    predicted = np.array([occupancy for _, occupancy in candidates])  # (H, 64)
    log_p_occupied = -np.logaddexp(0.0, -log_odds)  # log sigmoid(L)
    log_p_empty = -np.logaddexp(0.0, log_odds)  # log sigmoid(-L)
    log_likelihoods = np.where(predicted, log_p_occupied, log_p_empty).sum(axis=1)

    order = np.argsort(-log_likelihoods, kind="stable")
    ranked = tuple(
        Hypothesis(candidates[i][0], predicted[i], float(log_likelihoods[i])) for i in order
    )
    log_ratio = ranked[0].log_likelihood - ranked[1].log_likelihood if len(ranked) > 1 else math.inf

    best = ranked[0]
    strong = np.abs(log_odds) >= config.contradiction_log_odds
    contradicted = np.flatnonzero(strong & ((log_odds > 0) != best.occupancy))
    informative = _informative_squares(
        predicted[order], log_likelihoods[order], config.min_square_entropy_bits
    )

    if contradicted.size:
        status = Status.NO_FIT
    elif log_ratio >= math.log(config.decision_ratio):
        status = Status.NO_CHANGE if not best.moves else Status.DECIDED
    else:
        status = Status.NEED_MORE_VIEWS
    return InferenceResult(
        status=status,
        ranked=ranked,
        log_ratio=float(log_ratio),
        informative_squares=tuple(int(s) for s in informative),
        contradicted_squares=tuple(int(s) for s in contradicted),
    )


def should_ask_player(
    result: InferenceResult, views_taken: int, config: BoardReadingConfig
) -> bool:
    """True when the robot should stop looking and ask the player to check the board."""
    if result.status is Status.NO_FIT:
        return True
    return result.status is Status.NEED_MORE_VIEWS and views_taken >= config.max_views


def _informative_squares(
    predicted: np.ndarray, log_likelihoods: np.ndarray, min_entropy_bits: float
) -> np.ndarray:
    """Squares whose occupancy the likely hypotheses disagree on, most uncertain first.

    Weights each hypothesis by its posterior probability, then scores each square by the
    entropy of its predicted occupancy: the information a clear look at it would give.
    """
    weights = np.exp(log_likelihoods - log_likelihoods.max())
    weights /= weights.sum()
    p = np.clip(weights @ predicted, _EPS, 1 - _EPS)  # P(occupied) under the mixture
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    squares = np.flatnonzero(entropy >= min_entropy_bits)
    return squares[np.argsort(-entropy[squares], kind="stable")]
