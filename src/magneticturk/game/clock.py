"""Chess clock: base time + increment per player, or untimed. Flag-fall detection.

Fischer increment: the increment is added to a player's time when they press the clock
at the end of their move, unless their flag has already fallen. The clock can be paused
(e.g. while the game is in Paused or Fault mode) without charging either side.

Time comes from an injectable `now` function so tests can step time by hand.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import chess

from magneticturk.config import ClockConfig


class ClockError(RuntimeError):
    """The clock was used out of order, e.g. pressed before it was started."""


class ChessClock:
    def __init__(self, config: ClockConfig, now: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._now = now
        base = config.base_s
        self._remaining: dict[chess.Color, float | None] = {chess.WHITE: base, chess.BLACK: base}
        self.turn: chess.Color = chess.WHITE
        self._started = False
        self._running_since: float | None = None  # None while stopped or paused

    @property
    def timed(self) -> bool:
        return self.config.base_s is not None

    @property
    def running(self) -> bool:
        return self._running_since is not None

    def start(self, turn: chess.Color = chess.WHITE) -> None:
        if self._started:
            raise ClockError("Clock already started")
        self._started = True
        self.turn = turn
        self._running_since = self._now()

    def press(self) -> None:
        """The player on turn has finished their move: stop their time, start the other's."""
        if not self.running:
            raise ClockError("Clock is not running")
        self._charge()
        remaining = self._remaining[self.turn]
        if remaining is not None and remaining > 0:
            self._remaining[self.turn] = remaining + self.config.increment_s
        self.turn = not self.turn
        self._running_since = self._now()

    def pause(self) -> None:
        if self.running:
            self._charge()
            self._running_since = None

    def resume(self) -> None:
        if not self._started:
            raise ClockError("Clock not started")
        if not self.running:
            self._running_since = self._now()

    def remaining(self, color: chess.Color) -> float | None:
        """Seconds left for color, never negative. None for an untimed game."""
        left = self._remaining[color]
        if left is None:
            return None
        if self.running and color == self.turn:
            left -= self._now() - self._running_since
        return max(left, 0.0)

    def flagged(self) -> chess.Color | None:
        """The colour whose time has run out, if any."""
        for color in (chess.WHITE, chess.BLACK):
            if self.remaining(color) == 0.0:
                return color
        return None

    def _charge(self) -> None:
        left = self._remaining[self.turn]
        if left is not None:
            self._remaining[self.turn] = max(left - (self._now() - self._running_since), 0.0)
