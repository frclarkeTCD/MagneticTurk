"""Stockfish wrapper via UCI (python-chess). Skill level 0-20 and think-time settings.

Engine (interface): choose_move(board) -> move.
StockfishEngine: runs Stockfish as a separate program (it is GPLv3, so it is never linked in).
RandomEngine: plays a random legal move, for --fake runs and tests without Stockfish.
"""

from __future__ import annotations

import os
import random
import shutil
from abc import ABC, abstractmethod
from typing import Self

import chess
import chess.engine

from magneticturk.config import EngineConfig


class EngineError(RuntimeError):
    """Stockfish could not be found, started or configured."""


class Engine(ABC):
    @abstractmethod
    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return a legal move for the side to move. The board is not modified."""

    def close(self) -> None:
        """Release the engine. Optional for engines with nothing to release."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class StockfishEngine(Engine):
    SKILL_OPTION = "Skill Level"

    def __init__(self, config: EngineConfig, skill_level: int | None = None) -> None:
        self.config = config
        path = find_stockfish(config)
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(path)
        except (OSError, chess.engine.EngineError) as e:
            raise EngineError(f"Could not start Stockfish at {path}: {e}") from e
        try:
            self._engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
            self.set_skill_level(config.skill_level if skill_level is None else skill_level)
        except Exception:
            self._engine.quit()
            raise

    @property
    def skill_range(self) -> tuple[int, int]:
        option = self._engine.options[self.SKILL_OPTION]
        return int(option.min), int(option.max)

    def set_skill_level(self, level: int) -> None:
        lo, hi = self.skill_range
        if not lo <= level <= hi:
            raise ValueError(f"Skill level must be between {lo} and {hi}, got {level}")
        self._engine.configure({self.SKILL_OPTION: level})
        self.skill_level = level

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over():
            raise ValueError("No move to choose: the game is over")
        result = self._engine.play(board, chess.engine.Limit(time=self.config.think_time_s))
        if result.move is None:
            raise EngineError("Stockfish returned no move")
        return result.move

    def close(self) -> None:
        self._engine.quit()


class RandomEngine(Engine):
    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("No move to choose: the game is over")
        return self._random.choice(moves)


def find_stockfish(config: EngineConfig) -> str:
    """Full path to Stockfish, searching PATH and then config.extra_search_dirs."""
    search = os.pathsep.join([os.environ.get("PATH", ""), *config.extra_search_dirs])
    path = shutil.which(config.path, path=search)
    if path is None:
        raise EngineError(
            f"Stockfish not found as {config.path!r}. Install it "
            "(Pi: sudo apt install stockfish, Mac: brew install stockfish) or set engine.path."
        )
    return path
