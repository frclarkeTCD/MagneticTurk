"""Game state: wraps python-chess. Position, move history, legality, game-over detection.

The rest of the program only changes the position through Game.push(), so the move
history always matches the real board. Vision uses occupied_squares() and the board
copy to predict what each legal move would look like; the supervisor uses
occupied_squares() to check the board after the robot moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.pgn


class IllegalMoveError(ValueError):
    """A move that is not legal in the current position, or played after the game ended."""


@dataclass(frozen=True)
class GameResult:
    winner: chess.Color | None  # None for a draw
    reason: str  # e.g. "checkmate", "stalemate", "time", "resignation"


class Game:
    def __init__(self, robot_color: chess.Color, fen: str = chess.STARTING_FEN) -> None:
        self.robot_color = robot_color
        self._board = chess.Board(fen)
        self._result: GameResult | None = None  # set by time, resignation or a draw claim

    @property
    def board(self) -> chess.Board:
        """A copy of the current position, safe to modify."""
        return self._board.copy()

    @property
    def turn(self) -> chess.Color:
        return self._board.turn

    @property
    def robot_to_move(self) -> bool:
        return self._board.turn == self.robot_color

    def legal_moves(self) -> list[chess.Move]:
        return list(self._board.legal_moves)

    def push(self, move: chess.Move) -> None:
        if self.result() is not None:
            raise IllegalMoveError(f"Game is over, cannot play {move.uci()}")
        if not self._board.is_legal(move):
            raise IllegalMoveError(f"{move.uci()} is not legal in {self._board.fen()}")
        self._board.push(move)

    def promotion_options(
        self, from_square: chess.Square, to_square: chess.Square
    ) -> list[chess.PieceType]:
        """Piece types the move from -> to can promote to. Empty if it is not a promotion.

        Vision only sees which squares changed, so a promotion is ambiguous until the
        player picks the piece on the companion display.
        """
        return [
            m.promotion
            for m in self._board.legal_moves
            if m.from_square == from_square and m.to_square == to_square and m.promotion
        ]

    def occupied_squares(self) -> frozenset[chess.Square]:
        return frozenset(chess.SquareSet(self._board.occupied))

    def flag(self, color: chess.Color) -> None:
        """color ran out of time. A draw if the opponent cannot possibly checkmate."""
        if self.result() is not None:
            return
        if self._board.has_insufficient_material(not color):
            self._result = GameResult(winner=None, reason="timeout vs insufficient material")
        else:
            self._result = GameResult(winner=not color, reason="time")

    def resign(self, color: chess.Color) -> None:
        if self.result() is None:
            self._result = GameResult(winner=not color, reason="resignation")

    def can_claim_draw(self) -> bool:
        return self.result() is None and self._board.can_claim_draw()

    def claim_draw(self) -> None:
        """Threefold repetition or fifty-move rule, claimed by the player on the display."""
        if not self.can_claim_draw():
            raise IllegalMoveError("No draw can be claimed in this position")
        reason = (
            "threefold repetition"
            if self._board.can_claim_threefold_repetition()
            else "fifty moves"
        )
        self._result = GameResult(winner=None, reason=reason)

    def result(self) -> GameResult | None:
        """None while the game is still going."""
        if self._result is not None:
            return self._result
        outcome = self._board.outcome(claim_draw=False)
        if outcome is None:
            return None
        reason = outcome.termination.name.lower().replace("_", " ")
        return GameResult(winner=outcome.winner, reason=reason)

    def san_history(self) -> list[str]:
        replay = self._board.root()
        sans = []
        for move in self._board.move_stack:
            sans.append(replay.san(move))
            replay.push(move)
        return sans

    def pgn(self) -> str:
        game = chess.pgn.Game.from_board(self._board)
        if self._result is not None:
            winner = self._result.winner
            game.headers["Result"] = (
                "1/2-1/2" if winner is None else "1-0" if winner == chess.WHITE else "0-1"
            )
        return str(game)
