import chess
import pytest

from magneticturk.game.game import Game, IllegalMoveError


def play(game: Game, *ucis: str) -> None:
    for uci in ucis:
        game.push(chess.Move.from_uci(uci))


def test_new_game():
    game = Game(robot_color=chess.BLACK)
    assert game.turn == chess.WHITE
    assert not game.robot_to_move
    assert len(game.legal_moves()) == 20
    assert game.occupied_squares() == frozenset(range(16)) | frozenset(range(48, 64))
    assert game.result() is None


def test_push_and_history():
    game = Game(robot_color=chess.BLACK)
    play(game, "e2e4", "e7e5", "g1f3")
    assert game.robot_to_move
    assert game.san_history() == ["e4", "e5", "Nf3"]


def test_illegal_move_is_rejected_and_board_unchanged():
    game = Game(robot_color=chess.BLACK)
    with pytest.raises(IllegalMoveError):
        play(game, "e2e5")
    assert game.san_history() == []


def test_board_is_a_copy():
    game = Game(robot_color=chess.BLACK)
    game.board.push_uci("e2e4")
    assert game.san_history() == []


def test_castling_moves_two_pieces():
    game = Game(robot_color=chess.BLACK, fen="4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    play(game, "e1g1")
    occupied = game.occupied_squares()
    assert {chess.G1, chess.F1} <= occupied
    assert not {chess.E1, chess.H1} & occupied


def test_en_passant_empties_a_third_square():
    game = Game(robot_color=chess.BLACK, fen="4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1")
    play(game, "d5e6")
    occupied = game.occupied_squares()
    assert chess.E6 in occupied
    assert chess.E5 not in occupied
    assert chess.D5 not in occupied


def test_promotion_options():
    game = Game(robot_color=chess.BLACK, fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert set(game.promotion_options(chess.A7, chess.A8)) == {
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
    }
    assert game.promotion_options(chess.E1, chess.E2) == []


def test_checkmate():
    game = Game(robot_color=chess.BLACK)
    play(game, "f2f3", "e7e5", "g2g4", "d8h4")
    result = game.result()
    assert result.winner == chess.BLACK
    assert result.reason == "checkmate"
    with pytest.raises(IllegalMoveError, match="over"):
        play(game, "e2e4")


def test_stalemate():
    game = Game(robot_color=chess.BLACK, fen="7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert game.result().winner is None
    assert game.result().reason == "stalemate"


def test_flag_loses():
    game = Game(robot_color=chess.BLACK)
    game.flag(chess.WHITE)
    assert game.result().winner == chess.BLACK
    assert game.result().reason == "time"


def test_flag_against_bare_king_is_a_draw():
    game = Game(robot_color=chess.BLACK, fen="4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    game.flag(chess.WHITE)
    assert game.result().winner is None


def test_resign():
    game = Game(robot_color=chess.BLACK)
    game.resign(chess.WHITE)
    assert game.result().winner == chess.BLACK
    assert "1-0" not in game.pgn()
    assert "0-1" in game.pgn()


def test_threefold_claim():
    game = Game(robot_color=chess.BLACK)
    assert not game.can_claim_draw()
    with pytest.raises(IllegalMoveError):
        game.claim_draw()
    play(game, "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8")
    assert game.can_claim_draw()
    game.claim_draw()
    assert game.result().reason == "threefold repetition"


def test_pgn_has_moves():
    game = Game(robot_color=chess.BLACK)
    play(game, "e2e4", "e7e5")
    assert "1. e4 e5" in game.pgn()
