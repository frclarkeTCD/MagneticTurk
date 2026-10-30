import chess
import numpy as np
import pytest

from magneticturk.config import load_config
from magneticturk.vision.move_inference import (
    Status,
    hypotheses_for,
    infer_move,
    occupancy_array,
    should_ask_player,
)

CLEAR = 4.0  # log-odds of a clear view of a square (p ~ 0.98)


@pytest.fixture
def config():
    return load_config().board.board_reading


def after(board: chess.Board, uci: str) -> chess.Board:
    board = board.copy()
    board.push_uci(uci)
    return board


def clear_view(board: chess.Board) -> np.ndarray:
    """Every square seen clearly."""
    return np.where(occupancy_array(board), CLEAR, -CLEAR)


def only(log_odds: np.ndarray, squares: list[chess.Square]) -> np.ndarray:
    """Keep evidence for the given squares only; the rest are unseen (0)."""
    partial = np.zeros(64)
    partial[squares] = log_odds[squares]
    return partial


def test_occupancy_array_matches_board():
    occupancy = occupancy_array(chess.Board())
    assert occupancy.sum() == 32
    assert occupancy[chess.E2] and not occupancy[chess.E4]


def test_hypotheses_group_promotions_and_include_no_move():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    hypotheses = hypotheses_for(board)
    assert hypotheses[0][0] == ()
    promotions = [moves for moves, _ in hypotheses if len(moves) > 1]
    assert len(promotions) == 1
    assert {m.promotion for m in promotions[0]} == {
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
    }


def test_every_opening_move_is_recognised(config):
    board = chess.Board()
    for move in board.legal_moves:
        result = infer_move(board, clear_view(after(board, move.uci())), config)
        assert result.status is Status.DECIDED
        assert result.best.moves == (move,)


@pytest.mark.parametrize(
    ("fen", "uci"),
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),  # castling: four squares change
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
        ("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1", "d5e6"),  # en passant: three squares change
        ("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1", "d4e5"),  # capture: only one square empties
    ],
)
def test_special_moves(config, fen, uci):
    board = chess.Board(fen)
    result = infer_move(board, clear_view(after(board, uci)), config)
    assert result.status is Status.DECIDED
    assert result.best.moves == (chess.Move.from_uci(uci),)


def test_promotion_is_decided_but_needs_piece_choice(config):
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    result = infer_move(board, clear_view(after(board, "a7a8q")), config)
    assert result.status is Status.DECIDED
    assert result.best.is_promotion
    assert len(result.best.moves) == 4


def test_two_possible_captures_by_one_piece_need_a_choice(config):
    # The knight can take on e5 or d4: either way f3 empties and nothing else changes.
    board = chess.Board("4k3/8/8/4p3/3p4/5N2/8/4K3 w - - 0 1")
    result = infer_move(board, clear_view(after(board, "f3e5")), config)
    assert result.status is Status.DECIDED
    assert result.best.needs_choice
    assert not result.best.is_promotion
    assert set(result.best.moves) == {chess.Move.from_uci("f3e5"), chess.Move.from_uci("f3d4")}


def test_unchanged_board(config):
    board = chess.Board()
    result = infer_move(board, clear_view(board), config)
    assert result.status is Status.NO_CHANGE
    assert result.best.moves == ()


def test_partial_view_asks_for_the_squares_that_matter(config):
    # Only e2 has been seen: it is empty, so the pawn moved, but to e3 or e4?
    board = chess.Board()
    evidence = only(clear_view(after(board, "e2e4")), [chess.E2])
    result = infer_move(board, evidence, config)
    assert result.status is Status.NEED_MORE_VIEWS
    assert set(result.informative_squares[:2]) == {chess.E3, chess.E4}
    assert {m for h in result.ranked[:2] for m in h.moves} == {
        chess.Move.from_uci("e2e3"),
        chess.Move.from_uci("e2e4"),
    }

    # Looking at the most informative square settles it.
    evidence[chess.E4] = CLEAR
    result = infer_move(board, evidence, config)
    assert result.status is Status.DECIDED
    assert result.best.moves == (chess.Move.from_uci("e2e4"),)


def test_no_evidence_at_all(config):
    result = infer_move(chess.Board(), np.zeros(64), config)
    assert result.status is Status.NEED_MORE_VIEWS
    assert result.log_ratio == 0


def test_impossible_change_does_not_fit(config):
    board = chess.Board()
    evidence = clear_view(board)
    evidence[chess.A1] = -CLEAR  # a rook vanished: no legal move does that
    result = infer_move(board, evidence, config)
    assert result.status is Status.NO_FIT
    assert result.contradicted_squares == (chess.A1,)
    assert should_ask_player(result, views_taken=1, config=config)


def test_weak_contradiction_is_not_a_misfit(config):
    board = chess.Board()
    evidence = clear_view(after(board, "e2e4"))
    evidence[chess.A1] = -1.0  # a doubtful reading, below the contradiction threshold
    result = infer_move(board, evidence, config)
    assert result.status is Status.DECIDED
    assert result.best.moves == (chess.Move.from_uci("e2e4"),)


def test_game_over_position(config):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate
    result = infer_move(board, clear_view(board), config)
    assert result.status is Status.NO_CHANGE
    assert len(result.ranked) == 1


def test_should_ask_player_after_max_views(config):
    result = infer_move(chess.Board(), np.zeros(64), config)
    assert not should_ask_player(result, views_taken=config.max_views - 1, config=config)
    assert should_ask_player(result, views_taken=config.max_views, config=config)


@pytest.mark.parametrize("bad", [np.zeros(63), np.full(64, np.nan)])
def test_rejects_bad_input(config, bad):
    with pytest.raises(ValueError):
        infer_move(chess.Board(), bad, config)


def test_noisy_views_over_random_games(config):
    """Simulated camera noise, looking again up to max_views times like the real robot.

    Each view reads every square with strength 2.5 plus noise (about 5% of readings
    point the wrong way, weakly) and misses 10% of squares. Views add in log-odds.
    When the robot decides, it must be right.
    """
    rng = np.random.default_rng(0)
    decided = wrong = asked = 0
    for _ in range(40):
        board = chess.Board()
        for _ in range(60):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            move = moves[rng.integers(len(moves))]
            truth = occupancy_array(after(board, move.uci()))
            evidence = np.zeros(64)
            for views in range(1, config.max_views + 1):
                view = np.where(truth, 2.5, -2.5) + rng.normal(0, 1.5, 64)
                view[rng.random(64) < 0.1] = 0.0
                evidence += view
                result = infer_move(board, evidence, config)
                if result.status is not Status.NEED_MORE_VIEWS:
                    break
            if result.status is Status.DECIDED:
                decided += 1
                wrong += move not in result.best.moves
            elif should_ask_player(result, views, config):
                asked += 1
            board.push(move)
    trials = decided + asked
    assert wrong == 0
    assert decided / trials > 0.9
