import dataclasses

import chess
import pytest

from magneticturk.config import load_config
from magneticturk.game.engine import (
    Engine,
    EngineError,
    RandomEngine,
    StockfishEngine,
    find_stockfish,
)


@pytest.fixture
def engine_config():
    # Short think time keeps the tests quick.
    return dataclasses.replace(load_config().game.engine, think_time_s=0.05)


def have_stockfish() -> bool:
    try:
        find_stockfish(load_config().game.engine)
    except EngineError:
        return False
    return True


needs_stockfish = pytest.mark.skipif(not have_stockfish(), reason="Stockfish not installed")


def test_random_engine_plays_legal_moves():
    engine = RandomEngine(seed=1)
    assert isinstance(engine, Engine)
    board = chess.Board()
    for _ in range(40):
        if board.is_game_over():
            break
        move = engine.choose_move(board)
        assert board.is_legal(move)
        board.push(move)


def test_random_engine_is_repeatable_with_a_seed():
    board = chess.Board()
    assert RandomEngine(seed=7).choose_move(board) == RandomEngine(seed=7).choose_move(board)


def test_random_engine_game_over():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    with pytest.raises(ValueError):
        RandomEngine().choose_move(board)


def test_missing_stockfish(engine_config):
    config = dataclasses.replace(engine_config, path="no-such-stockfish", extra_search_dirs=())
    with pytest.raises(EngineError, match="not found"):
        find_stockfish(config)


@needs_stockfish
def test_stockfish_plays_a_legal_move(engine_config):
    board = chess.Board()
    with StockfishEngine(engine_config) as engine:
        move = engine.choose_move(board)
    assert board.is_legal(move)
    assert board.fen() == chess.Board().fen()


@needs_stockfish
def test_stockfish_finds_mate_in_one_at_full_strength(engine_config):
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    with StockfishEngine(engine_config, skill_level=20) as engine:
        assert engine.choose_move(board) == chess.Move.from_uci("a1a8")


@needs_stockfish
def test_stockfish_skill_level_range(engine_config):
    with StockfishEngine(engine_config) as engine:
        assert engine.skill_range == (0, 20)
        engine.set_skill_level(0)
        assert engine.skill_level == 0
        with pytest.raises(ValueError):
            engine.set_skill_level(21)


@needs_stockfish
def test_stockfish_bad_skill_level_at_start(engine_config):
    with pytest.raises(ValueError):
        StockfishEngine(engine_config, skill_level=-1)


@needs_stockfish
def test_stockfish_game_over(engine_config):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    with StockfishEngine(engine_config) as engine, pytest.raises(ValueError):
        engine.choose_move(board)
