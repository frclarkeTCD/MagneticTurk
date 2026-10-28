import chess
import pytest

from magneticturk.config import ClockConfig
from magneticturk.game.clock import ChessClock, ClockError


class FakeTime:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def now():
    return FakeTime()


def clock(now, base_s=60.0, increment_s=2.0) -> ChessClock:
    return ChessClock(ClockConfig(base_s=base_s, increment_s=increment_s), now=now)


def test_only_the_side_to_move_loses_time(now):
    c = clock(now)
    c.start()
    now.advance(10)
    assert c.remaining(chess.WHITE) == 50
    assert c.remaining(chess.BLACK) == 60


def test_press_switches_turn_and_adds_increment(now):
    c = clock(now)
    c.start()
    now.advance(10)
    c.press()
    assert c.turn == chess.BLACK
    assert c.remaining(chess.WHITE) == 52
    now.advance(5)
    assert c.remaining(chess.BLACK) == 55
    assert c.remaining(chess.WHITE) == 52


def test_start_with_black(now):
    c = clock(now)
    c.start(chess.BLACK)
    now.advance(1)
    assert c.remaining(chess.BLACK) == 59


def test_pause_stops_time(now):
    c = clock(now)
    c.start()
    now.advance(10)
    c.pause()
    assert not c.running
    now.advance(100)
    assert c.remaining(chess.WHITE) == 50
    c.resume()
    now.advance(5)
    assert c.remaining(chess.WHITE) == 45


def test_pause_and_resume_are_idempotent(now):
    c = clock(now)
    c.start()
    c.pause()
    c.pause()
    c.resume()
    c.resume()
    now.advance(1)
    assert c.remaining(chess.WHITE) == 59


def test_flag_fall(now):
    c = clock(now)
    c.start()
    now.advance(59)
    assert c.flagged() is None
    now.advance(2)
    assert c.flagged() == chess.WHITE
    assert c.remaining(chess.WHITE) == 0


def test_no_increment_after_flag_fall(now):
    c = clock(now)
    c.start()
    now.advance(70)
    c.press()
    assert c.remaining(chess.WHITE) == 0
    assert c.flagged() == chess.WHITE


def test_untimed(now):
    c = clock(now, base_s=None)
    assert not c.timed
    c.start()
    now.advance(10_000)
    assert c.remaining(chess.WHITE) is None
    assert c.flagged() is None
    c.press()
    assert c.turn == chess.BLACK


def test_misuse(now):
    c = clock(now)
    with pytest.raises(ClockError):
        c.press()
    with pytest.raises(ClockError):
        c.resume()
    c.start()
    with pytest.raises(ClockError):
        c.start()
    c.pause()
    with pytest.raises(ClockError):
        c.press()
