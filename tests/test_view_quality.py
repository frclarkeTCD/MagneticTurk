import chess
import numpy as np
import pytest

from magneticturk.config import load_config
from magneticturk.geometry.transforms import CameraIntrinsics, look_at, square_center
from magneticturk.vision.view_quality import view_quality

CENTRE = np.array([112.0, 112.0, 0.0])


@pytest.fixture
def cfg():
    return load_config()


def quality(cfg, eye, board, target=CENTRE):
    intrinsics = CameraIntrinsics.from_fov(cfg.camera.nominal_hfov_deg, cfg.camera.resolution)
    pose = look_at(np.asarray(eye, dtype=float), target)
    return view_quality(
        pose,
        intrinsics,
        board,
        cfg.board.square_size_mm,
        cfg.board.pieces,
        cfg.vision.view_quality,
    )


def test_overhead_view_of_empty_board_sees_everything_well(cfg):
    q = quality(cfg, [112, 112, 350], chess.Board(None))
    assert q.shape == (64,)
    assert np.all(q > 0.7)
    assert q[chess.D4] > q[chess.A1]  # centre squares are seen more squarely


def test_square_behind_a_tall_piece_is_occluded(cfg):
    # Camera low on the rank-1 side: the king on e2 stands between it and e3.
    board = chess.Board("8/8/8/8/8/8/4K3/8 w - - 0 1")
    eye = square_center(chess.E1, 28) + [0, -150, 120]
    q = quality(cfg, eye, board, target=square_center(chess.E4, 28))
    empty = quality(cfg, eye, chess.Board(None), target=square_center(chess.E4, 28))
    occluded = cfg.vision.view_quality.occluded_quality
    assert q[chess.E3] == pytest.approx(empty[chess.E3] * occluded)
    assert q[chess.E2] == pytest.approx(empty[chess.E2])  # a piece never hides itself
    assert q[chess.A3] == pytest.approx(empty[chess.A3])  # not in the king's shadow


def test_seen_from_the_other_side_it_is_clear(cfg):
    board = chess.Board("8/8/8/8/8/8/4K3/8 w - - 0 1")
    eye = square_center(chess.E8, 28) + [0, 150, 120]
    q = quality(cfg, eye, board, target=square_center(chess.E4, 28))
    empty = quality(cfg, eye, chess.Board(None), target=square_center(chess.E4, 28))
    assert q[chess.E3] == pytest.approx(empty[chess.E3])


def test_higher_views_hide_less(cfg):
    board = chess.Board()
    low = quality(cfg, [112, -150, 120], board)
    high = quality(cfg, [112, -150, 400], board)
    hidden = cfg.vision.view_quality.occluded_quality
    assert (low <= hidden + 1e-9).sum() > (high <= hidden + 1e-9).sum()


def test_flat_views_count_for_nothing(cfg):
    q = quality(cfg, [112, -2000, 50], chess.Board(None))
    assert np.all(q == 0)


def test_squares_out_of_frame_are_unseen(cfg):
    # Close over a1, aimed at a1: the far corner is out of the picture.
    corner = square_center(chess.A1, 28)
    q = quality(cfg, corner + [0, 0, 120], chess.Board(None), target=corner)
    assert q[chess.A1] > 0
    assert q[chess.H8] == 0


def test_distant_views_lose_quality(cfg):
    near = quality(cfg, [112, 112, 300], chess.Board(None))
    far = quality(cfg, [112, 112, 1500], chess.Board(None))
    assert far[chess.D4] < near[chess.D4]
