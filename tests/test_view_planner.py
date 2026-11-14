import dataclasses

import chess
import numpy as np
import pytest

from magneticturk.config import load_config
from magneticturk.geometry.transforms import CameraIntrinsics
from magneticturk.vision.fusion import BoardBelief
from magneticturk.vision.move_inference import Status, infer_move, occupancy_array
from magneticturk.vision.simulated_views import ViewSimulator
from magneticturk.vision.view_planner import candidate_poses, first_view, plan_next_view
from magneticturk.vision.view_quality import view_quality


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def intrinsics(cfg):
    return CameraIntrinsics.from_fov(cfg.camera.nominal_hfov_deg, cfg.camera.resolution)


def plan(cfg, intrinsics, board, inference, previous, **kwargs):
    return plan_next_view(
        board,
        inference,
        previous,
        intrinsics,
        cfg.board.square_size_mm,
        cfg.board.pieces,
        cfg.vision.view_quality,
        cfg.vision.view_planner,
        **kwargs,
    )


def e2_moved_but_where(cfg):
    """Only e2 has been seen (empty): the pawn went to e3 or e4."""
    board = chess.Board()
    after = board.copy()
    after.push_uci("e2e4")
    evidence = np.zeros(64)
    evidence[chess.E2] = -4.0
    return board, infer_move(board, evidence, cfg.board.board_reading)


def test_first_view_comes_from_config(cfg):
    pose = first_view(cfg.vision.view_planner)
    assert np.allclose(pose.position, cfg.vision.view_planner.first_view_eye_mm)


def test_candidates_cover_heights_grid_and_targets(cfg):
    planner = cfg.vision.view_planner
    per_side = len(
        np.arange(
            -planner.candidate_margin_mm,
            224 + planner.candidate_margin_mm + 1,
            planner.candidate_spacing_mm,
        )
    )
    with_focus = candidate_poses(planner, 28, np.ones(64))
    assert len(with_focus) == len(planner.candidate_heights_mm) * per_side**2 * 2
    assert len(candidate_poses(planner, 28, np.zeros(64))) == len(with_focus) // 2


def test_looks_where_the_candidates_differ(cfg, intrinsics):
    board, inference = e2_moved_but_where(cfg)
    assert inference.status is Status.NEED_MORE_VIEWS
    start = first_view(cfg.vision.view_planner)
    pose = plan(cfg, intrinsics, board, inference, [start])
    q = view_quality(pose, intrinsics, board, 28, cfg.board.pieces, cfg.vision.view_quality)
    assert q[chess.E3] > 0.5 and q[chess.E4] > 0.5


def test_does_not_revisit_earlier_views(cfg, intrinsics):
    board, inference = e2_moved_but_where(cfg)
    first = plan(cfg, intrinsics, board, inference, [first_view(cfg.vision.view_planner)])
    second = plan(cfg, intrinsics, board, inference, [first_view(cfg.vision.view_planner), first])
    separation = np.linalg.norm(second.position - first.position)
    assert separation >= cfg.vision.view_planner.min_view_separation_mm


def test_only_reachable_poses(cfg, intrinsics):
    board, inference = e2_moved_but_where(cfg)
    previous = [first_view(cfg.vision.view_planner)]
    pose = plan(cfg, intrinsics, board, inference, previous, reachable=lambda p: p.position[0] < 0)
    assert pose.position[0] < 0
    assert plan(cfg, intrinsics, board, inference, previous, reachable=lambda p: False) is None


def test_nothing_to_plan_once_decided(cfg, intrinsics):
    board = chess.Board()
    after = board.copy()
    after.push_uci("e2e4")
    evidence = np.where(occupancy_array(after), 4.0, -4.0)
    inference = infer_move(board, evidence, cfg.board.board_reading)
    assert inference.status is Status.DECIDED
    assert plan(cfg, intrinsics, board, inference, []) is None


def read_board(cfg, intrinsics, sim, board, truth, choose_next):
    """The robot's board-reading loop: look, fuse, infer, plan, up to max_views."""
    belief = BoardBelief(cfg.vision.fusion)
    poses = [first_view(cfg.vision.view_planner)]
    while True:
        belief.add(sim.view(poses[-1], truth, board))
        result = infer_move(board, belief.log_odds, cfg.board.board_reading)
        if result.status is not Status.NEED_MORE_VIEWS:
            return result
        if belief.views >= cfg.board.board_reading.max_views:
            return result
        pose = choose_next(board, result, poses)
        if pose is None:
            return result
        poses.append(pose)


def play_simulated_games(cfg, intrinsics, choose_next, games, moves, seed=0):
    rng = np.random.default_rng(seed)
    sim = ViewSimulator(
        intrinsics,
        cfg.board.square_size_mm,
        cfg.board.pieces,
        cfg.vision.view_quality,
        cfg.vision.simulation,
        rng,
    )
    decided = wrong = total = 0
    for _ in range(games):
        board = chess.Board()
        for _ in range(moves):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            move = legal[rng.integers(len(legal))]
            truth = board.copy()
            truth.push(move)
            result = read_board(cfg, intrinsics, sim, board, truth, choose_next)
            total += 1
            if result.status is Status.DECIDED:
                decided += 1
                wrong += move not in result.best.moves
            board.push(move)
    return decided / total, wrong


def test_active_reading_beats_looking_again_from_the_same_place(cfg, intrinsics):
    """End to end on simulated views, starting from an angled view on the arm's side."""
    planner = dataclasses.replace(cfg.vision.view_planner, first_view_eye_mm=(112.0, -100.0, 250.0))
    cfg = dataclasses.replace(cfg, vision=dataclasses.replace(cfg.vision, view_planner=planner))

    def planned(board, result, poses):
        return plan(cfg, intrinsics, board, result, poses)

    def repeat(board, result, poses):
        return poses[0]

    planned_rate, planned_wrong = play_simulated_games(cfg, intrinsics, planned, 3, 25)
    repeat_rate, _ = play_simulated_games(cfg, intrinsics, repeat, 3, 25)
    assert planned_wrong == 0
    assert planned_rate > 0.85
    assert planned_rate > repeat_rate + 0.3
