"""Load and validate config/*.yaml into typed, read-only dataclasses.

Every tunable number in the project comes from here. Loading fails loudly with a
ConfigError that names the file and the problem, so a typo never reaches the servos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
SQUARES_PER_SIDE = 8


class ConfigError(ValueError):
    """A config file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class GripperConfig:
    open_deg: float
    closed_deg: float
    grip_height_above_board_mm: float


@dataclass(frozen=True)
class GraspAlignConfig:
    tolerance_mm: float
    max_corrections: int


@dataclass(frozen=True)
class RobotConfig:
    servo_channels: dict[str, int]
    i2c_address: int
    pwm_frequency_hz: float
    pulse_min_us: float
    pulse_max_us: float
    servo_travel_deg: float
    joint_limits_deg: dict[str, tuple[float, float]]
    home_pose_deg: dict[str, float]
    max_speed_deg_s: float
    link_lengths_mm: dict[str, float]
    gripper: GripperConfig
    grasp_align: GraspAlignConfig

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(self.servo_channels)


@dataclass(frozen=True)
class CameraConfig:
    resolution: tuple[int, int]
    alignment_resolution: tuple[int, int]
    autofocus: bool


@dataclass(frozen=True)
class MarkerConfig:
    dictionary: str
    size_mm: float
    positions: dict[int, tuple[float, float]]


@dataclass(frozen=True)
class CaptureTrayConfig:
    position_mm: tuple[float, float]
    slots: int


@dataclass(frozen=True)
class BoardReadingConfig:
    max_views: int
    decision_ratio: float
    contradiction_log_odds: float
    min_square_entropy_bits: float


@dataclass(frozen=True)
class BoardConfig:
    square_size_mm: float
    board_size_mm: float
    markers: MarkerConfig
    capture_tray: CaptureTrayConfig
    board_reading: BoardReadingConfig


@dataclass(frozen=True)
class EngineConfig:
    path: str
    extra_search_dirs: tuple[str, ...]
    threads: int
    hash_mb: int
    skill_level: int
    think_time_s: float


@dataclass(frozen=True)
class ClockConfig:
    base_s: float | None  # None = untimed
    increment_s: float


@dataclass(frozen=True)
class GameConfig:
    robot_color: chess.Color
    engine: EngineConfig
    clock: ClockConfig


@dataclass(frozen=True)
class Config:
    robot: RobotConfig
    camera: CameraConfig
    board: BoardConfig
    game: GameConfig


def load_config(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Config:
    config_dir = Path(config_dir)
    return Config(
        robot=load_robot_config(config_dir / "robot.yaml"),
        camera=load_camera_config(config_dir / "camera.yaml"),
        board=load_board_config(config_dir / "board.yaml"),
        game=load_game_config(config_dir / "game.yaml"),
    )


def load_robot_config(path: Path | str) -> RobotConfig:
    path = Path(path)
    raw = _read_yaml(path)
    try:
        robot = RobotConfig(
            servo_channels={j: int(c) for j, c in raw["servo_channels"].items()},
            i2c_address=int(raw["i2c_address"]),
            pwm_frequency_hz=float(raw["pwm_frequency_hz"]),
            pulse_min_us=float(raw["pulse_us"]["min"]),
            pulse_max_us=float(raw["pulse_us"]["max"]),
            servo_travel_deg=float(raw["servo_travel_deg"]),
            joint_limits_deg={
                j: (float(lo), float(hi)) for j, (lo, hi) in raw["joint_limits_deg"].items()
            },
            home_pose_deg={j: float(a) for j, a in raw["home_pose_deg"].items()},
            max_speed_deg_s=float(raw["max_speed_deg_s"]),
            link_lengths_mm={k: float(v) for k, v in raw["link_lengths_mm"].items()},
            gripper=GripperConfig(
                open_deg=float(raw["gripper"]["open_deg"]),
                closed_deg=float(raw["gripper"]["closed_deg"]),
                grip_height_above_board_mm=float(raw["gripper"]["grip_height_above_board_mm"]),
            ),
            grasp_align=GraspAlignConfig(
                tolerance_mm=float(raw["grasp_align"]["tolerance_mm"]),
                max_corrections=int(raw["grasp_align"]["max_corrections"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{path}: missing or malformed value ({e!r})") from e
    _validate_robot(robot, path)
    return robot


def load_camera_config(path: Path | str) -> CameraConfig:
    path = Path(path)
    raw = _read_yaml(path)
    try:
        camera = CameraConfig(
            resolution=_int_pair(raw["resolution"]),
            alignment_resolution=_int_pair(raw["alignment_resolution"]),
            autofocus=bool(raw["autofocus"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{path}: missing or malformed value ({e!r})") from e
    for name in ("resolution", "alignment_resolution"):
        if min(getattr(camera, name)) <= 0:
            raise ConfigError(f"{path}: {name} must be positive, got {getattr(camera, name)}")
    return camera


def load_board_config(path: Path | str) -> BoardConfig:
    path = Path(path)
    raw = _read_yaml(path)
    try:
        board = BoardConfig(
            square_size_mm=float(raw["square_size_mm"]),
            board_size_mm=float(raw["board_size_mm"]),
            markers=MarkerConfig(
                dictionary=str(raw["markers"]["dictionary"]),
                size_mm=float(raw["markers"]["size_mm"]),
                positions={
                    int(i): _float_pair(p) for i, p in (raw["markers"]["positions"] or {}).items()
                },
            ),
            capture_tray=CaptureTrayConfig(
                position_mm=_float_pair(raw["capture_tray"]["position_mm"]),
                slots=int(raw["capture_tray"]["slots"]),
            ),
            board_reading=BoardReadingConfig(
                max_views=int(raw["board_reading"]["max_views"]),
                decision_ratio=float(raw["board_reading"]["decision_ratio"]),
                contradiction_log_odds=float(raw["board_reading"]["contradiction_log_odds"]),
                min_square_entropy_bits=float(raw["board_reading"]["min_square_entropy_bits"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{path}: missing or malformed value ({e!r})") from e
    _validate_board(board, path)
    return board


def load_game_config(path: Path | str) -> GameConfig:
    path = Path(path)
    raw = _read_yaml(path)
    colors = {"white": chess.WHITE, "black": chess.BLACK}
    if raw.get("robot_color") not in colors:
        raise ConfigError(f"{path}: robot_color must be white or black")
    try:
        base_s = raw["clock"]["base_s"]
        game = GameConfig(
            robot_color=colors[raw["robot_color"]],
            engine=EngineConfig(
                path=str(raw["engine"]["path"]),
                extra_search_dirs=tuple(str(d) for d in raw["engine"]["extra_search_dirs"] or ()),
                threads=int(raw["engine"]["threads"]),
                hash_mb=int(raw["engine"]["hash_mb"]),
                skill_level=int(raw["engine"]["skill_level"]),
                think_time_s=float(raw["engine"]["think_time_s"]),
            ),
            clock=ClockConfig(
                base_s=None if base_s is None else float(base_s),
                increment_s=float(raw["clock"]["increment_s"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{path}: missing or malformed value ({e!r})") from e
    _validate_game(game, path)
    return game


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"{path}: file not found") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML ({e})") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return raw


def _int_pair(value: Any) -> tuple[int, int]:
    a, b = value
    return int(a), int(b)


def _float_pair(value: Any) -> tuple[float, float]:
    a, b = value
    return float(a), float(b)


def _validate_robot(robot: RobotConfig, path: Path) -> None:
    joints = set(robot.servo_channels)
    for name in ("joint_limits_deg", "home_pose_deg"):
        keys = set(getattr(robot, name))
        if keys != joints:
            raise ConfigError(
                f"{path}: {name} joints {sorted(keys)} do not match "
                f"servo_channels joints {sorted(joints)}"
            )
    if len(set(robot.servo_channels.values())) != len(robot.servo_channels):
        raise ConfigError(f"{path}: two joints share a servo channel")
    if robot.pulse_min_us >= robot.pulse_max_us:
        raise ConfigError(f"{path}: pulse_us min must be below max")
    for name in ("servo_travel_deg", "pwm_frequency_hz", "max_speed_deg_s"):
        if getattr(robot, name) <= 0:
            raise ConfigError(f"{path}: {name} must be positive")
    for joint, (lo, hi) in robot.joint_limits_deg.items():
        if not 0 <= lo < hi <= robot.servo_travel_deg:
            raise ConfigError(
                f"{path}: joint_limits_deg.{joint} must satisfy "
                f"0 <= min < max <= servo_travel_deg, got [{lo}, {hi}]"
            )
        if not lo <= robot.home_pose_deg[joint] <= hi:
            raise ConfigError(f"{path}: home_pose_deg.{joint} is outside its joint limits")
    if "gripper" in robot.joint_limits_deg:
        lo, hi = robot.joint_limits_deg["gripper"]
        for name in ("open_deg", "closed_deg"):
            if not lo <= getattr(robot.gripper, name) <= hi:
                raise ConfigError(f"{path}: gripper.{name} is outside the gripper joint limits")
    if robot.grasp_align.tolerance_mm <= 0 or robot.grasp_align.max_corrections < 1:
        raise ConfigError(f"{path}: grasp_align needs tolerance_mm > 0 and max_corrections >= 1")


def _validate_board(board: BoardConfig, path: Path) -> None:
    if board.square_size_mm <= 0:
        raise ConfigError(f"{path}: square_size_mm must be positive")
    if not math.isclose(board.square_size_mm * SQUARES_PER_SIDE, board.board_size_mm):
        raise ConfigError(
            f"{path}: board_size_mm ({board.board_size_mm}) must equal "
            f"{SQUARES_PER_SIDE} x square_size_mm ({board.square_size_mm})"
        )
    if board.markers.size_mm <= 0:
        raise ConfigError(f"{path}: markers.size_mm must be positive")
    if board.capture_tray.slots < 1:
        raise ConfigError(f"{path}: capture_tray.slots must be at least 1")
    if board.board_reading.max_views < 1:
        raise ConfigError(f"{path}: board_reading.max_views must be at least 1")
    if board.board_reading.decision_ratio <= 1:
        raise ConfigError(f"{path}: board_reading.decision_ratio must be greater than 1")
    if board.board_reading.contradiction_log_odds <= 0:
        raise ConfigError(f"{path}: board_reading.contradiction_log_odds must be positive")
    if not 0 <= board.board_reading.min_square_entropy_bits < 1:
        raise ConfigError(f"{path}: board_reading.min_square_entropy_bits must be in [0, 1)")


def _validate_game(game: GameConfig, path: Path) -> None:
    engine = game.engine
    if engine.threads < 1 or engine.hash_mb < 1:
        raise ConfigError(f"{path}: engine.threads and engine.hash_mb must be at least 1")
    if engine.think_time_s <= 0:
        raise ConfigError(f"{path}: engine.think_time_s must be positive")
    # The skill level's valid range comes from the engine itself, checked when it starts.
    if game.clock.base_s is not None and game.clock.base_s <= 0:
        raise ConfigError(f"{path}: clock.base_s must be positive, or null for untimed")
    if game.clock.increment_s < 0:
        raise ConfigError(f"{path}: clock.increment_s cannot be negative")
