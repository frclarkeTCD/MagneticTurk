import shutil
from pathlib import Path

import chess
import pytest
import yaml

from magneticturk.config import DEFAULT_CONFIG_DIR, ConfigError, load_config


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A writable copy of the repo's config folder."""
    target = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target


def edit_yaml(path: Path, edit) -> None:
    data = yaml.safe_load(path.read_text())
    edit(data)
    path.write_text(yaml.safe_dump(data))


def test_repo_config_loads():
    config = load_config()
    assert config.robot.joints == (
        "base",
        "shoulder",
        "elbow",
        "wrist_pitch",
        "wrist_roll",
        "gripper",
    )
    assert config.robot.joint_limits_deg["shoulder"] == (20.0, 160.0)
    assert config.robot.i2c_address == 0x40
    assert config.camera.resolution == (1280, 960)
    assert config.board.board_reading.max_views == 4


def test_config_is_read_only():
    config = load_config()
    with pytest.raises(AttributeError):
        config.robot.max_speed_deg_s = 1000


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="file not found"):
        load_config(tmp_path)


def test_missing_key_names_the_file(config_dir: Path):
    edit_yaml(config_dir / "robot.yaml", lambda d: d.pop("max_speed_deg_s"))
    with pytest.raises(ConfigError, match="robot.yaml.*max_speed_deg_s"):
        load_config(config_dir)


def test_invalid_yaml(config_dir: Path):
    (config_dir / "camera.yaml").write_text("resolution: [1280, 960\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(config_dir)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["joint_limits_deg"].pop("elbow"), "do not match"),
        (lambda d: d["home_pose_deg"].pop("base"), "do not match"),
        (lambda d: d["servo_channels"].update(elbow=0), "share a servo channel"),
        (lambda d: d["joint_limits_deg"].update(elbow=[160, 20]), "elbow"),
        (lambda d: d["joint_limits_deg"].update(base=[0, 270]), "servo_travel_deg"),
        (lambda d: d["home_pose_deg"].update(shoulder=10), "outside its joint limits"),
        (lambda d: d["gripper"].update(closed_deg=95), "gripper.closed_deg"),
        (lambda d: d["pulse_us"].update(min=2500, max=500), "min must be below max"),
        (lambda d: d.update(pwm_frequency_hz=0), "pwm_frequency_hz"),
        (lambda d: d["grasp_align"].update(max_corrections=0), "grasp_align"),
    ],
)
def test_robot_validation(config_dir: Path, edit, message):
    edit_yaml(config_dir / "robot.yaml", edit)
    with pytest.raises(ConfigError, match=message):
        load_config(config_dir)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(board_size_mm=230), "8 x square_size_mm"),
        (lambda d: d["board_reading"].update(max_views=0), "max_views"),
        (lambda d: d["board_reading"].update(decision_ratio=1), "decision_ratio"),
        (lambda d: d["capture_tray"].update(slots=0), "slots"),
    ],
)
def test_board_validation(config_dir: Path, edit, message):
    edit_yaml(config_dir / "board.yaml", edit)
    with pytest.raises(ConfigError, match=message):
        load_config(config_dir)


def test_marker_positions(config_dir: Path):
    edit_yaml(
        config_dir / "board.yaml",
        lambda d: d["markers"].update(positions={0: [-10, -10], 7: [234, 112]}),
    )
    markers = load_config(config_dir).board.markers
    assert markers.positions == {0: (-10.0, -10.0), 7: (234.0, 112.0)}


def test_camera_resolution_must_be_positive(config_dir: Path):
    edit_yaml(config_dir / "camera.yaml", lambda d: d.update(resolution=[0, 960]))
    with pytest.raises(ConfigError, match="resolution must be positive"):
        load_config(config_dir)


def test_game_config():
    game = load_config().game
    assert game.robot_color == chess.BLACK
    assert game.engine.threads == 1
    assert game.clock.base_s == 600


def test_untimed_clock(config_dir: Path):
    edit_yaml(config_dir / "game.yaml", lambda d: d["clock"].update(base_s=None))
    assert load_config(config_dir).game.clock.base_s is None


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(robot_color="red"), "robot_color"),
        (lambda d: d["engine"].update(think_time_s=0), "think_time_s"),
        (lambda d: d["engine"].update(threads=0), "threads"),
        (lambda d: d["clock"].update(base_s=0), "base_s"),
        (lambda d: d["clock"].update(increment_s=-1), "increment_s"),
    ],
)
def test_game_validation(config_dir: Path, edit, message):
    edit_yaml(config_dir / "game.yaml", edit)
    with pytest.raises(ConfigError, match=message):
        load_config(config_dir)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["board_reading"].update(contradiction_log_odds=0), "contradiction_log_odds"),
        (lambda d: d["board_reading"].update(min_square_entropy_bits=1), "min_square_entropy"),
    ],
)
def test_board_reading_validation(config_dir: Path, edit, message):
    edit_yaml(config_dir / "board.yaml", edit)
    with pytest.raises(ConfigError, match=message):
        load_config(config_dir)
