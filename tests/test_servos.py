import math
from types import SimpleNamespace

import pytest

from magneticturk.config import load_config
from magneticturk.hardware.servos import (
    FakeServos,
    PCA9685Servos,
    Servos,
    angle_to_pulse_us,
    clamp_angles,
)


@pytest.fixture
def robot():
    return load_config().robot


class FakePCA:
    """Stands in for adafruit_pca9685.PCA9685: just stores what was written."""

    def __init__(self) -> None:
        self.frequency = None
        self.channels = [SimpleNamespace(duty_cycle=None) for _ in range(16)]
        self.deinitialised = False

    def deinit(self) -> None:
        self.deinitialised = True


def test_clamp_inside_limits_is_unchanged(robot):
    assert clamp_angles({"shoulder": 90}, robot) == {"shoulder": 90}


def test_clamp_to_each_joints_limits(robot):
    assert clamp_angles({"shoulder": 0, "elbow": 200, "gripper": 5}, robot) == {
        "shoulder": 20,
        "elbow": 160,
        "gripper": 10,
    }


def test_clamp_logs_a_warning(robot, caplog):
    clamp_angles({"elbow": 200}, robot)
    assert "Clamped elbow" in caplog.text


def test_clamp_rejects_unknown_joint(robot):
    with pytest.raises(KeyError, match="knee"):
        clamp_angles({"knee": 90}, robot)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_clamp_rejects_non_finite(robot, bad):
    with pytest.raises(ValueError):
        clamp_angles({"base": bad}, robot)


def test_pulse_mapping(robot):
    assert angle_to_pulse_us(0, robot) == 500
    assert angle_to_pulse_us(90, robot) == 1500
    assert angle_to_pulse_us(180, robot) == 2500


def test_fake_records_requested_and_sent(robot):
    servos = FakeServos(robot)
    assert isinstance(servos, Servos)
    sent = servos.set_angles({"shoulder": 5, "base": 45})
    assert sent == {"shoulder": 20, "base": 45}
    assert servos.history[-1].requested == {"shoulder": 5, "base": 45}
    assert servos.history[-1].sent == {"shoulder": 20, "base": 45}
    assert servos.positions == {"shoulder": 20, "base": 45}


def test_fake_positions_accumulate(robot):
    servos = FakeServos(robot)
    servos.set_angles({"base": 10})
    servos.set_angles({"elbow": 100})
    assert servos.positions == {"base": 10, "elbow": 100}


def test_home(robot):
    servos = FakeServos(robot)
    servos.home()
    assert servos.positions == robot.home_pose_deg


def test_bad_command_sends_nothing(robot):
    servos = FakeServos(robot)
    with pytest.raises(KeyError):
        servos.set_angles({"base": 10, "knee": 90})
    assert servos.history == []
    assert servos.positions == {}


def test_pca9685_sets_frequency(robot):
    pca = FakePCA()
    PCA9685Servos(robot, pca=pca)
    assert pca.frequency == robot.pwm_frequency_hz


def test_pca9685_duty_cycle(robot):
    pca = FakePCA()
    PCA9685Servos(robot, pca=pca).set_angles({"elbow": 90})
    # 1500 us of a 20000 us period at 50 Hz
    assert pca.channels[robot.servo_channels["elbow"]].duty_cycle == round(1500 / 20000 * 0xFFFF)


def test_pca9685_clamps_before_sending(robot):
    pca = FakePCA()
    PCA9685Servos(robot, pca=pca).set_angles({"shoulder": 179})
    _, hi = robot.joint_limits_deg["shoulder"]
    expected = round(angle_to_pulse_us(hi, robot) / 20000 * 0xFFFF)
    assert pca.channels[robot.servo_channels["shoulder"]].duty_cycle == expected


def test_pca9685_close_turns_outputs_off(robot):
    pca = FakePCA()
    servos = PCA9685Servos(robot, pca=pca)
    servos.home()
    servos.close()
    assert all(pca.channels[c].duty_cycle == 0 for c in robot.servo_channels.values())
    assert pca.deinitialised
