"""Servo interface.

Servos (interface): set_angles({joint: degrees}).
PCA9685Servos: real implementation over I2C. MUST clamp every command to config limits.
FakeServos: records commanded angles so tests and plots can inspect them.

Both implementations clamp through the same clamp_angles(), so behaviour seen in
tests with FakeServos is exactly what the real arm receives. Speed limiting is the
job of robot/motion.py, which plans small steps; this layer only enforces limits.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from magneticturk.config import RobotConfig

log = logging.getLogger(__name__)

PCA9685_DUTY_MAX = 0xFFFF  # adafruit_pca9685 duty_cycle is 16-bit
MICROSECONDS_PER_SECOND = 1_000_000


def clamp_angles(angles: Mapping[str, float], config: RobotConfig) -> dict[str, float]:
    """Clamp each angle to its joint limits. Unknown joints and NaN/inf are errors."""
    clamped = {}
    for joint, angle in angles.items():
        if joint not in config.joint_limits_deg:
            raise KeyError(f"Unknown joint {joint!r}, expected one of {config.joints}")
        if not math.isfinite(angle):
            raise ValueError(f"Angle for {joint} is not a finite number: {angle}")
        lo, hi = config.joint_limits_deg[joint]
        safe = min(max(float(angle), lo), hi)
        if safe != angle:
            log.warning(
                "Clamped %s from %.1f to %.1f deg (limits [%s, %s])", joint, angle, safe, lo, hi
            )
        clamped[joint] = safe
    return clamped


def angle_to_pulse_us(angle: float, config: RobotConfig) -> float:
    span = config.pulse_max_us - config.pulse_min_us
    return config.pulse_min_us + span * angle / config.servo_travel_deg


class Servos(ABC):
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.positions: dict[str, float] = {}  # last commanded (clamped) angle per joint

    def set_angles(self, angles: Mapping[str, float]) -> dict[str, float]:
        """Command the given joints. Returns the angles actually sent, after clamping."""
        safe = clamp_angles(angles, self.config)
        self._send(safe)
        self.positions.update(safe)
        return safe

    def home(self) -> dict[str, float]:
        return self.set_angles(self.config.home_pose_deg)

    def close(self) -> None:
        """Stop driving the servos. Optional for implementations with nothing to release."""

    @abstractmethod
    def _send(self, angles: Mapping[str, float]) -> None:
        """Send already-clamped angles to the hardware."""


@dataclass(frozen=True)
class ServoCommand:
    requested: dict[str, float]
    sent: dict[str, float]


class FakeServos(Servos):
    """Records every command, both as requested and as sent after clamping."""

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self.history: list[ServoCommand] = []

    def set_angles(self, angles: Mapping[str, float]) -> dict[str, float]:
        sent = super().set_angles(angles)
        self.history.append(ServoCommand(requested=dict(angles), sent=sent))
        return sent

    def _send(self, angles: Mapping[str, float]) -> None:
        pass


class PCA9685Servos(Servos):
    """Servos on a PCA9685 board over I2C.

    `pca` can be any object with `frequency` and `channels[i].duty_cycle`, so tests can
    check the exact duty cycles sent without hardware. Left as None, the real
    adafruit_pca9685 driver is created (Pi only).
    """

    def __init__(self, config: RobotConfig, pca: object | None = None) -> None:
        super().__init__(config)
        if pca is None:
            pca = _open_pca9685(config.i2c_address)
        self._pca = pca
        self._pca.frequency = config.pwm_frequency_hz

    def _send(self, angles: Mapping[str, float]) -> None:
        period_us = MICROSECONDS_PER_SECOND / self.config.pwm_frequency_hz
        for joint, angle in angles.items():
            pulse_us = angle_to_pulse_us(angle, self.config)
            duty = round(pulse_us / period_us * PCA9685_DUTY_MAX)
            self._pca.channels[self.config.servo_channels[joint]].duty_cycle = duty

    def close(self) -> None:
        """Turn every channel's output off (servos go limp) and release I2C."""
        for channel in self.config.servo_channels.values():
            self._pca.channels[channel].duty_cycle = 0
        deinit = getattr(self._pca, "deinit", None)
        if deinit is not None:
            deinit()


def _open_pca9685(address: int) -> object:
    import board
    import busio
    from adafruit_pca9685 import PCA9685

    return PCA9685(busio.I2C(board.SCL, board.SDA), address=address)
