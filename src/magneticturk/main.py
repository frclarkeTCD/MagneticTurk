"""Entry point. Runs the operating-mode state machine.

Modes: Calibration -> Ready -> PlayerTurn <-> RobotTurn, plus Paused, Fault, GameOver.
Wires together hardware (real or fake), vision, robot, game, supervisor and web server.

Usage:
    python -m magneticturk.main --fake      # laptop, fake hardware
    python -m magneticturk.main             # on the Pi
"""
from enum import Enum, auto


class Mode(Enum):
    CALIBRATION = auto()
    READY = auto()
    PLAYER_TURN = auto()
    ROBOT_TURN = auto()
    PAUSED = auto()
    FAULT = auto()
    GAME_OVER = auto()


def main() -> None:
    raise NotImplementedError("Stage 10: integration")


if __name__ == "__main__":
    main()
