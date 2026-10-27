# MagneticTurk

**A vision-guided robotic arm that plays chess on a real board.**

A camera on the arm's wrist reads the position, Stockfish chooses the reply, and the arm
picks up and places its own pieces, then checks with the camera that the move landed correctly.

> Photo / GIF of the robot goes here.

📄 **[Data sheet (PDF)](docs/datasheet/magneticturk_datasheet.pdf)**: full description, block diagrams and specifications.

## How it works

The camera rides on the arm, so the robot moves to see better.

- **Active board reading**: each view gives every square a confidence of being occupied.
  Evidence is combined across views, and the robot moves the camera to the next-best
  viewpoint until exactly one legal move explains what it sees.
- **Move inference without piece recognition**: the game history tells the robot which piece
  is where, so it only needs occupancy, not piece identity.
- **Visual servoing for grasps**: the arm moves above a piece, then the camera measures the
  offset and corrects it in small steps before gripping.
- **Verification**: every move is checked with the camera; failures are retried once, then reported.

The clock, prompts and errors appear on a web page served by the robot, which you open on any phone or computer.

## Hardware

Raspberry Pi Zero 2 W · Raspberry Pi Camera Module 3 · PCA9685 servo controller ·
6DOF metal arm (5 joints + gripper) · separate 6 V servo supply · printed ArUco board markers.

## Repository layout

| Path | Contents |
|------|----------|
| `src/magneticturk/` | The application: vision, robot, game, supervisor, web |
| `config/` | All tunable numbers: servo limits, board geometry, camera settings |
| `tools/` | Scripts run by hand: servo jog, calibration, marker generation |
| `tests/` | Tests that run on a laptop using fake hardware |
| `docs/` | Data sheet and figures |

## Running

Development happens on a laptop with fake hardware:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m magneticturk.main --fake
```

On the Raspberry Pi, see [`docs/pi-setup.md`](docs/pi-setup.md).

## Status

🚧 In development. See [Issues](../../issues) for the build plan.

## Acknowledgements

Version 1 (magnetic under-board design) was based on ChessBot V1 by Joshua Stanley Robotics (MIT).
Stockfish is GPLv3 and is run as a separate program.
AI tools (Anthropic's Claude) were used as a design, writing and coding assistant on this personal project.
