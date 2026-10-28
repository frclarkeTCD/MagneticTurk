# CLAUDE.md: context for AI assistants working on this repo

## What this project is
MagneticTurk is a chess-playing robot: a 6DOF hobby arm (5 joints + gripper) with a
wrist-mounted camera, controlled by a Raspberry Pi Zero 2 W. It reads the board with
**active vision**, picks a reply with Stockfish, moves pieces with the arm, and verifies
every move with the camera. The full design is in `docs/datasheet/` (read the PDF/tex
before making design-level changes).

## Architecture (mirrors data sheet Figure 3)
camera → perception (vision/) → chess logic (game/) → robot control (robot/) → servos
supervisor.py compares observed vs expected board after every action.
web/ serves the companion display (clocks, settings, moves, prompts, faults).
main.py runs the operating-mode state machine:
Calibration, Ready, PlayerTurn, RobotTurn, Paused, Fault, GameOver.

## Key design decisions (do not change without asking)
1. **Hardware behind interfaces.** `hardware/camera.py` and `hardware/servos.py` each have a
   real and a fake implementation. Everything else depends only on the interface.
   All tests must run on a laptop with fakes, without a Pi, camera or servos.
2. **Active board reading.** Per-square occupancy *probabilities*, weighted by view quality
   (angle, distance, possible occlusion by taller neighbours, glare), fused across views
   in log-odds. A next-best-view planner chooses the next camera pose.
3. **Move inference from hypotheses.** The robot never classifies piece types. Given the
   previous position, enumerate legal moves; each predicts an occupancy change. Stop looking
   once one hypothesis clearly wins (likelihood ratio threshold in config). Max 4 views,
   then pause and ask the player. Promotion piece is confirmed on the companion display.
4. **Board localisation with ArUco markers** around the board edge, so partial views work.
5. **Grasp = closed-loop visual servoing**: measure offset, correct, re-check.
   Converge to < 2 mm, max 3 corrections, else fault. Grip pieces near the top.
6. **All numbers live in `config/*.yaml`.** No magic numbers in code.
7. **Only `geometry/transforms.py` converts between frames** (board, camera, base, gripper).
8. **On failure, save images + state** to `logs/` so they can become test cases.

## Conventions
- Python 3.11+, type hints everywhere, dataclasses for plain data.
- Formatting/linting: `ruff format` and `ruff check`. Tests: `pytest`.
- Every module gets tests in `tests/`. Write tests alongside code, not after.
- Units: millimetres, degrees, seconds. Board squares as python-chess square indices / names.
- Keep dependencies light: this runs on a Pi Zero 2 W with 512 MB RAM. No deep-learning models.
- Small, focused changes; one feature per branch; clear commit messages.

## Safety rules (hardware)
- Never change joint limits, speeds, or the home pose in `config/robot.yaml` without asking.
- Never run anything that moves the real arm. Only the human runs hardware code.
- Real-servo code must clamp every command to configured limits before sending.
- Servos are powered from a separate 6 V supply, never the Pi.

## Build order (update the ✅ as stages complete)
1. ✅ Skeleton, config, fake camera/servos, CI
2. ✅ game/: game state, Stockfish wrapper, clock
3. ⬜ vision/move_inference.py: hypothesis matching on synthetic probabilities
4. ⬜ vision/markers.py + occupancy.py on real phone photos of the board
5. ⬜ vision/fusion.py + view_planner.py with simulated views
6. ⬜ web/: companion display
7. ⬜ Hardware bring-up: tools/servo_jog.py (one servo, bench supply)
8. ⬜ robot/kinematics.py + taught poses, pick-and-place
9. ⬜ vision/grasp_align.py on the real wrist camera
10. ⬜ supervisor.py + full integration + special moves

Current stage: **3**
