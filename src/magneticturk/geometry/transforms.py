"""The ONLY place coordinate frames are converted.

Frames: board (origin at a1 corner, mm), camera, robot base, gripper (tool centre point).
Provides: square name <-> board-frame position, board <-> base (from calibration),
camera <-> gripper (hand-eye calibration), and image pixel <-> board via marker pose.
"""
