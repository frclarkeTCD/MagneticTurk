"""Forward and inverse kinematics for the 5-joint arm.

IK: gripper position (+ pointing straight down) -> joint angles, or 'unreachable'.
Also checks targets against joint limits and the measured workspace.
"""
