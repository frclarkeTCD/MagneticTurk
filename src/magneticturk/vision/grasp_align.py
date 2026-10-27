"""Closed-loop visual servoing above a piece before gripping.

Loop: capture -> measure piece offset from gripper centre -> correct -> repeat.
Stop when within tolerance (config) or after max corrections, then report failure.
"""
