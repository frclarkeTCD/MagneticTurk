"""Identify the player's move from occupancy beliefs. No piece recognition needed.

Given the previous position (python-chess Board) and per-square beliefs:
enumerate legal moves, compute each move's predicted occupancy change, score each
hypothesis's likelihood, and return the winner once it beats the runner-up by the
configured ratio. Otherwise report which squares would best tell the leaders apart.
"""
