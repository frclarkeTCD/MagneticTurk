"""Verification and recovery.

After every action, compare the observed board with the expected position.
On mismatch: retry once, then stop, return the arm home, and report a fault.
Saves images and state to logs/ on any failure.
"""
