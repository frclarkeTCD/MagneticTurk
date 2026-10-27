"""Combine occupancy evidence across views.

Keeps a per-square belief in log-odds; each new view adds weighted evidence.
Uncertain squares are those whose belief is near 0.5.
"""
