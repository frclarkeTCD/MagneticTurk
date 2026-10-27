"""Servo interface.

Servos (interface): set_angles({joint: degrees}).
PCA9685Servos: real implementation over I2C. MUST clamp every command to config limits.
FakeServos: records commanded angles so tests and plots can inspect them.
"""
