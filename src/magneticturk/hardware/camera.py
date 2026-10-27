"""Camera interface.

Camera (interface): capture(resolution) -> image (numpy array).
PiCamera: real implementation using Picamera2 (Pi only, imported lazily).
FakeCamera: replays images from a folder, for laptop development and tests.
"""
