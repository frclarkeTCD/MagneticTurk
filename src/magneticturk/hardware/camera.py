"""Camera interface.

Camera (interface): capture(resolution) -> image (numpy array, H x W x 3, BGR as OpenCV expects).
PiCamera: real implementation using Picamera2 (Pi only, imported lazily).
FakeCamera: replays images from a folder, for laptop development and tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from magneticturk.config import CameraConfig

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


class Camera(ABC):
    @abstractmethod
    def capture(self, resolution: tuple[int, int]) -> np.ndarray:
        """Return one image of size resolution = (width, height)."""

    def close(self) -> None:
        """Release the camera. Optional: most fakes need nothing."""


class FakeCamera(Camera):
    """Returns pre-recorded images in order, resized to the requested resolution.

    Raises IndexError when the images run out, unless loop=True. Every requested
    resolution is recorded in `requests` so tests can check what was asked for.
    """

    def __init__(self, images: Iterable[np.ndarray], loop: bool = False) -> None:
        self._images = list(images)
        if not self._images:
            raise ValueError("FakeCamera needs at least one image")
        self._loop = loop
        self._next = 0
        self.requests: list[tuple[int, int]] = []

    @classmethod
    def from_folder(cls, folder: Path | str, loop: bool = False) -> FakeCamera:
        """Load every image in folder, in filename order."""
        import cv2

        paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError(f"No images found in {folder}")
        images = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image {path}")
            images.append(image)
        return cls(images, loop=loop)

    def capture(self, resolution: tuple[int, int]) -> np.ndarray:
        self.requests.append(resolution)
        if self._next >= len(self._images):
            if not self._loop:
                raise IndexError(f"FakeCamera ran out of images after {len(self._images)}")
            self._next = 0
        image = self._images[self._next]
        self._next += 1
        return _resize(image, resolution)

    @property
    def remaining(self) -> int:
        return len(self._images) - self._next


class PiCamera(Camera):
    """Raspberry Pi Camera Module 3 through Picamera2. Only importable on the Pi."""

    def __init__(self, config: CameraConfig) -> None:
        from picamera2 import Picamera2

        self._config = config
        self._camera = Picamera2()
        self._resolution: tuple[int, int] | None = None

    def capture(self, resolution: tuple[int, int]) -> np.ndarray:
        if resolution != self._resolution:
            self._configure(resolution)
        # Picamera2's "RGB888" is stored B, G, R in memory, which is what OpenCV expects.
        return self._camera.capture_array("main")

    def close(self) -> None:
        if self._resolution is not None:
            self._camera.stop()
            self._resolution = None
        self._camera.close()

    def _configure(self, resolution: tuple[int, int]) -> None:
        if self._resolution is not None:
            self._camera.stop()
        still = self._camera.create_still_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self._camera.configure(still)
        self._camera.start()
        if self._config.autofocus:
            from libcamera import controls

            self._camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self._resolution = resolution


def _resize(image: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    width, height = resolution
    if image.shape[1] == width and image.shape[0] == height:
        return image.copy()
    import cv2

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
