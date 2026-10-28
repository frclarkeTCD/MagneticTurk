from pathlib import Path

import cv2
import numpy as np
import pytest

from magneticturk.hardware.camera import Camera, FakeCamera


def solid(value: int, width: int = 64, height: int = 48) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_fake_is_a_camera():
    assert isinstance(FakeCamera([solid(0)]), Camera)


def test_replays_in_order_and_records_requests():
    camera = FakeCamera([solid(10), solid(20)])
    first = camera.capture((64, 48))
    second = camera.capture((64, 48))
    assert first[0, 0, 0] == 10
    assert second[0, 0, 0] == 20
    assert camera.requests == [(64, 48), (64, 48)]
    assert camera.remaining == 0


def test_runs_out_without_loop():
    camera = FakeCamera([solid(0)])
    camera.capture((64, 48))
    with pytest.raises(IndexError):
        camera.capture((64, 48))


def test_loops_when_asked():
    camera = FakeCamera([solid(1), solid(2)], loop=True)
    values = [camera.capture((64, 48))[0, 0, 0] for _ in range(5)]
    assert values == [1, 2, 1, 2, 1]


def test_resizes_to_requested_resolution():
    image = FakeCamera([solid(0, 640, 480)]).capture((320, 240))
    assert image.shape == (240, 320, 3)


def test_capture_returns_a_copy():
    camera = FakeCamera([solid(5)], loop=True)
    camera.capture((64, 48))[:] = 0
    assert camera.capture((64, 48))[0, 0, 0] == 5


def test_needs_an_image():
    with pytest.raises(ValueError):
        FakeCamera([])


def test_from_folder_in_filename_order(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "b.png"), solid(200))
    cv2.imwrite(str(tmp_path / "a.png"), solid(100))
    (tmp_path / "notes.txt").write_text("not an image")
    camera = FakeCamera.from_folder(tmp_path)
    assert camera.remaining == 2
    assert camera.capture((64, 48))[0, 0, 0] == 100
    assert camera.capture((64, 48))[0, 0, 0] == 200


def test_from_empty_folder(tmp_path: Path):
    with pytest.raises(ValueError, match="No images"):
        FakeCamera.from_folder(tmp_path)
