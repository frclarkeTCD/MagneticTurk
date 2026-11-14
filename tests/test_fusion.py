import numpy as np
import pytest

from magneticturk.config import load_config
from magneticturk.vision.fusion import BoardBelief, ViewEvidence


@pytest.fixture
def fusion():
    return load_config().vision.fusion


def view(p: float, quality: float = 1.0) -> ViewEvidence:
    return ViewEvidence(probability=np.full(64, p), quality=np.full(64, quality))


def test_starts_with_no_evidence(fusion):
    belief = BoardBelief(fusion)
    assert np.all(belief.log_odds == 0)
    assert np.allclose(belief.probability, 0.5)
    assert belief.views == 0


def test_views_add_up(fusion):
    belief = BoardBelief(fusion)
    belief.add(view(0.8))
    once = belief.log_odds[0]
    belief.add(view(0.8))
    assert belief.log_odds[0] == pytest.approx(2 * once)
    assert belief.views == 2


def test_quality_scales_evidence(fusion):
    full, half = BoardBelief(fusion), BoardBelief(fusion)
    full.add(view(0.8, quality=1.0))
    half.add(view(0.8, quality=0.5))
    assert half.log_odds[0] == pytest.approx(full.log_odds[0] / 2)


def test_unseen_squares_are_unchanged(fusion):
    belief = BoardBelief(fusion)
    quality = np.zeros(64)
    quality[10] = 1
    belief.add(ViewEvidence(probability=np.full(64, 0.9), quality=quality))
    assert belief.log_odds[10] > 0
    assert np.count_nonzero(belief.log_odds) == 1


def test_disagreeing_views_cancel(fusion):
    belief = BoardBelief(fusion)
    belief.add(view(0.8))
    belief.add(view(0.2))
    assert np.allclose(belief.log_odds, 0)


def test_one_certain_view_is_capped(fusion):
    belief = BoardBelief(fusion)
    belief.add(view(1.0))
    assert np.all(np.isfinite(belief.log_odds))
    assert belief.log_odds[0] == pytest.approx(fusion.max_view_log_odds)
    belief.add(view(0.0))
    assert np.allclose(belief.log_odds, 0)


@pytest.mark.parametrize(
    ("probability", "quality"),
    [
        (np.full(63, 0.5), np.ones(63)),
        (np.full(64, 1.5), np.ones(64)),
        (np.full(64, 0.5), -np.ones(64)),
    ],
)
def test_rejects_bad_evidence(probability, quality):
    with pytest.raises(ValueError):
        ViewEvidence(probability=probability, quality=quality)
