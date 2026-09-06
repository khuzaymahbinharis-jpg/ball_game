import cv2
import numpy as np
import pytest

from track_game.camera_motion import GlobalCameraMotionEstimator
from track_game.config import CameraMotionConfig


def test_translation_estimation_and_blank_fallback():
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, size=(120, 160), dtype=np.uint8)
    current = cv2.warpAffine(
        previous,
        np.array([[1, 0, 5], [0, 1, 3]], dtype=np.float32),
        (160, 120),
    )
    estimator = GlobalCameraMotionEstimator(
        CameraMotionConfig(
            enabled=True,
            downscale_width=160,
            max_features=160,
            min_tracked_points=12,
            transform_type="translation",
        )
    )
    result = estimator.estimate(previous, current, 1)
    assert result.success
    assert result.transform[0][2] == pytest.approx(5, abs=0.6)
    assert result.transform[1][2] == pytest.approx(3, abs=0.6)

    fallback = estimator.estimate(np.zeros_like(previous), np.zeros_like(previous), 2)
    assert not fallback.success
    assert fallback.transform == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
