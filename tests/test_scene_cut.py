import cv2
import numpy as np

from track_game.camera_motion import GlobalCameraMotionEstimator
from track_game.config import CameraMotionConfig
from track_game.config import SceneCutConfig
from track_game.scene_cut import SceneCutDetector


def test_abrupt_nonsemantic_frame_change_is_a_cut():
    detector = SceneCutDetector(
        SceneCutConfig(
            enabled=True,
            downscale_width=64,
            threshold=0.3,
            minimum_frame_difference=0.1,
        )
    )
    result = detector.compare(
        np.zeros((48, 64), dtype=np.uint8),
        np.full((48, 64), 255, dtype=np.uint8),
        1,
    )
    assert result.is_cut
    assert result.frame_difference == 1.0


def test_motion_compensation_does_not_treat_same_shot_zoom_as_cut():
    rng = np.random.default_rng(8)
    previous = rng.integers(0, 256, size=(160, 240), dtype=np.uint8)
    transform = np.array([[1.06, 0.0, -7.2], [0.0, 1.06, -4.8]], dtype=np.float32)
    current = cv2.warpAffine(previous, transform, (240, 160))
    camera = GlobalCameraMotionEstimator(
        CameraMotionConfig(
            enabled=True,
            transform_type="partial_affine",
            downscale_width=240,
            min_tracked_points=12,
        )
    ).estimate(previous, current, 1)
    assert camera.success
    detector = SceneCutDetector(
        SceneCutConfig(
            enabled=True,
            downscale_width=240,
            threshold=0.04,
            minimum_frame_difference=0.04,
        )
    )
    without_compensation = detector.compare(previous, current, 1)
    result = detector.compare(previous, current, 1, camera)
    assert without_compensation.is_cut
    assert not result.is_cut
    assert result.motion_compensated
    assert result.aligned_frame_difference < result.frame_difference


def test_hard_cut_stays_a_cut_when_camera_estimation_fails():
    previous = np.zeros((80, 120), dtype=np.uint8)
    current = np.full((80, 120), 255, dtype=np.uint8)
    camera = GlobalCameraMotionEstimator(
        CameraMotionConfig(enabled=True, downscale_width=120, min_tracked_points=6)
    ).estimate(previous, current, 1)
    assert not camera.success
    result = SceneCutDetector(
        SceneCutConfig(
            enabled=True,
            downscale_width=120,
            threshold=0.3,
            minimum_frame_difference=0.1,
        )
    ).compare(previous, current, 1, camera)
    assert result.is_cut
    assert result.camera_failure_used
