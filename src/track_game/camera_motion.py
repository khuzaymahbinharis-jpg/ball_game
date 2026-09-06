"""Non-semantic global camera motion estimated from generic image keypoints."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import CameraMotionConfig


@dataclass(frozen=True)
class CameraMotionEstimate:
    frame_id: int
    success: bool
    transform: tuple[tuple[float, float, float], tuple[float, float, float]]
    tracked_points: int
    inlier_points: int
    confidence: float
    reason: str

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.transform, dtype=np.float32)


def identity_camera_motion(frame_id: int, reason: str = "disabled") -> CameraMotionEstimate:
    return CameraMotionEstimate(
        frame_id=frame_id,
        success=False,
        transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        tracked_points=0,
        inlier_points=0,
        confidence=0.0,
        reason=reason,
    )


class GlobalCameraMotionEstimator:
    """Estimate translation or partial affine motion without classifying image content."""

    def __init__(self, config: CameraMotionConfig | None = None):
        self.config = config or CameraMotionConfig(enabled=True)

    @staticmethod
    def _gray(frame: Any, frames_are_bgr: bool) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            return array.astype(np.uint8, copy=False)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("camera-motion frames must be grayscale, RGB, or BGR")
        conversion = cv2.COLOR_BGR2GRAY if frames_are_bgr else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array[:, :, :3], conversion)

    def _scaled(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = gray.shape
        if width <= self.config.downscale_width:
            return gray, 1.0
        scale = self.config.downscale_width / width
        return (
            cv2.resize(gray, (self.config.downscale_width, max(1, round(height * scale)))),
            scale,
        )

    def estimate(
        self,
        previous_frame: Any,
        current_frame: Any,
        frame_id: int,
        *,
        frames_are_bgr: bool = False,
    ) -> CameraMotionEstimate:
        if not self.config.enabled:
            return identity_camera_motion(frame_id)
        previous, previous_scale = self._scaled(
            self._gray(previous_frame, frames_are_bgr)
        )
        current, current_scale = self._scaled(self._gray(current_frame, frames_are_bgr))
        if previous.shape != current.shape or abs(previous_scale - current_scale) > 1e-6:
            return identity_camera_motion(frame_id, "frame_shape_changed")
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_feature_distance_px,
            blockSize=5,
        )
        if points is None or len(points) < self.config.min_tracked_points:
            return identity_camera_motion(frame_id, "too_few_features")
        settings = dict(
            winSize=(self.config.lk_window_size, self.config.lk_window_size),
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points, None, **settings
        )
        if next_points is None or forward_status is None:
            return identity_camera_motion(frame_id, "forward_flow_failed")
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, next_points, None, **settings
        )
        if back_points is None or back_status is None:
            return identity_camera_motion(frame_id, "backward_flow_failed")
        forward_ok = forward_status.reshape(-1).astype(bool)
        backward_ok = back_status.reshape(-1).astype(bool)
        fb_error = np.linalg.norm(
            points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1
        )
        valid = (
            forward_ok
            & backward_ok
            & (fb_error <= self.config.max_forward_backward_error_px)
        )
        tracked = int(valid.sum())
        if tracked < self.config.min_tracked_points:
            return CameraMotionEstimate(
                **{
                    **identity_camera_motion(frame_id, "too_few_tracked_points").__dict__,
                    "tracked_points": tracked,
                }
            )
        old = points.reshape(-1, 2)[valid]
        new = next_points.reshape(-1, 2)[valid]
        if self.config.transform_type == "partial_affine":
            transform, mask = cv2.estimateAffinePartial2D(
                old,
                new,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold_px,
            )
            if transform is None or mask is None:
                return CameraMotionEstimate(
                    **{
                        **identity_camera_motion(frame_id, "affine_estimation_failed").__dict__,
                        "tracked_points": tracked,
                    }
                )
            inliers = mask.reshape(-1).astype(bool)
        else:
            displacement = new - old
            median = np.median(displacement, axis=0)
            residual = np.linalg.norm(displacement - median, axis=1)
            inliers = residual <= self.config.ransac_threshold_px
            transform = np.array(
                [[1.0, 0.0, median[0]], [0.0, 1.0, median[1]]], dtype=np.float32
            )
        inlier_count = int(inliers.sum())
        confidence = (inlier_count / tracked) * min(
            1.0, tracked / (self.config.min_tracked_points * 2)
        )
        if inlier_count < self.config.min_tracked_points or confidence < self.config.minimum_transform_confidence:
            return CameraMotionEstimate(
                frame_id,
                False,
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                tracked,
                inlier_count,
                confidence,
                "low_transform_confidence",
            )
        transform = transform.astype(np.float32)
        transform[:, 2] /= previous_scale
        return CameraMotionEstimate(
            frame_id,
            True,
            tuple(tuple(float(value) for value in row) for row in transform),
            tracked,
            inlier_count,
            confidence,
            "ok",
        )


def transform_points(points: np.ndarray, estimate: CameraMotionEstimate) -> np.ndarray:
    """Apply a successful full-resolution camera transform to N x 2 points."""

    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float32)))
    return homogeneous @ estimate.matrix.T
