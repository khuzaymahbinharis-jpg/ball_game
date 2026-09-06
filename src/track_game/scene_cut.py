"""Cheap non-semantic broadcast scene-cut detection."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .camera_motion import CameraMotionEstimate
from .config import SceneCutConfig


@dataclass(frozen=True)
class SceneCutResult:
    frame_id: int
    is_cut: bool
    score: float
    frame_difference: float
    histogram_change: float
    camera_failure_used: bool


class SceneCutDetector:
    def __init__(self, config: SceneCutConfig | None = None):
        self.config = config or SceneCutConfig(enabled=True)

    @staticmethod
    def _gray(frame: Any, frames_are_bgr: bool) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            return array.astype(np.uint8, copy=False)
        conversion = cv2.COLOR_BGR2GRAY if frames_are_bgr else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array[:, :, :3], conversion)

    def _scaled(self, frame: Any, frames_are_bgr: bool) -> np.ndarray:
        gray = self._gray(frame, frames_are_bgr)
        height, width = gray.shape
        if width <= self.config.downscale_width:
            return gray
        scale = self.config.downscale_width / width
        return cv2.resize(gray, (self.config.downscale_width, max(1, round(height * scale))))

    def compare(
        self,
        previous_frame: Any,
        current_frame: Any,
        frame_id: int,
        camera: CameraMotionEstimate | None = None,
        *,
        frames_are_bgr: bool = False,
    ) -> SceneCutResult:
        if not self.config.enabled:
            return SceneCutResult(frame_id, False, 0.0, 0.0, 0.0, False)
        previous = self._scaled(previous_frame, frames_are_bgr)
        current = self._scaled(current_frame, frames_are_bgr)
        if previous.shape != current.shape:
            return SceneCutResult(frame_id, True, 1.0, 1.0, 1.0, True)
        frame_difference = float(
            np.mean(cv2.absdiff(previous, current), dtype=np.float64) / 255.0
        )
        hist_previous = cv2.calcHist([previous], [0], None, [32], [0, 256])
        hist_current = cv2.calcHist([current], [0], None, [32], [0, 256])
        cv2.normalize(hist_previous, hist_previous)
        cv2.normalize(hist_current, hist_current)
        histogram_change = float(
            cv2.compareHist(hist_previous, hist_current, cv2.HISTCMP_BHATTACHARYYA)
        )
        weight_sum = self.config.frame_difference_weight + self.config.histogram_weight
        score = (
            self.config.frame_difference_weight * frame_difference
            + self.config.histogram_weight * histogram_change
        ) / weight_sum
        camera_failure = camera is not None and not camera.success
        if camera_failure:
            score = min(1.0, score + self.config.camera_failure_bonus)
        is_cut = (
            frame_difference >= self.config.minimum_frame_difference
            and score >= self.config.threshold
        )
        return SceneCutResult(
            frame_id,
            is_cut,
            score,
            frame_difference,
            histogram_change,
            camera_failure,
        )
