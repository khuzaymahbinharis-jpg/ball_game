"""Classical ball tracking initialized and corrected only by VLM detections."""

from dataclasses import dataclass, replace
from math import hypot, log
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from .config import BallTrackingConfig
from .schema import BallDetection, Point
from .tracking import TrackedFrame, _hungarian_minimize


@dataclass(frozen=True)
class BallTrackFrame:
    frame_id: int
    position: Point | None
    confidence: float | None
    source: str
    lost: bool


@dataclass(frozen=True)
class BallTrackingStats:
    vlm_corrections: int = 0
    hungarian_matches: int = 0
    rejected_vlm_anchors: int = 0
    optical_flow_updates: int = 0
    predicted_updates: int = 0
    possession_prior_updates: int = 0
    lost_frames: int = 0
    lost_events: int = 0
    recovered_events: int = 0
    scene_cut_resets: int = 0


@dataclass(frozen=True)
class BallTrackingResult:
    frames: tuple[BallTrackFrame, ...]
    stats: BallTrackingStats


class VLMInitializedBallTracker:
    """VLM-only ball anchors with Hungarian gating, LK/Kalman, and lost state."""

    def __init__(self, config: BallTrackingConfig | None = None):
        self.config = config or BallTrackingConfig(enabled=True)

    @staticmethod
    def _gray(frame: Any, frames_are_bgr: bool) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            return array.astype(np.uint8, copy=False)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("ball tracker frames must be grayscale, RGB, or BGR")
        conversion = cv2.COLOR_BGR2GRAY if frames_are_bgr else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array[:, :, :3], conversion)

    @staticmethod
    def _kalman(center: np.ndarray) -> Any:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kalman.processNoiseCov = np.diag([0.35, 0.35, 1.5, 1.5]).astype(np.float32)
        kalman.measurementNoiseCov = np.diag([1.5, 1.5]).astype(np.float32)
        kalman.errorCovPost = np.diag([2.0, 2.0, 5.0, 5.0]).astype(np.float32)
        kalman.statePost = np.array(
            [[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32
        )
        return kalman

    def _features(
        self,
        gray: np.ndarray,
        center: np.ndarray,
        detection: BallDetection | None = None,
    ) -> np.ndarray:
        height, width = gray.shape
        radius = self.config.default_roi_radius_px
        if detection is not None and detection.box is not None:
            half_width = (detection.box.right - detection.box.left) * width / 2
            half_height = (detection.box.bottom - detection.box.top) * height / 2
            radius = max(radius, round(max(half_width, half_height) * 2.5))
        x, y = int(round(center[0])), int(round(center[1]))
        mask = np.zeros_like(gray)
        cv2.rectangle(
            mask,
            (max(0, x - radius), max(0, y - radius)),
            (min(width - 1, x + radius), min(height - 1, y + radius)),
            255,
            -1,
        )
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features,
            qualityLevel=0.01,
            minDistance=2,
            mask=mask,
            blockSize=3,
        )
        if features is None:
            features = center.astype(np.float32).reshape(1, 1, 2)
        return features.astype(np.float32)

    def _flow(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        features: np.ndarray,
        center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        settings = dict(
            winSize=(self.config.lk_window_size, self.config.lk_window_size),
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        next_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
            previous, current, features, None, **settings
        )
        if next_points is None or forward_status is None or forward_error is None:
            return None
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, next_points, None, **settings
        )
        if back_points is None or back_status is None:
            return None
        forward_status = forward_status.reshape(-1).astype(bool)
        back_status = back_status.reshape(-1).astype(bool)
        errors = forward_error.reshape(-1)
        fb_error = np.linalg.norm(features.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)
        valid = (
            forward_status
            & back_status
            & (errors <= self.config.max_lk_error)
            & (fb_error <= self.config.max_forward_backward_error_px)
        )
        if int(valid.sum()) < self.config.min_tracked_points:
            return None
        old_valid = features.reshape(-1, 2)[valid]
        new_valid = next_points.reshape(-1, 2)[valid]
        displacement = np.median(new_valid - old_valid, axis=0)
        height, width = current.shape
        if hypot(float(displacement[0]), float(displacement[1])) / hypot(width, height) > self.config.max_displacement_fraction:
            return None
        observation = center + displacement
        if not (0 <= observation[0] < width and 0 <= observation[1] < height):
            return None
        quality = float(valid.mean()) * max(
            0.0, 1.0 - float(errors[valid].mean()) / self.config.max_lk_error
        )
        return observation.astype(np.float32), new_valid.reshape(-1, 1, 2), quality

    @staticmethod
    def _predicted_center(center: np.ndarray, kalman: Any | None) -> np.ndarray:
        if kalman is None:
            return center.copy()
        predicted_state = kalman.transitionMatrix @ kalman.statePost
        return predicted_state[:2].reshape(-1).astype(np.float32)

    @staticmethod
    def _box_size_change(
        previous: BallDetection | None, current: BallDetection
    ) -> float:
        if previous is None or previous.box is None or current.box is None:
            return 0.0
        previous_area = (
            (previous.box.right - previous.box.left)
            * (previous.box.bottom - previous.box.top)
        )
        current_area = (
            (current.box.right - current.box.left)
            * (current.box.bottom - current.box.top)
        )
        if previous_area <= 0 or current_area <= 0:
            return 1.0
        return min(abs(log(current_area / previous_area)), 2.0) / 2.0

    def _association_cost(
        self,
        center: np.ndarray,
        predicted: np.ndarray,
        detection: BallDetection,
        previous_detection: BallDetection | None,
        width: int,
        height: int,
    ) -> float:
        """Score a VLM candidate against the persistent ball trajectory."""

        settings = self.config.hungarian
        candidate = np.array(
            [detection.center.x * width, detection.center.y * height],
            dtype=np.float32,
        )
        diagonal = hypot(width, height)
        center_distance = hypot(
            float(candidate[0] - center[0]), float(candidate[1] - center[1])
        ) / (diagonal * settings.distance_scale)
        motion_distance = hypot(
            float(candidate[0] - predicted[0]), float(candidate[1] - predicted[1])
        ) / (diagonal * settings.distance_scale)
        return (
            settings.center_distance_weight * center_distance
            + settings.motion_distance_weight * motion_distance
            + settings.size_change_weight
            * self._box_size_change(previous_detection, detection)
            + settings.detection_confidence_weight * (1.0 - detection.confidence)
        )

    def _hungarian_match_cost(
        self,
        center: np.ndarray,
        predicted: np.ndarray,
        detection: BallDetection,
        previous_detection: BallDetection | None,
        width: int,
        height: int,
    ) -> float | None:
        """Return an accepted global-assignment cost, or None for the dummy match.

        There is currently one semantic ball candidate per VLM response, so this is
        a 1x1 real assignment plus an unmatched dummy column. Keeping the same
        Hungarian formulation as player tracking makes the gate deterministic and
        ready for a future schema that can expose multiple VLM ball candidates.
        """

        cost = self._association_cost(
            center,
            predicted,
            detection,
            previous_detection,
            width,
            height,
        )
        limit = self.config.hungarian.max_assignment_cost
        pairs = _hungarian_minimize([[cost, limit + 1e-6]])
        if pairs == [(0, 0)] and cost <= limit:
            return cost
        return None

    def track_frames(
        self,
        frames: Iterable[Any],
        anchors: dict[int, BallDetection | None],
        player_timeline: Iterable[TrackedFrame] | None = None,
        *,
        frames_are_bgr: bool = False,
        scene_cut_frames: set[int] | None = None,
    ) -> BallTrackingResult:
        player_by_frame = (
            {} if player_timeline is None else {frame.frame_id: frame for frame in player_timeline}
        )
        output: list[BallTrackFrame] = []
        stats = BallTrackingStats()
        previous_gray: np.ndarray | None = None
        center: np.ndarray | None = None
        features: np.ndarray | None = None
        kalman: Any | None = None
        confidence = 0.0
        coast_frames = 0
        lost = True
        ever_initialized = False
        last_vlm_detection: BallDetection | None = None
        cut_frames = scene_cut_frames or set()

        for frame_id, frame in enumerate(frames):
            gray = self._gray(frame, frames_are_bgr)
            height, width = gray.shape
            anchor = anchors.get(frame_id) if frame_id in anchors else None
            if frame_id in cut_frames:
                if not lost:
                    stats = replace(stats, lost_events=stats.lost_events + 1)
                center = None
                features = None
                kalman = None
                confidence = 0.0
                coast_frames = 0
                lost = True
                last_vlm_detection = None
                stats = replace(
                    stats, scene_cut_resets=stats.scene_cut_resets + 1
                )
            if frame_id in anchors and anchor is not None:
                was_lost = lost
                anchor_center = np.array(
                    [anchor.center.x * width, anchor.center.y * height], dtype=np.float32
                )
                assignment_cost: float | None = None
                accept_anchor = was_lost or center is None
                if not accept_anchor and self.config.association_method == "direct":
                    accept_anchor = True
                elif not accept_anchor and center is not None:
                    assignment_cost = self._hungarian_match_cost(
                        center,
                        self._predicted_center(center, kalman),
                        anchor,
                        last_vlm_detection,
                        width,
                        height,
                    )
                    accept_anchor = assignment_cost is not None
                if accept_anchor:
                    if kalman is None or not self.config.use_kalman or was_lost:
                        kalman = self._kalman(anchor_center) if self.config.use_kalman else None
                        center = anchor_center
                    else:
                        kalman.predict()
                        corrected = kalman.correct(anchor_center.reshape(2, 1)).reshape(-1)
                        center = corrected[:2].astype(np.float32)
                    features = self._features(gray, center, anchor)
                    if assignment_cost is None:
                        confidence = anchor.confidence
                    else:
                        settings = self.config.hungarian
                        assignment_quality = max(
                            0.0,
                            1.0 - assignment_cost / settings.max_assignment_cost,
                        )
                        total_weight = (
                            settings.anchor_detection_weight + settings.assignment_weight
                        )
                        confidence = (
                            settings.anchor_detection_weight * anchor.confidence
                            + settings.assignment_weight * assignment_quality
                        ) / total_weight
                    coast_frames = 0
                    lost = False
                    last_vlm_detection = anchor
                    stats = replace(
                        stats,
                        vlm_corrections=stats.vlm_corrections + 1,
                        hungarian_matches=stats.hungarian_matches
                        + int(assignment_cost is not None),
                        recovered_events=stats.recovered_events
                        + int(was_lost and ever_initialized),
                    )
                    ever_initialized = True
                    point = Point(
                        min(1.0, max(0.0, float(center[0]) / width)),
                        min(1.0, max(0.0, float(center[1]) / height)),
                    )
                    output.append(BallTrackFrame(frame_id, point, confidence, "vlm", False))
                    previous_gray = gray
                    continue
                stats = replace(
                    stats,
                    rejected_vlm_anchors=stats.rejected_vlm_anchors + 1,
                )

            predicted: np.ndarray | None = None
            if not lost and center is not None:
                if self.config.use_kalman and kalman is not None:
                    prediction = kalman.predict().reshape(-1)
                    predicted = prediction[:2].astype(np.float32)
                else:
                    predicted = center.copy()
                flow = (
                    None
                    if previous_gray is None or features is None
                    else self._flow(previous_gray, gray, features, center)
                )
                if flow is not None:
                    observation, features, quality = flow
                    if self.config.use_kalman and kalman is not None:
                        corrected = kalman.correct(observation.reshape(2, 1)).reshape(-1)
                        center = corrected[:2].astype(np.float32)
                    else:
                        center = observation
                    confidence *= self.config.flow_confidence_decay * (0.75 + 0.25 * quality)
                    coast_frames = 0
                    stats = replace(
                        stats, optical_flow_updates=stats.optical_flow_updates + 1
                    )
                    if len(features) < max(4, self.config.min_tracked_points):
                        features = self._features(gray, center)
                    source = "optical_flow"
                else:
                    center = predicted
                    coast_frames += 1
                    confidence *= self.config.coast_confidence_decay
                    player_frame = player_by_frame.get(frame_id)
                    possessor = None if player_frame is None else next(
                        (player for player in player_frame.players if player.possesses_ball), None
                    )
                    if possessor is not None and center is not None:
                        target = np.array(
                            [
                                (possessor.box.left + possessor.box.right) * width / 2,
                                (possessor.box.top + possessor.box.bottom) * height / 2,
                            ],
                            dtype=np.float32,
                        )
                        normalized_distance = hypot(
                            float(center[0] - target[0]) / width,
                            float(center[1] - target[1]) / height,
                        )
                        if normalized_distance <= self.config.possession_prior_max_distance:
                            weight = self.config.possession_prior_weight
                            center = center * (1 - weight) + target * weight
                            stats = replace(
                                stats,
                                possession_prior_updates=stats.possession_prior_updates + 1,
                            )
                    stats = replace(stats, predicted_updates=stats.predicted_updates + 1)
                    source = "predicted"

                if (
                    center is None
                    or not (0 <= center[0] < width and 0 <= center[1] < height)
                    or coast_frames > self.config.max_coast_frames
                    or confidence < self.config.lost_confidence_threshold
                ):
                    lost = True
                    features = None
                    stats = replace(stats, lost_events=stats.lost_events + 1)
                if not lost and center is not None:
                    point = Point(
                        min(1.0, max(0.0, float(center[0]) / width)),
                        min(1.0, max(0.0, float(center[1]) / height)),
                    )
                    output.append(BallTrackFrame(frame_id, point, confidence, source, False))
                else:
                    stats = replace(stats, lost_frames=stats.lost_frames + 1)
                    output.append(BallTrackFrame(frame_id, None, None, "lost", True))
            else:
                stats = replace(stats, lost_frames=stats.lost_frames + 1)
                output.append(BallTrackFrame(frame_id, None, None, "lost", True))
            previous_gray = gray

        return BallTrackingResult(tuple(output), stats)

    def track_pil_frames(
        self,
        frames: list[Image.Image],
        anchors: dict[int, BallDetection | None],
        player_timeline: Iterable[TrackedFrame] | None = None,
        scene_cut_frames: set[int] | None = None,
    ) -> BallTrackingResult:
        return self.track_frames(
            frames,
            anchors,
            player_timeline,
            scene_cut_frames=scene_cut_frames,
        )

    def track_video(
        self,
        path: str | Path,
        anchors: dict[int, BallDetection | None],
        player_timeline: Iterable[TrackedFrame] | None = None,
        scene_cut_frames: set[int] | None = None,
    ) -> BallTrackingResult:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(path)

        def decoded_frames() -> Iterable[np.ndarray]:
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    yield frame
            finally:
                capture.release()

        return self.track_frames(
            decoded_frames(),
            anchors,
            player_timeline,
            frames_are_bgr=True,
            scene_cut_frames=scene_cut_frames,
        )


def apply_ball_track(
    timeline: list[TrackedFrame], result: BallTrackingResult
) -> list[TrackedFrame]:
    by_frame = {frame.frame_id: frame for frame in result.frames}
    if set(by_frame) != {frame.frame_id for frame in timeline}:
        raise ValueError("ball track and player timeline must cover the same frame IDs")
    return [
        replace(
            frame,
            ball=by_frame[frame.frame_id].position,
            ball_confidence=by_frame[frame.frame_id].confidence,
        )
        for frame in timeline
    ]
