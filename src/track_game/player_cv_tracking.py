"""Short-horizon classical tracking of players already localized by the VLM."""

from dataclasses import asdict, dataclass
from math import hypot
from pathlib import Path
from statistics import median_low
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np

from .camera_motion import (
    CameraMotionEstimate,
    GlobalCameraMotionEstimator,
    identity_camera_motion,
    transform_points,
)
from .config import PipelineConfig
from .scene_cut import SceneCutDetector, SceneCutResult
from .schema import Box, FrameDetection, Point
from .shot_context import ShotContextDecision, VLMShotContextFilter
from .tracking import PlayerTracker, TrackedFrame, TrackedPlayer


@dataclass(frozen=True)
class PlayerCVTrackingStats:
    frames_processed: int
    anchor_reinitializations: int
    optical_flow_updates: int
    camera_fallback_updates: int
    coast_updates: int
    lost_track_frames: int
    uncertain_track_frames: int
    confirmed_track_frames: int
    uncertainty_recoveries: int
    scene_cut_resets: int
    closeup_suppressed_anchors: int
    camera_seconds: float
    scene_cut_seconds: float
    player_tracking_seconds: float
    video_decode_seconds: float


@dataclass(frozen=True)
class PlayerCVTrackingResult:
    timeline: tuple[TrackedFrame, ...]
    stats: PlayerCVTrackingStats
    camera_motion: tuple[CameraMotionEstimate, ...]
    scene_cuts: tuple[SceneCutResult, ...]
    shot_context: tuple[ShotContextDecision, ...]
    confidence_history: tuple[dict[str, Any], ...]


@dataclass
class _LocalPlayerState:
    player: TrackedPlayer
    box_px: np.ndarray
    foot_px: np.ndarray
    features: np.ndarray
    confidence: float
    coast_frames: int = 0
    last_displacement: np.ndarray | None = None


class VLMInitializedPlayerCVTracker:
    """Batched sparse LK; it cannot create a track without a VLM-initialized player."""

    # VLM anchors correct drift, but snapping all the way to a slightly jittery
    # anchor makes a foot marker visibly jump every sampling interval.  The
    # prediction already represents the immediately preceding video frame, so
    # retain most of that continuity and ease toward the semantic anchor.
    _ANCHOR_GEOMETRY_WEIGHT = 0.45
    _VELOCITY_COAST_DAMPING = 0.82

    def __init__(self, config: PipelineConfig, logical_tracker: PlayerTracker):
        if not config.player_cv_tracking.enabled:
            raise ValueError("player CV tracking must be enabled")
        self.config = config
        self.logical_tracker = logical_tracker
        self.camera_estimator = GlobalCameraMotionEstimator(config.camera_motion)
        self.cut_detector = SceneCutDetector(config.scene_cut)
        self.shot_filter = VLMShotContextFilter(config.shot_context)
        self._states: dict[int, _LocalPlayerState] = {}
        self._previous_visual_state: dict[int, str] = {}
        self._anchor_reinitializations = 0
        self._flow_updates = 0
        self._camera_updates = 0
        self._coast_updates = 0
        self._lost_track_frames = 0
        self._uncertain_track_frames = 0
        self._confirmed_track_frames = 0
        self._uncertainty_recoveries = 0
        self._scene_cut_resets = 0
        self._closeup_suppressed_anchors = 0
        self._last_scene_cut_frame: int | None = None
        self._continuity_horizon_frames = config.player_cv_tracking.max_coast_frames

    @staticmethod
    def _gray(frame: Any, frames_are_bgr: bool) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            return array.astype(np.uint8, copy=False)
        conversion = cv2.COLOR_BGR2GRAY if frames_are_bgr else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array[:, :, :3], conversion)

    def _tracking_gray(self, frame: Any, frames_are_bgr: bool) -> np.ndarray:
        gray = self._gray(frame, frames_are_bgr)
        height, width = gray.shape
        target = self.config.player_cv_tracking.downscale_width
        if width <= target:
            return gray
        scale = target / width
        return cv2.resize(gray, (target, max(1, round(height * scale))))

    def _state_name(self, confidence: float, *, lost: bool = False) -> str:
        settings = self.config.tracking.confidence
        if lost or confidence < settings.lost_threshold:
            return "lost"
        if not settings.enabled or confidence >= settings.confirmed_threshold:
            return "confirmed"
        return "uncertain"

    @staticmethod
    def _box_pixels(player: TrackedPlayer, width: int, height: int) -> np.ndarray:
        return np.array(
            [
                player.box.left * width,
                player.box.top * height,
                player.box.right * width,
                player.box.bottom * height,
            ],
            dtype=np.float32,
        )

    def _features(self, gray: np.ndarray, box: np.ndarray) -> np.ndarray:
        height, width = gray.shape
        left, top, right, bottom = (float(value) for value in box)
        padding = self.config.player_cv_tracking.box_padding_fraction
        pad_x, pad_y = (right - left) * padding, (bottom - top) * padding
        x1, y1 = max(0, round(left - pad_x)), max(0, round(top - pad_y))
        x2, y2 = min(width - 1, round(right + pad_x)), min(height - 1, round(bottom + pad_y))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 1, 2), dtype=np.float32)
        mask = np.zeros_like(gray)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.player_cv_tracking.max_features_per_player,
            qualityLevel=self.config.player_cv_tracking.quality_level,
            minDistance=self.config.player_cv_tracking.min_feature_distance_px,
            mask=mask,
            blockSize=5,
        )
        return (
            np.empty((0, 1, 2), dtype=np.float32)
            if points is None
            else points.astype(np.float32)
        )

    @staticmethod
    def _clip_geometry(
        box: np.ndarray, foot: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        box = box.astype(np.float32)
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        if box[2] <= box[0]:
            box[0] = min(box[0], max(0, width - 2))
            box[2] = min(width - 1, box[0] + 1)
        if box[3] <= box[1]:
            box[1] = min(box[1], max(0, height - 2))
            box[3] = min(height - 1, box[1] + 1)
        foot = np.array(
            [np.clip(foot[0], 0, width - 1), np.clip(foot[1], 0, height - 1)],
            dtype=np.float32,
        )
        return box, foot

    def _tracked_player(
        self, state: _LocalPlayerState, width: int, height: int
    ) -> TrackedPlayer:
        box, foot = self._clip_geometry(state.box_px, state.foot_px, width, height)
        state.box_px, state.foot_px = box, foot
        confidence = min(1.0, max(0.0, state.confidence))
        tracking_state = self._state_name(confidence)
        return TrackedPlayer(
            state.player.track_id,
            state.player.team,
            Box(
                float(box[0]) / width,
                float(box[1]) / height,
                float(box[2]) / width,
                float(box[3]) / height,
            ),
            Point(float(foot[0]) / width, float(foot[1]) / height),
            state.player.confidence,
            state.player.possesses_ball,
            confidence,
            tracking_state,
        )

    def initialize(
        self, players: tuple[TrackedPlayer, ...], gray: np.ndarray
    ) -> None:
        """Reconcile VLM-associated players without blinking retained tracks.

        A track is still created exclusively from a VLM player.  At later
        anchors, matched VLM geometry is blended with the preceding per-frame
        CV prediction, while a player omitted from a single anchor keeps
        coasting for as long as the logical Hungarian track remains eligible
        for reassociation.
        """

        height, width = gray.shape
        previous_states = self._states
        new_states: dict[int, _LocalPlayerState] = {}
        for player in players:
            detected_box = self._box_pixels(player, width, height)
            detected_foot = np.array(
                [player.foot.x * width, player.foot.y * height], dtype=np.float32
            )
            previous_state = previous_states.get(player.track_id)
            if previous_state is None:
                box, foot = detected_box, detected_foot
                last_displacement = None
            else:
                weight = self._ANCHOR_GEOMETRY_WEIGHT
                box = (1.0 - weight) * previous_state.box_px + weight * detected_box
                foot = (1.0 - weight) * previous_state.foot_px + weight * detected_foot
                box, foot = self._clip_geometry(box, foot, width, height)
                last_displacement = previous_state.last_displacement
            confidence = (
                player.tracking_confidence
                if self.config.tracking.confidence.enabled
                else player.confidence
            )
            tracking_state = self._state_name(confidence)
            previous = self._previous_visual_state.get(player.track_id)
            if previous in {"uncertain", "lost"} and tracking_state == "confirmed":
                self._uncertainty_recoveries += 1
            initialized = TrackedPlayer(
                player.track_id,
                player.team,
                Box(
                    float(box[0]) / width,
                    float(box[1]) / height,
                    float(box[2]) / width,
                    float(box[3]) / height,
                ),
                Point(float(foot[0]) / width, float(foot[1]) / height),
                player.confidence,
                player.possesses_ball,
                confidence,
                tracking_state,
            )
            new_states[player.track_id] = _LocalPlayerState(
                initialized,
                box,
                foot,
                self._features(gray, box),
                confidence,
                last_displacement=last_displacement,
            )
            self._previous_visual_state[player.track_id] = tracking_state
            self._anchor_reinitializations += 1

        # Hungarian deliberately retains an unmatched identity for a bounded
        # number of anchors.  Keep its already-VLM-initialized CV state too;
        # the old replacement behaviour removed the marker for exactly the
        # gap in which that identity was awaiting reassociation.
        logical_states = self.logical_tracker.track_states
        confidence_settings = self.config.tracking.confidence
        for track_id, state in previous_states.items():
            if track_id in new_states or logical_states.get(track_id) != "lost":
                continue
            if confidence_settings.enabled:
                state.confidence *= confidence_settings.missed_anchor_decay
            tracking_state = self._state_name(state.confidence)
            state.player = TrackedPlayer(
                state.player.track_id,
                state.player.team,
                state.player.box,
                state.player.foot,
                state.player.confidence,
                state.player.possesses_ball,
                min(1.0, max(0.0, state.confidence)),
                tracking_state,
            )
            self._previous_visual_state[track_id] = tracking_state
            new_states[track_id] = state
        self._states = new_states

    def reset(self) -> None:
        for track_id in self._states:
            self._previous_visual_state[track_id] = "lost"
        self._states.clear()

    @staticmethod
    def _camera_warp(
        state: _LocalPlayerState, camera: CameraMotionEstimate
    ) -> None:
        corners = np.array(
            [
                [state.box_px[0], state.box_px[1]],
                [state.box_px[2], state.box_px[1]],
                [state.box_px[2], state.box_px[3]],
                [state.box_px[0], state.box_px[3]],
            ],
            dtype=np.float32,
        )
        warped = transform_points(corners, camera)
        state.box_px = np.array(
            [warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max()],
            dtype=np.float32,
        )
        state.foot_px = transform_points(state.foot_px.reshape(1, 2), camera)[0]
        if len(state.features):
            state.features = transform_points(
                state.features.reshape(-1, 2), camera
            ).reshape(-1, 1, 2)

    def advance(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        camera: CameraMotionEstimate,
    ) -> tuple[TrackedPlayer, ...]:
        if not self._states:
            return ()
        settings = self.config.player_cv_tracking
        confidence_settings = self.config.tracking.confidence
        height, width = current_gray.shape
        point_groups: dict[int, tuple[int, int]] = {}
        all_points: list[np.ndarray] = []
        offset = 0
        for track_id, state in self._states.items():
            count = len(state.features)
            if count:
                all_points.append(state.features)
                point_groups[track_id] = (offset, offset + count)
                offset += count
        next_points = forward_status = forward_error = back_points = back_status = None
        if all_points:
            previous_points = np.concatenate(all_points).astype(np.float32)
            flow_settings = dict(
                winSize=(settings.lk_window_size, settings.lk_window_size),
                maxLevel=settings.lk_max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            next_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
                previous_gray, current_gray, previous_points, None, **flow_settings
            )
            if next_points is not None:
                back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    current_gray, previous_gray, next_points, None, **flow_settings
                )
        output: list[TrackedPlayer] = []
        for track_id in list(self._states):
            state = self._states[track_id]
            valid_old = valid_new = errors = None
            bounds = point_groups.get(track_id)
            if (
                bounds is not None
                and next_points is not None
                and forward_status is not None
                and forward_error is not None
                and back_points is not None
                and back_status is not None
            ):
                start, end = bounds
                old = previous_points[start:end].reshape(-1, 2)
                new = next_points[start:end].reshape(-1, 2)
                back = back_points[start:end].reshape(-1, 2)
                errors_all = forward_error[start:end].reshape(-1)
                valid = (
                    forward_status[start:end].reshape(-1).astype(bool)
                    & back_status[start:end].reshape(-1).astype(bool)
                    & (errors_all <= settings.max_lk_error)
                    & (
                        np.linalg.norm(old - back, axis=1)
                        <= settings.max_forward_backward_error_px
                    )
                )
                valid_old, valid_new, errors = old[valid], new[valid], errors_all[valid]
            successful = valid_new is not None and len(valid_new) >= settings.min_tracked_points
            if successful:
                displacement = np.median(valid_new - valid_old, axis=0)
                if hypot(float(displacement[0]), float(displacement[1])) / hypot(width, height) > settings.max_displacement_fraction:
                    successful = False
            if successful:
                residual = displacement
                if camera.success:
                    previous_foot = state.foot_px.copy()
                    self._camera_warp(state, camera)
                    residual = displacement - (state.foot_px - previous_foot)
                state.box_px += np.array(
                    [residual[0], residual[1], residual[0], residual[1]],
                    dtype=np.float32,
                )
                state.foot_px += residual
                state.features = valid_new.reshape(-1, 1, 2).astype(np.float32)
                state.last_displacement = displacement.astype(np.float32)
                state.coast_frames = 0
                source_count = 0 if bounds is None else bounds[1] - bounds[0]
                quality = (len(valid_new) / max(1, source_count)) * max(
                    0.0, 1.0 - float(errors.mean()) / settings.max_lk_error
                )
                quality = min(1.0, quality)
                if confidence_settings.enabled:
                    state.confidence *= confidence_settings.flow_decay * (0.98 + 0.02 * quality)
                self._flow_updates += 1
                if len(state.features) < settings.feature_refresh_below:
                    state.features = self._features(current_gray, state.box_px)
            elif camera.success:
                self._camera_warp(state, camera)
                state.coast_frames += 1
                if confidence_settings.enabled:
                    state.confidence *= confidence_settings.camera_fallback_decay * (
                        0.75 + 0.25 * camera.confidence
                    )
                self._camera_updates += 1
                state.features = self._features(current_gray, state.box_px)
            else:
                state.coast_frames += 1
                if confidence_settings.enabled:
                    state.confidence *= confidence_settings.coast_decay
                if state.last_displacement is not None:
                    displacement = (
                        state.last_displacement * self._VELOCITY_COAST_DAMPING
                    ).astype(np.float32)
                    state.box_px += np.array(
                        [
                            displacement[0],
                            displacement[1],
                            displacement[0],
                            displacement[1],
                        ],
                        dtype=np.float32,
                    )
                    state.foot_px += displacement
                    state.last_displacement = displacement
                self._coast_updates += 1
                state.features = self._features(current_gray, state.box_px)
            # The configured short coast limit predates sparse anchor sampling.
            # Keep an initialized marker through the logical reassociation
            # window; scene cuts still clear it immediately, and an expired
            # Hungarian identity is removed at the next anchor reconciliation.
            lost = state.coast_frames > self._continuity_horizon_frames
            if lost:
                self._lost_track_frames += 1
                self._previous_visual_state[track_id] = "lost"
                del self._states[track_id]
                continue
            tracked = self._tracked_player(state, width, height)
            state.player = tracked
            self._previous_visual_state[track_id] = tracked.tracking_state
            if tracked.tracking_state == "uncertain":
                self._uncertain_track_frames += 1
            else:
                self._confirmed_track_frames += 1
            output.append(tracked)
        return tuple(sorted(output, key=lambda player: player.track_id))

    def track_frames(
        self,
        frames: Iterable[Any],
        anchors: dict[int, FrameDetection],
        *,
        frames_are_bgr: bool = False,
    ) -> PlayerCVTrackingResult:
        timeline: list[TrackedFrame] = []
        camera_log: list[CameraMotionEstimate] = []
        cut_log: list[SceneCutResult] = []
        shot_context_log: list[ShotContextDecision] = []
        confidence_log: list[dict[str, Any]] = []
        previous_gray: np.ndarray | None = None
        camera_seconds = scene_seconds = player_seconds = decode_seconds = 0.0

        anchor_ids = sorted(anchors)
        anchor_gaps = [
            right - left
            for left, right in zip(anchor_ids, anchor_ids[1:])
            if right > left
        ]
        if anchor_gaps:
            # include_last can make the final sampling gap shorter than the
            # actual cadence (for example 6, ..., 6, 5 at 5 FPS).
            nominal_gap = median_low(anchor_gaps)
            self._continuity_horizon_frames = max(
                self.config.player_cv_tracking.max_coast_frames,
                nominal_gap * (self.config.tracking.max_missed_anchors + 1),
            )

        for frame_id, frame in enumerate(frames):
            decode_started = perf_counter()
            gray = self._tracking_gray(frame, frames_are_bgr)
            decode_seconds += perf_counter() - decode_started
            camera = identity_camera_motion(frame_id, "first_frame")
            cut = SceneCutResult(frame_id, False, 0.0, 0.0, 0.0, False)
            if previous_gray is not None:
                started = perf_counter()
                camera = self.camera_estimator.estimate(previous_gray, gray, frame_id)
                camera_seconds += perf_counter() - started
                started = perf_counter()
                cut = self.cut_detector.compare(previous_gray, gray, frame_id, camera)
                if (
                    cut.is_cut
                    and self._last_scene_cut_frame is not None
                    and frame_id - self._last_scene_cut_frame
                    < self.config.scene_cut.minimum_gap_frames
                ):
                    cut = SceneCutResult(
                        cut.frame_id,
                        False,
                        cut.score,
                        cut.frame_difference,
                        cut.histogram_change,
                        cut.camera_failure_used,
                        cut.aligned_frame_difference,
                        cut.motion_compensated,
                    )
                scene_seconds += perf_counter() - started
            camera_log.append(camera)
            cut_log.append(cut)
            if cut.is_cut:
                self._last_scene_cut_frame = frame_id
                self.reset()
                self.logical_tracker.mark_scene_cut()
                self._scene_cut_resets += 1

            predicted: tuple[TrackedPlayer, ...] = ()
            if previous_gray is not None and not cut.is_cut:
                started = perf_counter()
                predicted = self.advance(previous_gray, gray, camera)
                player_seconds += perf_counter() - started

            detection = anchors.get(frame_id)
            if detection is not None:
                detection, shot_context = self.shot_filter.filter(detection)
                shot_context_log.append(shot_context)
                if not shot_context.trackable:
                    # Anchor reconciliation normally retains a missed identity.
                    # A close-up is an explicit non-trackable shot boundary, so
                    # clear spatial state instead of coasting markers across it.
                    self.logical_tracker.update(detection)
                    self.reset()
                    self.logical_tracker.mark_scene_cut()
                    self._closeup_suppressed_anchors += 1
                    current = TrackedFrame(frame_id, (), None, None)
                else:
                    if predicted:
                        self.logical_tracker.apply_predictions(predicted, frame_id)
                    tracked = self.logical_tracker.update(detection)
                    self.initialize(tracked.players, gray)
                    current = TrackedFrame(
                        frame_id,
                        tuple(
                            self._states[track_id].player
                            for track_id in sorted(self._states)
                        ),
                        detection.ball,
                        detection.ball_confidence,
                    )
            else:
                current = TrackedFrame(frame_id, predicted, None, None)
            timeline.append(current)
            for player in current.players:
                confidence_log.append(
                    {
                        "frame_id": frame_id,
                        "track_id": player.track_id,
                        "confidence": player.tracking_confidence,
                        "state": player.tracking_state,
                    }
                )
            previous_gray = gray

        return PlayerCVTrackingResult(
            tuple(timeline),
            PlayerCVTrackingStats(
                frames_processed=len(timeline),
                anchor_reinitializations=self._anchor_reinitializations,
                optical_flow_updates=self._flow_updates,
                camera_fallback_updates=self._camera_updates,
                coast_updates=self._coast_updates,
                lost_track_frames=self._lost_track_frames,
                uncertain_track_frames=self._uncertain_track_frames,
                confirmed_track_frames=self._confirmed_track_frames,
                uncertainty_recoveries=self._uncertainty_recoveries,
                scene_cut_resets=self._scene_cut_resets,
                closeup_suppressed_anchors=self._closeup_suppressed_anchors,
                camera_seconds=camera_seconds,
                scene_cut_seconds=scene_seconds,
                player_tracking_seconds=player_seconds,
                video_decode_seconds=decode_seconds,
            ),
            tuple(camera_log),
            tuple(cut_log),
            tuple(shot_context_log),
            tuple(confidence_log),
        )

    def track_video(
        self, path: str | Path, anchors: dict[int, FrameDetection]
    ) -> PlayerCVTrackingResult:
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

        return self.track_frames(decoded_frames(), anchors, frames_are_bgr=True)


def camera_log_as_dict(result: PlayerCVTrackingResult) -> list[dict[str, Any]]:
    return [asdict(item) for item in result.camera_motion]


def scene_cut_log_as_dict(result: PlayerCVTrackingResult) -> list[dict[str, Any]]:
    return [asdict(item) for item in result.scene_cuts]
