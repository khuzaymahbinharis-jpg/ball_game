from dataclasses import replace

import cv2
import numpy as np

from track_game.config import (
    CameraMotionConfig,
    PipelineConfig,
    PlayerCVTrackingConfig,
    SceneCutConfig,
    TrackConfidenceConfig,
    TrackingConfig,
)
from track_game.player_cv_tracking import VLMInitializedPlayerCVTracker
from track_game.schema import FrameDetection
from track_game.tracking import HungarianPlayerTracker


def detection(frame_id, left=0.20):
    return FrameDetection.from_dict(
        {
            "frame_id": frame_id,
            "players": [
                {
                    "detection_id": f"p-{frame_id}",
                    "team": "A",
                    "box": {"left": left, "top": 0.2, "right": left + 0.2, "bottom": 0.8},
                    "foot": {"x": left + 0.1, "y": 0.8},
                    "confidence": 0.9,
                    "team_confidence": 0.9,
                }
            ],
            "ball": None,
            "possession": None,
            "uncertainty_notes": [],
        }
    )


def cv_config(*, cuts=False):
    return PipelineConfig(
        tracking=TrackingConfig(
            association_method="hungarian",
            confidence=TrackConfidenceConfig(enabled=True),
        ),
        player_cv_tracking=PlayerCVTrackingConfig(enabled=True, method="sparse_lk"),
        camera_motion=CameraMotionConfig(enabled=False),
        scene_cut=SceneCutConfig(
            enabled=cuts,
            downscale_width=64,
            threshold=0.3,
            minimum_frame_difference=0.1,
        ),
    )


def textured_player_frame(shift=0):
    frame = np.zeros((100, 160), dtype=np.uint8)
    rng = np.random.default_rng(11)
    patch = rng.integers(30, 255, size=(60, 32), dtype=np.uint8)
    frame[20:80, 32 + shift : 64 + shift] = patch
    return frame


def test_sparse_lk_follows_only_vlm_initialized_player_and_resets():
    config = cv_config()
    tracker = VLMInitializedPlayerCVTracker(
        config, HungarianPlayerTracker(config.tracking)
    )
    result = tracker.track_frames(
        [textured_player_frame(0), textured_player_frame(4)], {0: detection(0)}
    )
    assert len(result.timeline[1].players) == 1
    assert result.timeline[1].players[0].foot.x > result.timeline[0].players[0].foot.x
    assert result.stats.optical_flow_updates == 1
    tracker.reset()
    assert tracker.advance(
        textured_player_frame(0),
        textured_player_frame(4),
        result.camera_motion[-1],
    ) == ()


def test_scene_cut_clears_spatial_tracks_until_next_vlm_anchor():
    config = cv_config(cuts=True)
    tracker = VLMInitializedPlayerCVTracker(
        config, HungarianPlayerTracker(config.tracking)
    )
    frames = [textured_player_frame(), np.full((100, 160), 255, dtype=np.uint8)]
    result = tracker.track_frames(frames, {0: detection(0)})
    assert result.stats.scene_cut_resets == 1
    assert result.timeline[1].players == ()


def test_cv_never_discovers_players_without_vlm_initialization():
    config = cv_config()
    tracker = VLMInitializedPlayerCVTracker(
        config, HungarianPlayerTracker(config.tracking)
    )
    result = tracker.track_frames(
        [textured_player_frame(0), textured_player_frame(4)], {}
    )
    assert all(frame.players == () for frame in result.timeline)


def test_confidence_decays_to_uncertain_and_recovers_at_anchor():
    settings = TrackConfidenceConfig(
        enabled=True,
        confirmed_threshold=0.7,
        lost_threshold=0.05,
        flow_decay=0.5,
    )
    config = replace(cv_config(), tracking=TrackingConfig(
        association_method="hungarian", confidence=settings
    ))
    tracker = VLMInitializedPlayerCVTracker(
        config, HungarianPlayerTracker(config.tracking)
    )
    result = tracker.track_frames(
        [textured_player_frame(0), textured_player_frame(1), textured_player_frame(2)],
        {0: detection(0), 2: detection(2, 0.2125)},
    )
    assert result.timeline[1].players[0].tracking_state == "uncertain"
    assert result.timeline[2].players[0].tracking_state == "confirmed"
    assert result.stats.uncertainty_recoveries >= 1
