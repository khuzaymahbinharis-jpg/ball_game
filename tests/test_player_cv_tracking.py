from dataclasses import replace

import cv2
import numpy as np
import pytest

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


def empty_detection(frame_id):
    return FrameDetection.from_dict(
        {
            "frame_id": frame_id,
            "players": [],
            "ball": None,
            "possession": None,
            "uncertainty_notes": ["player temporarily omitted"],
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


def test_closeup_anchor_clears_retained_tracks_instead_of_coasting_them():
    config = cv_config()
    tracker = VLMInitializedPlayerCVTracker(
        config, HungarianPlayerTracker(config.tracking)
    )
    closeup = detection(1, left=0.05)
    closeup_player = closeup.players[0]
    closeup = replace(
        closeup,
        players=(
            replace(
                closeup_player,
                box=replace(closeup_player.box, right=0.95, bottom=0.95),
                foot=replace(closeup_player.foot, x=0.5, y=0.95),
            ),
        ),
    )
    result = tracker.track_frames(
        [textured_player_frame(0), textured_player_frame(1)],
        {0: detection(0), 1: closeup},
    )
    assert result.timeline[0].players
    assert result.timeline[1].players == ()
    assert result.stats.closeup_suppressed_anchors == 1
    assert not result.shot_context[-1].trackable


def test_clipped_box_at_bottom_right_retains_positive_area():
    box, foot = VLMInitializedPlayerCVTracker._clip_geometry(
        np.array([160.0, 100.0, 170.0, 110.0]),
        np.array([170.0, 110.0]),
        width=160,
        height=100,
    )
    assert box[2] > box[0]
    assert box[3] > box[1]
    assert tuple(foot) == (159.0, 99.0)


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


def test_single_anchor_omission_coasts_without_marker_blink_and_recovers_id():
    config = cv_config()
    tracker = VLMInitializedPlayerCVTracker(
        config, HungarianPlayerTracker(config.tracking)
    )
    result = tracker.track_frames(
        [textured_player_frame()] + [np.zeros((100, 160), dtype=np.uint8) for _ in range(12)],
        {
            0: detection(0),
            6: empty_detection(6),
            12: detection(12),
        },
    )

    visible_ids = [tuple(player.track_id for player in frame.players) for frame in result.timeline]
    assert visible_ids == [(1,)] * 13
    assert result.stats.coast_updates >= 6
    assert tracker.logical_tracker.lifecycle.tracks_lost == 1
    assert tracker.logical_tracker.lifecycle.tracks_recovered == 1


def test_anchor_geometry_eases_from_prediction_instead_of_snapping():
    config = cv_config()
    tracker = VLMInitializedPlayerCVTracker(
        config, HungarianPlayerTracker(config.tracking)
    )
    result = tracker.track_frames(
        [textured_player_frame() for _ in range(3)],
        {0: detection(0, left=0.20), 2: detection(2, left=0.30)},
    )

    predicted_x = result.timeline[1].players[0].foot.x
    anchor_x = result.timeline[2].players[0].foot.x
    raw_vlm_x = 0.40
    expected = predicted_x + tracker._ANCHOR_GEOMETRY_WEIGHT * (
        raw_vlm_x - predicted_x
    )
    assert anchor_x == pytest.approx(expected, abs=1e-6)
    assert abs(anchor_x - predicted_x) < abs(raw_vlm_x - predicted_x)


def test_continuity_horizon_uses_normal_cadence_not_short_final_gap():
    config = cv_config()
    tracker = VLMInitializedPlayerCVTracker(
        config, HungarianPlayerTracker(config.tracking)
    )
    tracker.track_frames(
        [textured_player_frame() for _ in range(18)],
        {
            0: detection(0),
            6: detection(6),
            12: detection(12),
            17: detection(17),
        },
    )

    assert tracker._continuity_horizon_frames == 18
