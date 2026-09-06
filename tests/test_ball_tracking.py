import cv2
import numpy as np
import pytest

from track_game.ball_tracking import VLMInitializedBallTracker
from track_game.config import BallTrackingConfig
from track_game.schema import BallDetection, Box, Point


def ball(x, width=96, height=64):
    normalized_x = x / width
    normalized_y = 32 / height
    return BallDetection(
        Point(normalized_x, normalized_y),
        Box(
            (x - 5) / width,
            (32 - 5) / height,
            (x + 5) / width,
            (32 + 5) / height,
        ),
        0.95,
    )


def moving_ball_frames(count=8):
    frames = []
    for index in range(count):
        frame = np.zeros((64, 96), dtype=np.uint8)
        cv2.circle(frame, (20 + index * 2, 32), 5, 255, -1)
        cv2.line(frame, (20 + index * 2, 27), (20 + index * 2, 37), 80, 1)
        frames.append(frame)
    return frames


def test_optical_flow_tracks_only_after_vlm_initialization():
    tracker = VLMInitializedBallTracker(BallTrackingConfig(enabled=True))
    result = tracker.track_frames(moving_ball_frames(), {0: ball(20)})
    assert result.frames[0].source == "vlm"
    assert all(frame.position is not None for frame in result.frames)
    assert result.frames[-1].source == "optical_flow"
    assert result.frames[-1].position.x == pytest.approx(34 / 96, abs=0.03)
    assert result.stats.optical_flow_updates == 7


def test_ball_declares_lost_and_recovers_on_later_vlm_anchor():
    frames = moving_ball_frames(2) + [np.zeros((64, 96), dtype=np.uint8) for _ in range(8)]
    cv2.circle(frames[8], (50, 32), 5, 255, -1)
    cv2.circle(frames[9], (52, 32), 5, 255, -1)
    config = BallTrackingConfig(enabled=True, max_coast_frames=2)
    result = VLMInitializedBallTracker(config).track_frames(
        frames, {0: ball(20), 8: ball(50)}
    )
    assert any(frame.lost for frame in result.frames[3:8])
    assert result.frames[8].source == "vlm"
    assert result.stats.lost_events == 1
    assert result.stats.recovered_events == 1
