import pytest

from track_game.schema import Box, Point, Team
from track_game.tracking import TrackedFrame, TrackedPlayer
from track_game.trajectory_quality import player_trajectory_quality


def _frame(frame_id, x):
    player = TrackedPlayer(
        1,
        Team.A,
        Box(x - 0.02, 0.2, x + 0.02, 0.5),
        Point(x, 0.5),
        0.9,
    )
    return TrackedFrame(frame_id, (player,), None, None)


def test_trajectory_quality_reports_anchor_steps_and_constant_motion():
    metrics = player_trajectory_quality(
        [_frame(0, 0.1), _frame(1, 0.2), _frame(2, 0.3), _frame(3, 0.4)],
        anchor_frames={2},
    )

    assert metrics["consecutive_steps"] == 3
    assert metrics["anchor_steps"] == 1
    assert metrics["median_step"] == pytest.approx(0.1)
    assert metrics["median_acceleration"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["median_anchor_step"] == pytest.approx(0.1)


def test_trajectory_quality_excludes_scene_cut_step():
    metrics = player_trajectory_quality(
        [_frame(0, 0.1), _frame(1, 0.2), _frame(2, 0.9)],
        scene_cut_frames={2},
    )

    assert metrics["consecutive_steps"] == 1
    assert metrics["p95_step"] == pytest.approx(0.1)
