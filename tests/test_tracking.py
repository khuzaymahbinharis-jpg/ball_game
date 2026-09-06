from dataclasses import replace

from track_game.config import HungarianConfig, TrackingConfig
from track_game.schema import FrameDetection, Team
from track_game.tracking import (
    HungarianPlayerTracker,
    NearestNeighbourTracker,
    _hungarian_minimize,
    interpolate_frames,
    TrackedPlayer,
)
from track_game.schema import Box, Point


def detection(frame, x, team="A"):
    return FrameDetection.from_dict(
        {
            "frame_id": frame,
            "players": [
                {
                    "detection_id": str(frame),
                    "team": team,
                    "box": {"left": x, "top": 0.2, "right": x + 0.1, "bottom": 0.8},
                    "foot": {"x": x + 0.05, "y": 0.8},
                    "confidence": 0.9,
                    "team_confidence": 0.9,
                }
            ],
            "ball": {
                "center": {"x": x, "y": 0.8},
                "box": None,
                "confidence": 0.8,
            },
            "possession": None,
            "uncertainty_notes": [],
        }
    )


def players_detection(frame, players):
    return FrameDetection.from_dict(
        {
            "frame_id": frame,
            "players": [
                {
                    "detection_id": f"{frame}-{index}",
                    "team": team,
                    "box": {
                        "left": x - 0.025,
                        "top": 0.2,
                        "right": x + 0.025,
                        "bottom": 0.8,
                    },
                    "foot": {"x": x, "y": 0.8},
                    "confidence": 0.9,
                    "team_confidence": 0.9,
                }
                for index, (x, team) in enumerate(players)
            ],
            "ball": None,
            "possession": None,
            "uncertainty_notes": [],
        }
    )


def test_nearby_same_team_detection_keeps_identity():
    tracker = NearestNeighbourTracker(max_distance=0.2)
    first = tracker.update(detection(0, 0.1))
    second = tracker.update(detection(10, 0.15))
    assert first.players[0].track_id == second.players[0].track_id


def test_team_constraint_prevents_bad_match():
    tracker = NearestNeighbourTracker(max_distance=0.2, team_constraint=True)
    first = tracker.update(detection(0, 0.1, "A"))
    second = tracker.update(detection(1, 0.1, "B"))
    assert first.players[0].track_id != second.players[0].track_id


def test_interpolation_preserves_id_and_midpoint():
    tracker = NearestNeighbourTracker(max_distance=0.5)
    left = tracker.update(detection(0, 0.1))
    right = tracker.update(detection(2, 0.3))
    middle = interpolate_frames(left, right)[1]
    assert middle.players[0].track_id == left.players[0].track_id
    assert middle.players[0].box.left == 0.2


def test_interpolation_preserves_complete_anchor_detections():
    tracker = NearestNeighbourTracker(max_distance=0.01)
    left = tracker.update(detection(0, 0.1))
    right = tracker.update(detection(2, 0.5))
    frames = interpolate_frames(left, right)
    assert frames[0] == left
    assert frames[-1] == right
    assert len(frames[0].players) == len(frames[-1].players) == 1
    assert frames[0].players[0].track_id != frames[-1].players[0].track_id


def test_hungarian_solver_finds_global_minimum():
    pairs = _hungarian_minimize([[10.0, 2.0, 9.0], [7.0, 5.0, 6.0]])
    assert set(pairs) == {(0, 1), (1, 2)}


def test_hungarian_global_assignment_avoids_greedy_fragmentation():
    first = players_detection(0, [(0.10, "A"), (0.21, "A")])
    second = players_detection(15, [(0.20, "A"), (0.31, "A")])
    greedy = NearestNeighbourTracker(max_distance=0.2)
    greedy.update(first)
    greedy_ids = {player.track_id for player in greedy.update(second).players}

    cost = HungarianConfig(
        foot_distance_weight=1.0,
        iou_weight=0.0,
        size_change_weight=0.0,
        motion_weight=0.0,
        max_assignment_cost=1.1,
    )
    config = TrackingConfig(
        max_normalized_distance=0.2,
        association_method="hungarian",
        hungarian=cost,
    )
    hungarian = HungarianPlayerTracker(config)
    original_ids = {player.track_id for player in hungarian.update(first).players}
    new_ids = {player.track_id for player in hungarian.update(second).players}

    assert len(greedy_ids - {1, 2}) == 1
    assert new_ids == original_ids


def test_hungarian_team_mismatch_is_soft_and_label_is_stable():
    tracker = HungarianPlayerTracker(
        replace(TrackingConfig(), association_method="hungarian")
    )
    first = tracker.update(detection(0, 0.1, "A"))
    second = tracker.update(detection(15, 0.1, "B"))
    assert second.players[0].track_id == first.players[0].track_id
    assert second.players[0].team == Team.A


def test_lost_hungarian_track_recovers_then_expires_after_limit():
    tracker = HungarianPlayerTracker(
        TrackingConfig(association_method="hungarian", max_missed_anchors=2)
    )
    first = tracker.update(detection(0, 0.1))
    tracker.update(players_detection(15, []))
    assert tracker.track_states[first.players[0].track_id] == "lost"
    recovered = tracker.update(detection(30, 0.1))
    assert recovered.players[0].track_id == first.players[0].track_id
    assert tracker.track_states[first.players[0].track_id] == "active"
    assert tracker.lifecycle.tracks_lost == 1
    assert tracker.lifecycle.tracks_recovered == 1

    tracker.update(players_detection(45, []))
    tracker.update(players_detection(60, []))
    tracker.update(players_detection(75, []))
    replacement = tracker.update(detection(90, 0.1))
    assert replacement.players[0].track_id != first.players[0].track_id
    assert tracker.lifecycle.tracks_expired == 1


def test_hungarian_uses_cv_prediction_at_next_anchor():
    config = TrackingConfig(
        max_normalized_distance=0.25,
        association_method="hungarian",
        hungarian=HungarianConfig(
            foot_distance_weight=1.0,
            iou_weight=0.0,
            size_change_weight=0.0,
            motion_weight=0.0,
            team_mismatch_penalty=0.0,
            max_assignment_cost=1.1,
        ),
    )
    tracker = HungarianPlayerTracker(config)
    first = tracker.update(players_detection(0, [(0.2, "A"), (0.8, "A")]))
    predictions = tuple(
        TrackedPlayer(
            player.track_id,
            player.team,
            Box(0.72 if player.foot.x < 0.5 else 0.18, 0.2, 0.78 if player.foot.x < 0.5 else 0.24, 0.8),
            Point(0.75 if player.foot.x < 0.5 else 0.21, 0.8),
            player.confidence,
        )
        for player in first.players
    )
    tracker.apply_predictions(predictions, 15)
    second = tracker.update(players_detection(15, [(0.21, "A"), (0.75, "A")]))
    by_x = {round(player.foot.x, 2): player.track_id for player in second.players}
    assert by_x[0.75] == first.players[0].track_id
    assert by_x[0.21] == first.players[1].track_id
