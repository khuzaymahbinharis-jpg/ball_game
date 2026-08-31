from track_game.schema import FrameDetection
from track_game.tracking import NearestNeighbourTracker, interpolate_frames


def detection(frame, x, team="A"):
    return FrameDetection.from_dict(
        {
            "frame_id": frame,
            "players": [
                {
                    "detection_id": str(frame),
                    "team": team,
                    "box": {"left": x, "top": 0.2, "right": x + 0.1, "bottom": 0.8},
                    "confidence": 0.9,
                }
            ],
            "ball": {"x": x, "y": 0.8},
            "ball_confidence": 0.8,
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
