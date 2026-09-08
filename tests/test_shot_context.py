from track_game.config import PipelineConfig, ShotContextConfig
from track_game.schema import FrameDetection
from track_game.shot_context import VLMShotContextFilter, trackable_ball_anchors


def detection(*, frame_id=0, boxes=()):
    players = []
    for index, (left, top, right, bottom) in enumerate(boxes):
        players.append(
            {
                "detection_id": f"p{index}",
                "team": "A" if index % 2 == 0 else "B",
                "box": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
                "foot": {"x": (left + right) / 2, "y": bottom},
                "confidence": 0.9,
                "team_confidence": 0.9,
            }
        )
    return FrameDetection.from_dict(
        {
            "frame_id": frame_id,
            "players": players,
            "ball": (
                None
                if not players
                else {
                    "center": {"x": 0.5, "y": 0.5},
                    "box": None,
                    "confidence": 0.8,
                }
            ),
            "possession": None,
            "uncertainty_notes": [],
        }
    )


def test_dominant_vlm_box_marks_closeup_and_removes_all_markers():
    original = detection(boxes=((0.10, 0.05, 0.85, 0.90), (0.80, 0.20, 0.95, 0.65)))
    filtered, decision = VLMShotContextFilter().filter(original)
    assert not decision.trackable
    assert decision.reason == "dominant_vlm_player_closeup"
    assert decision.dominant_player_box_area > 0.6
    assert filtered.players == ()
    assert filtered.ball_detection is None
    assert filtered.possession is None
    assert "close-up" in filtered.uncertainty_notes[-1]


def test_wide_gameplay_with_many_small_vlm_boxes_remains_trackable():
    boxes = tuple(
        (0.04 + index * 0.08, 0.35, 0.08 + index * 0.08, 0.70)
        for index in range(8)
    )
    original = detection(frame_id=18, boxes=boxes)
    filtered, decision = VLMShotContextFilter().filter(original)
    assert decision.trackable
    assert filtered is original


def test_empty_vlm_detection_is_not_reclassified_semantically():
    original = detection(boxes=())
    filtered, decision = VLMShotContextFilter().filter(original)
    assert decision.trackable
    assert filtered is original


def test_thresholds_are_configurable_without_classical_semantics():
    original = detection(boxes=((0.1, 0.1, 0.3, 0.7),))
    default_decision = VLMShotContextFilter().classify(original)
    sensitive = VLMShotContextFilter(
        ShotContextConfig(minimum_dominant_box_area=0.10)
    ).classify(original)
    assert default_decision.trackable
    assert not sensitive.trackable


def test_shot_context_thresholds_are_serialized_with_pipeline_config():
    serialized = PipelineConfig(
        shot_context=ShotContextConfig(minimum_dominant_box_area=0.2)
    ).as_dict()
    assert serialized["shot_context"] == {
        "enabled": True,
        "maximum_players_in_closeup": 3,
        "minimum_dominant_box_area": 0.2,
    }


def test_disabled_filter_preserves_large_vlm_detection():
    original = detection(boxes=((0.05, 0.05, 0.95, 0.95),))
    filtered, decision = VLMShotContextFilter(
        ShotContextConfig(enabled=False)
    ).filter(original)
    assert decision.trackable
    assert filtered is original


def test_suppressed_shot_cannot_reinitialize_ball_tracker():
    closeup = detection(frame_id=0, boxes=((0.05, 0.05, 0.95, 0.95),))
    wide = detection(
        frame_id=6,
        boxes=tuple(
            (0.05 + index * 0.08, 0.4, 0.08 + index * 0.08, 0.7)
            for index in range(6)
        ),
    )
    filter_ = VLMShotContextFilter()
    decisions = tuple(filter_.classify(item) for item in (closeup, wide))
    anchors = trackable_ball_anchors({0: closeup, 6: wide}, decisions)
    assert anchors[0] is None
    assert anchors[6] is not None
