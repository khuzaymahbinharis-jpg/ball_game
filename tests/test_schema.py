import pytest
from track_game.schema import FrameDetection, frame_detection_json_schema


def valid_response():
    return {
        "frame_id": 2,
        "players": [
            {
                "detection_id": "p",
                "team": "A",
                "box": {"left": 0.1, "top": 0.2, "right": 0.3, "bottom": 0.8},
                "foot": {"x": 0.2, "y": 0.8},
                "confidence": 0.9,
                "team_confidence": 0.85,
            }
        ],
        "ball": {
            "center": {"x": 0.3, "y": 0.8},
            "box": {"left": 0.29, "top": 0.78, "right": 0.31, "bottom": 0.82},
            "confidence": 0.7,
        },
        "possession": {"player_detection_id": "p", "confidence": 0.65},
        "uncertainty_notes": [],
    }


def test_valid_response_is_parsed_strictly():
    result = FrameDetection.from_dict(valid_response())
    assert result.players[0].foot.x == pytest.approx(0.2)
    assert result.players[0].possesses_ball
    assert result.ball.x == pytest.approx(0.3)


def test_unknown_and_out_of_range_fields_are_rejected():
    response = valid_response()
    response["surprise"] = 1
    with pytest.raises(ValueError):
        FrameDetection.from_dict(response)
    response = valid_response()
    response["ball"]["center"]["x"] = 4
    with pytest.raises(ValueError):
        FrameDetection.from_dict(response)


def test_possession_must_reference_a_returned_player():
    response = valid_response()
    response["possession"]["player_detection_id"] = "missing"
    with pytest.raises(ValueError, match="reference"):
        FrameDetection.from_dict(response)


def test_json_schema_is_closed_and_requires_grounding_fields():
    schema = frame_detection_json_schema()
    assert schema["additionalProperties"] is False
    player = schema["properties"]["players"]["items"]
    assert {"box", "foot", "team_confidence"} <= set(player["required"])
