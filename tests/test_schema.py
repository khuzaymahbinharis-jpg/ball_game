import pytest
from track_game.schema import FrameDetection


def valid_response():
    return {
        "frame_id": 2,
        "players": [
            {
                "detection_id": "p",
                "team": "A",
                "box": {"left": 0.1, "top": 0.2, "right": 0.3, "bottom": 0.8},
                "confidence": 0.9,
                "possesses_ball": True,
            }
        ],
        "ball": {"x": 0.3, "y": 0.8},
        "ball_confidence": 0.7,
    }


def test_valid_response_is_parsed_strictly():
    result = FrameDetection.from_dict(valid_response())
    assert result.players[0].box.foot.x == pytest.approx(0.2)


def test_unknown_and_out_of_range_fields_are_rejected():
    response = valid_response()
    response["surprise"] = 1
    with pytest.raises(ValueError):
        FrameDetection.from_dict(response)
    response = valid_response()
    response["ball"]["x"] = 4
    with pytest.raises(ValueError):
        FrameDetection.from_dict(response)
