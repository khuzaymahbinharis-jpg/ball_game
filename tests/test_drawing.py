from PIL import Image

from track_game.drawing import annotation_overlay
from track_game.schema import Box, Point, Team
from track_game.tracking import TrackedFrame, TrackedPlayer


def test_player_marker_is_large_filled_and_antialiased():
    player = TrackedPlayer(
        1,
        Team.A,
        Box(0.4, 0.2, 0.6, 0.5),
        Point(0.5, 0.5),
        0.95,
    )
    frame = TrackedFrame(0, (player,), None, None)
    overlay = annotation_overlay(frame, (200, 100), (200, 100), show_ids=False)
    alpha = overlay.getchannel("A")
    bounds = alpha.getbbox()
    assert bounds is not None
    assert bounds[2] - bounds[0] >= 50
    assert 25 < alpha.getpixel((100, 51)) < 120
    assert any(0 < value < 255 for value in alpha.get_flattened_data())


def test_player_marker_grows_toward_near_edge_for_perspective():
    def marker_width(foot_y: float) -> int:
        player = TrackedPlayer(
            1,
            Team.A,
            Box(0.4, max(0.0, foot_y - 0.3), 0.6, foot_y),
            Point(0.5, foot_y),
            0.95,
        )
        frame = TrackedFrame(0, (player,), None, None)
        alpha = annotation_overlay(
            frame, (400, 200), (400, 200), show_ids=False
        ).getchannel("A")
        bounds = alpha.getbbox()
        assert bounds is not None
        return bounds[2] - bounds[0]

    assert marker_width(0.85) >= marker_width(0.25) + 20


def test_ball_marker_is_larger_and_filled():
    frame = TrackedFrame(0, (), Point(0.5, 0.5), 0.9)
    overlay = annotation_overlay(frame, (200, 100), (200, 100), show_ids=False)
    alpha = overlay.getchannel("A")
    bounds = alpha.getbbox()
    assert bounds is not None
    assert bounds[2] - bounds[0] >= 30
    assert alpha.getpixel((100, 50)) > 200
