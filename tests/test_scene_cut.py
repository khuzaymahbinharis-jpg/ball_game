import numpy as np

from track_game.config import SceneCutConfig
from track_game.scene_cut import SceneCutDetector


def test_abrupt_nonsemantic_frame_change_is_a_cut():
    detector = SceneCutDetector(
        SceneCutConfig(
            enabled=True,
            downscale_width=64,
            threshold=0.3,
            minimum_frame_difference=0.1,
        )
    )
    result = detector.compare(
        np.zeros((48, 64), dtype=np.uint8),
        np.full((48, 64), 255, dtype=np.uint8),
        1,
    )
    assert result.is_cut
    assert result.frame_difference == 1.0
