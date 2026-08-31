from typing import Protocol

from PIL import Image

from .schema import FrameDetection


class VLMProvider(Protocol):
    def detect(self, frame_id: int, image: Image.Image) -> FrameDetection: ...


class MockVLMProvider:
    """Deterministic fixture-like provider. It does no image interpretation."""

    def detect(self, frame_id: int, image: Image.Image) -> FrameDetection:
        del image
        shift = min(frame_id * 0.005, 0.25)
        return FrameDetection.from_dict(
            {
                "frame_id": frame_id,
                "players": [
                    {
                        "detection_id": f"{frame_id}-a",
                        "team": "A",
                        "box": {
                            "left": 0.1 + shift,
                            "top": 0.3,
                            "right": 0.18 + shift,
                            "bottom": 0.75,
                        },
                        "confidence": 0.94,
                        "possesses_ball": True,
                    },
                    {
                        "detection_id": f"{frame_id}-b",
                        "team": "B",
                        "box": {
                            "left": 0.68 - shift,
                            "top": 0.28,
                            "right": 0.76 - shift,
                            "bottom": 0.72,
                        },
                        "confidence": 0.91,
                    },
                ],
                "ball": {"x": 0.2 + shift, "y": 0.72},
                "ball_confidence": 0.88,
            }
        )
