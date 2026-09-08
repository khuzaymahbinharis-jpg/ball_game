"""VLM-grounded filtering for broadcast views that should not be tracked.

The VLM remains responsible for deciding what image regions are players.  This
module only applies an explainable geometric sanity check to those returned
boxes so a tight portrait/close-up is not mistaken for a playable court view.
"""

from dataclasses import dataclass, replace

from .config import ShotContextConfig
from .schema import BallDetection, FrameDetection


@dataclass(frozen=True)
class ShotContextDecision:
    frame_id: int
    trackable: bool
    reason: str
    vlm_player_count: int
    dominant_player_box_area: float


def _box_area(detection: object) -> float:
    box = detection.box
    return (box.right - box.left) * (box.bottom - box.top)


class VLMShotContextFilter:
    """Suppress detections when VLM boxes describe a tight broadcast close-up.

    This is deliberately not a classical semantic detector: it consumes only
    VLM-localized player geometry.  Empty and ordinary gameplay detections pass
    through unchanged.
    """

    def __init__(self, config: ShotContextConfig | None = None):
        self.config = config or ShotContextConfig()

    def classify(self, detection: FrameDetection) -> ShotContextDecision:
        areas = sorted((_box_area(player) for player in detection.players), reverse=True)
        dominant_area = areas[0] if areas else 0.0
        is_closeup = self.config.enabled and (
            0 < len(areas) <= self.config.maximum_players_in_closeup
            and dominant_area >= self.config.minimum_dominant_box_area
        )
        return ShotContextDecision(
            frame_id=detection.frame_id,
            trackable=not is_closeup,
            reason="dominant_vlm_player_closeup" if is_closeup else "trackable_or_empty_view",
            vlm_player_count=len(areas),
            dominant_player_box_area=dominant_area,
        )

    def filter(
        self, detection: FrameDetection
    ) -> tuple[FrameDetection, ShotContextDecision]:
        decision = self.classify(detection)
        if decision.trackable:
            return detection, decision
        note = (
            "tracking suppressed: VLM-localized geometry indicates a broadcast "
            "player close-up"
        )
        return (
            replace(
                detection,
                players=(),
                ball_detection=None,
                possession=None,
                uncertainty_notes=(*detection.uncertainty_notes, note),
            ),
            decision,
        )


def trackable_ball_anchors(
    detections: dict[int, FrameDetection],
    decisions: tuple[ShotContextDecision, ...],
) -> dict[int, BallDetection | None]:
    """Remove ball anchors on shots already classified as non-trackable."""

    suppressed = {item.frame_id for item in decisions if not item.trackable}
    return {
        frame_id: None if frame_id in suppressed else detection.ball_detection
        for frame_id, detection in detections.items()
    }
