from dataclasses import dataclass
from math import hypot

from .schema import Box, FrameDetection, Point, Team


@dataclass(frozen=True)
class TrackedPlayer:
    track_id: int
    team: Team
    box: Box
    confidence: float
    possesses_ball: bool = False


@dataclass(frozen=True)
class TrackedFrame:
    frame_id: int
    players: tuple[TrackedPlayer, ...]
    ball: Point | None
    ball_confidence: float | None


class NearestNeighbourTracker:
    """Greedy association using foot position and optional hard team constraint."""

    def __init__(
        self,
        max_distance: float = 0.2,
        team_constraint: bool = True,
        max_missed: int = 2,
    ):
        self.max_distance, self.team_constraint, self.max_missed = (
            max_distance,
            team_constraint,
            max_missed,
        )
        self._next_id = 1
        self._tracks: dict[int, tuple[TrackedPlayer, int]] = {}

    def update(self, detection: FrameDetection) -> TrackedFrame:
        candidates: list[tuple[float, int, int]] = []
        for index, player in enumerate(detection.players):
            for track_id, (old, missed) in self._tracks.items():
                if missed > self.max_missed or (
                    self.team_constraint and old.team != player.team
                ):
                    continue
                distance = hypot(
                    old.box.foot.x - player.box.foot.x,
                    old.box.foot.y - player.box.foot.y,
                )
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, index))
        assigned_tracks, assigned_detections, matches = set(), set(), {}
        for _, track_id, index in sorted(candidates):
            if track_id not in assigned_tracks and index not in assigned_detections:
                matches[index] = track_id
                assigned_tracks.add(track_id)
                assigned_detections.add(index)

        current: list[TrackedPlayer] = []
        new_tracks: dict[int, tuple[TrackedPlayer, int]] = {}
        for index, player in enumerate(detection.players):
            matched_id = matches.get(index)
            if matched_id is None:
                matched_id, self._next_id = self._next_id, self._next_id + 1
            tracked = TrackedPlayer(
                matched_id,
                player.team,
                player.box,
                player.confidence,
                player.possesses_ball,
            )
            current.append(tracked)
            new_tracks[matched_id] = (tracked, 0)
        for track_id, (old, missed) in self._tracks.items():
            if track_id not in new_tracks and missed < self.max_missed:
                new_tracks[track_id] = (old, missed + 1)
        self._tracks = new_tracks
        return TrackedFrame(
            detection.frame_id,
            tuple(current),
            detection.ball,
            detection.ball_confidence,
        )


def _lerp(a: float, b: float, ratio: float) -> float:
    return a + (b - a) * ratio


def interpolate_frames(left: TrackedFrame, right: TrackedFrame) -> list[TrackedFrame]:
    """Interpolate tracks shared by two anchors; endpoint frames are included."""
    if right.frame_id <= left.frame_id:
        raise ValueError("anchors must be in increasing frame order")
    left_by_id, right_by_id = (
        {p.track_id: p for p in f.players} for f in (left, right)
    )
    frames = []
    span = right.frame_id - left.frame_id
    for frame_id in range(left.frame_id, right.frame_id + 1):
        ratio = (frame_id - left.frame_id) / span
        players = []
        for track_id in sorted(left_by_id.keys() & right_by_id.keys()):
            a, b = left_by_id[track_id], right_by_id[track_id]
            box = Box(
                *(
                    _lerp(x, y, ratio)
                    for x, y in zip(
                        (a.box.left, a.box.top, a.box.right, a.box.bottom),
                        (b.box.left, b.box.top, b.box.right, b.box.bottom),
                    )
                )
            )
            players.append(
                TrackedPlayer(
                    track_id,
                    a.team,
                    box,
                    _lerp(a.confidence, b.confidence, ratio),
                    a.possesses_ball if ratio < 0.5 else b.possesses_ball,
                )
            )
        ball = None
        if left.ball is not None and right.ball is not None:
            ball = Point(
                _lerp(left.ball.x, right.ball.x, ratio),
                _lerp(left.ball.y, right.ball.y, ratio),
            )
        confidence = None
        if left.ball_confidence is not None and right.ball_confidence is not None:
            confidence = _lerp(left.ball_confidence, right.ball_confidence, ratio)
        frames.append(TrackedFrame(frame_id, tuple(players), ball, confidence))
    return frames


def interpolate_sequence(anchors: list[TrackedFrame]) -> list[TrackedFrame]:
    output: list[TrackedFrame] = []
    for index in range(len(anchors) - 1):
        segment = interpolate_frames(anchors[index], anchors[index + 1])
        output.extend(segment if index == 0 else segment[1:])
    return anchors[:] if len(anchors) < 2 else output
