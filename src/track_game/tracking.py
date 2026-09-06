from dataclasses import dataclass, replace
from math import hypot, log
from typing import Literal, Protocol

from .config import TrackingConfig
from .schema import Box, FrameDetection, PlayerDetection, Point, Team


@dataclass(frozen=True)
class TrackedPlayer:
    track_id: int
    team: Team
    box: Box
    foot: Point
    confidence: float
    possesses_ball: bool = False
    tracking_confidence: float = 1.0
    tracking_state: Literal["confirmed", "uncertain", "lost"] = "confirmed"


@dataclass(frozen=True)
class TrackedFrame:
    frame_id: int
    players: tuple[TrackedPlayer, ...]
    ball: Point | None
    ball_confidence: float | None


@dataclass(frozen=True)
class TrackLifecycleSnapshot:
    tracks_created: int
    tracks_expired: int
    tracks_lost: int
    tracks_recovered: int
    matches: int
    active_tracks: int
    currently_lost_tracks: int


class PlayerTracker(Protocol):
    @property
    def lifecycle(self) -> TrackLifecycleSnapshot: ...

    @property
    def track_states(self) -> dict[int, str]: ...

    def update(self, detection: FrameDetection) -> TrackedFrame: ...

    def apply_predictions(self, players: tuple[TrackedPlayer, ...], frame_id: int) -> None: ...

    def mark_scene_cut(self) -> None: ...


class _LifecycleCounter:
    def __init__(self) -> None:
        self.created = 0
        self.expired = 0
        self.lost = 0
        self.recovered = 0
        self.matches = 0

    def snapshot(self, missed_counts: list[int]) -> TrackLifecycleSnapshot:
        return TrackLifecycleSnapshot(
            tracks_created=self.created,
            tracks_expired=self.expired,
            tracks_lost=self.lost,
            tracks_recovered=self.recovered,
            matches=self.matches,
            active_tracks=sum(missed == 0 for missed in missed_counts),
            currently_lost_tracks=sum(missed > 0 for missed in missed_counts),
        )


class NearestNeighbourTracker:
    """Greedy baseline using foot distance and an optional hard team constraint."""

    def __init__(
        self,
        max_distance: float = 0.2,
        team_constraint: bool = True,
        max_missed: int = 2,
    ):
        if max_distance <= 0 or max_missed < 0:
            raise ValueError("invalid greedy tracker settings")
        self.max_distance, self.team_constraint, self.max_missed = (
            max_distance,
            team_constraint,
            max_missed,
        )
        self._next_id = 1
        self._tracks: dict[int, tuple[TrackedPlayer, int]] = {}
        self._counter = _LifecycleCounter()
        self._spatial_reset_pending = False

    @property
    def lifecycle(self) -> TrackLifecycleSnapshot:
        return self._counter.snapshot([missed for _, missed in self._tracks.values()])

    @property
    def track_states(self) -> dict[int, str]:
        return {
            track_id: "active" if missed == 0 else "lost"
            for track_id, (_, missed) in self._tracks.items()
        }

    def update(self, detection: FrameDetection) -> TrackedFrame:
        candidates: list[tuple[float, int, int]] = []
        if not self._spatial_reset_pending:
            for index, player in enumerate(detection.players):
                for track_id, (old, missed) in self._tracks.items():
                    if missed > self.max_missed or (
                        self.team_constraint and old.team != player.team
                    ):
                        continue
                    distance = hypot(
                        old.foot.x - player.foot.x,
                        old.foot.y - player.foot.y,
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
                self._counter.created += 1
            else:
                self._counter.matches += 1
                if self._tracks[matched_id][1] > 0:
                    self._counter.recovered += 1
            tracked = TrackedPlayer(
                matched_id,
                player.team,
                player.box,
                player.foot,
                player.confidence,
                player.possesses_ball,
            )
            current.append(tracked)
            new_tracks[matched_id] = (tracked, 0)
        for track_id, (old, missed) in self._tracks.items():
            if track_id in new_tracks:
                continue
            new_missed = missed + 1
            if missed == 0:
                self._counter.lost += 1
            if new_missed > self.max_missed:
                self._counter.expired += 1
            else:
                new_tracks[track_id] = (old, new_missed)
        self._tracks = new_tracks
        self._spatial_reset_pending = False
        return TrackedFrame(
            detection.frame_id,
            tuple(current),
            detection.ball,
            detection.ball_confidence,
        )

    def apply_predictions(self, players: tuple[TrackedPlayer, ...], frame_id: int) -> None:
        del frame_id
        for player in players:
            if player.track_id in self._tracks:
                _, missed = self._tracks[player.track_id]
                self._tracks[player.track_id] = (player, missed)

    def mark_scene_cut(self) -> None:
        self._spatial_reset_pending = True


@dataclass
class _HungarianTrack:
    player: TrackedPlayer
    last_seen_frame: int
    missed_anchors: int
    velocity: Point
    team_scores: dict[Team, float]
    predicted_frame_id: int | None = None


def _box_iou(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left.right, right.right) - max(left.left, right.left))
    intersection_height = max(0.0, min(left.bottom, right.bottom) - max(left.top, right.top))
    intersection = intersection_width * intersection_height
    left_area = (left.right - left.left) * (left.bottom - left.top)
    right_area = (right.right - right.left) * (right.bottom - right.top)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _size_change(left: Box, right: Box) -> float:
    left_area = (left.right - left.left) * (left.bottom - left.top)
    right_area = (right.right - right.left) * (right.bottom - right.top)
    return min(abs(log(right_area / left_area)), 2.0) / 2.0


def _hungarian_minimize(costs: list[list[float]]) -> list[tuple[int, int]]:
    """Return the minimum-cost rectangular assignment using Hungarian potentials."""

    if not costs or not costs[0]:
        return []
    row_count, column_count = len(costs), len(costs[0])
    if any(len(row) != column_count for row in costs):
        raise ValueError("cost matrix must be rectangular")
    transposed = row_count > column_count
    matrix = (
        [[costs[row][column] for row in range(row_count)] for column in range(column_count)]
        if transposed
        else costs
    )
    n, m = len(matrix), len(matrix[0])
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        p[0] = row
        min_values = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        column = 0
        while True:
            used[column] = True
            current_row = p[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, m + 1):
                if used[candidate]:
                    continue
                current = matrix[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if current < min_values[candidate]:
                    min_values[candidate] = current
                    way[candidate] = column
                if min_values[candidate] < delta:
                    delta, next_column = min_values[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_values[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    pairs = [(p[column] - 1, column - 1) for column in range(1, m + 1) if p[column]]
    return [(column, row) for row, column in pairs] if transposed else pairs


class HungarianPlayerTracker:
    """Global assignment with soft team evidence and lost-track reassociation."""

    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or replace(TrackingConfig(), association_method="hungarian")
        self._next_id = 1
        self._tracks: dict[int, _HungarianTrack] = {}
        self._counter = _LifecycleCounter()
        self._spatial_reset_pending = False
        self._last_assignment_costs: dict[int, float] = {}

    @property
    def lifecycle(self) -> TrackLifecycleSnapshot:
        return self._counter.snapshot(
            [track.missed_anchors for track in self._tracks.values()]
        )

    @property
    def track_states(self) -> dict[int, str]:
        return {
            track_id: "active" if state.missed_anchors == 0 else "lost"
            for track_id, state in self._tracks.items()
        }

    @property
    def last_assignment_costs(self) -> dict[int, float]:
        return dict(self._last_assignment_costs)

    def _confidence_state(self, confidence: float) -> Literal["confirmed", "uncertain", "lost"]:
        settings = self.config.confidence
        if not settings.enabled or confidence >= settings.confirmed_threshold:
            return "confirmed"
        return "uncertain" if confidence >= settings.lost_threshold else "lost"

    def _team_penalty(self, old: Team, new: Team) -> float:
        config = self.config.hungarian
        if old == Team.UNCERTAIN or new == Team.UNCERTAIN:
            return config.uncertain_team_penalty
        return 0.0 if old == new else config.team_mismatch_penalty

    def _association_cost(
        self, state: _HungarianTrack, detection: PlayerDetection, frame_id: int
    ) -> float:
        config = self.config.hungarian
        scale = self.config.max_normalized_distance
        elapsed = max(1, frame_id - state.last_seen_frame)
        foot_distance = hypot(
            state.player.foot.x - detection.foot.x,
            state.player.foot.y - detection.foot.y,
        ) / scale
        predicted = (
            state.player.foot
            if state.predicted_frame_id == frame_id
            else Point(
                state.player.foot.x + state.velocity.x * elapsed,
                state.player.foot.y + state.velocity.y * elapsed,
            )
        )
        motion_distance = hypot(
            predicted.x - detection.foot.x,
            predicted.y - detection.foot.y,
        ) / scale
        return (
            config.foot_distance_weight * foot_distance
            + config.iou_weight * (1.0 - _box_iou(state.player.box, detection.box))
            + config.size_change_weight * _size_change(state.player.box, detection.box)
            + config.motion_weight * motion_distance
            + self._team_penalty(state.player.team, detection.team)
        )

    def _stabilized_team(
        self, state: _HungarianTrack, detection: PlayerDetection
    ) -> Team:
        config = self.config.hungarian
        for team in (Team.A, Team.B):
            state.team_scores[team] *= config.team_score_decay
        if detection.team in (Team.A, Team.B):
            state.team_scores[detection.team] += detection.team_confidence
        current = state.player.team
        best = max((Team.A, Team.B), key=lambda team: state.team_scores[team])
        if current == Team.UNCERTAIN:
            return best if state.team_scores[best] > 0 else current
        if best != current and (
            state.team_scores[best]
            > state.team_scores[current] * config.team_switch_margin
        ):
            return best
        return current

    def _new_state(self, detection: PlayerDetection, frame_id: int) -> _HungarianTrack:
        track_id, self._next_id = self._next_id, self._next_id + 1
        self._counter.created += 1
        scores = {Team.A: 0.0, Team.B: 0.0}
        if detection.team in scores:
            scores[detection.team] = detection.team_confidence
        player = TrackedPlayer(
            track_id,
            detection.team,
            detection.box,
            detection.foot,
            detection.confidence,
            detection.possesses_ball,
            detection.confidence,
            self._confidence_state(detection.confidence),
        )
        return _HungarianTrack(player, frame_id, 0, Point(0.0, 0.0), scores)

    def update(self, detection: FrameDetection) -> TrackedFrame:
        track_ids = sorted(self._tracks)
        detections = list(detection.players)
        matches: dict[int, int] = {}
        self._last_assignment_costs = {}
        if track_ids and detections and not self._spatial_reset_pending:
            real_costs = [
                [
                    self._association_cost(self._tracks[track_id], item, detection.frame_id)
                    for item in detections
                ]
                for track_id in track_ids
            ]
            dummy_cost = self.config.hungarian.max_assignment_cost + 1e-6
            matrix = [row + [dummy_cost] * len(track_ids) for row in real_costs]
            for track_index, detection_index in _hungarian_minimize(matrix):
                if (
                    detection_index < len(detections)
                    and real_costs[track_index][detection_index]
                    <= self.config.hungarian.max_assignment_cost
                ):
                    matches[detection_index] = track_ids[track_index]

        current: list[TrackedPlayer] = []
        matched_track_ids = set(matches.values())
        for detection_index, item in enumerate(detections):
            track_id = matches.get(detection_index)
            if track_id is None:
                state = self._new_state(item, detection.frame_id)
                self._tracks[state.player.track_id] = state
                current.append(state.player)
                continue
            state = self._tracks[track_id]
            assignment_cost = real_costs[track_ids.index(track_id)][detection_index]
            self._last_assignment_costs[track_id] = assignment_cost
            self._counter.matches += 1
            if state.missed_anchors > 0:
                self._counter.recovered += 1
            elapsed = max(1, detection.frame_id - state.last_seen_frame)
            measured_velocity = Point(
                (item.foot.x - state.player.foot.x) / elapsed,
                (item.foot.y - state.player.foot.y) / elapsed,
            )
            smoothing = self.config.hungarian.velocity_smoothing
            state.velocity = Point(
                smoothing * state.velocity.x + (1 - smoothing) * measured_velocity.x,
                smoothing * state.velocity.y + (1 - smoothing) * measured_velocity.y,
            )
            stable_team = self._stabilized_team(state, item)
            if self.config.confidence.enabled:
                settings = self.config.confidence
                assignment_quality = max(
                    0.0,
                    1.0 - assignment_cost / self.config.hungarian.max_assignment_cost,
                )
                total_weight = settings.anchor_detection_weight + settings.assignment_weight
                tracking_confidence = (
                    settings.anchor_detection_weight * item.confidence
                    + settings.assignment_weight * assignment_quality
                ) / total_weight
            else:
                tracking_confidence = item.confidence
            state.player = TrackedPlayer(
                track_id,
                stable_team,
                item.box,
                item.foot,
                item.confidence,
                item.possesses_ball,
                tracking_confidence,
                self._confidence_state(tracking_confidence),
            )
            state.last_seen_frame = detection.frame_id
            state.missed_anchors = 0
            state.predicted_frame_id = None
            current.append(state.player)

        for track_id in list(track_ids):
            if track_id in matched_track_ids:
                continue
            state = self._tracks[track_id]
            if state.missed_anchors == 0:
                self._counter.lost += 1
            state.missed_anchors += 1
            if self.config.confidence.enabled:
                confidence = (
                    state.player.tracking_confidence
                    * self.config.confidence.missed_anchor_decay
                )
                state.player = replace(
                    state.player,
                    tracking_confidence=confidence,
                    tracking_state=self._confidence_state(confidence),
                )
            if state.missed_anchors > self.config.max_missed_anchors:
                self._counter.expired += 1
                del self._tracks[track_id]

        self._spatial_reset_pending = False
        return TrackedFrame(
            detection.frame_id,
            tuple(current),
            detection.ball,
            detection.ball_confidence,
        )

    def apply_predictions(self, players: tuple[TrackedPlayer, ...], frame_id: int) -> None:
        for player in players:
            state = self._tracks.get(player.track_id)
            if state is None:
                continue
            state.player = player
            state.predicted_frame_id = frame_id

    def mark_scene_cut(self) -> None:
        """Reset only spatial correspondence; logical tracks age normally at the next anchor."""

        self._spatial_reset_pending = True
        for state in self._tracks.values():
            state.predicted_frame_id = None


def build_player_tracker(config: TrackingConfig) -> PlayerTracker:
    if config.association_method == "greedy":
        return NearestNeighbourTracker(
            config.max_normalized_distance,
            config.team_constraint,
            config.max_missed_anchors,
        )
    if config.association_method == "hungarian":
        return HungarianPlayerTracker(config)
    raise ValueError(f"unsupported association method: {config.association_method}")


def _lerp(a: float, b: float, ratio: float) -> float:
    return a + (b - a) * ratio


def interpolate_frames(left: TrackedFrame, right: TrackedFrame) -> list[TrackedFrame]:
    """Interpolate tracks shared by two anchors; endpoint frames are included."""
    if right.frame_id <= left.frame_id:
        raise ValueError("anchors must be in increasing frame order")
    left_by_id, right_by_id = (
        {p.track_id: p for p in frame.players} for frame in (left, right)
    )
    frames = []
    span = right.frame_id - left.frame_id
    for frame_id in range(left.frame_id, right.frame_id + 1):
        if frame_id == left.frame_id:
            frames.append(left)
            continue
        if frame_id == right.frame_id:
            frames.append(right)
            continue
        ratio = (frame_id - left.frame_id) / span
        players = []
        for track_id in sorted(left_by_id.keys() & right_by_id.keys()):
            first, second = left_by_id[track_id], right_by_id[track_id]
            box = Box(
                *(
                    _lerp(x, y, ratio)
                    for x, y in zip(
                        (first.box.left, first.box.top, first.box.right, first.box.bottom),
                        (second.box.left, second.box.top, second.box.right, second.box.bottom),
                    )
                )
            )
            players.append(
                TrackedPlayer(
                    track_id,
                    first.team,
                    box,
                    Point(
                        _lerp(first.foot.x, second.foot.x, ratio),
                        _lerp(first.foot.y, second.foot.y, ratio),
                    ),
                    _lerp(first.confidence, second.confidence, ratio),
                    first.possesses_ball if ratio < 0.5 else second.possesses_ball,
                    _lerp(first.tracking_confidence, second.tracking_confidence, ratio),
                    (
                        "confirmed"
                        if first.tracking_state == second.tracking_state == "confirmed"
                        else "uncertain"
                    ),
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
