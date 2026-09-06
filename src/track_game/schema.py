"""Strict structured response models for the VLM boundary."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


SCHEMA_VERSION = "frame-detection-v2"


class Team(str, Enum):
    A = "A"
    B = "B"
    UNCERTAIN = "uncertain"
    # Compatibility alias for the original scaffold.
    UNKNOWN = "uncertain"


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _exact_fields(data: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError(f"{name} requires exactly {', '.join(sorted(fields))}")
    return data


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Point":
        data = _exact_fields(data, {"x", "y"}, "point")
        return cls(_number(data["x"], "x"), _number(data["y"], "y"))


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            _number(getattr(self, name), name)
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("box must have positive area")

    @property
    def foot(self) -> Point:
        return Point((self.left + self.right) / 2, self.bottom)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Box":
        fields = {"left", "top", "right", "bottom"}
        data = _exact_fields(data, fields, "box")
        return cls(*(_number(data[k], k) for k in ("left", "top", "right", "bottom")))


@dataclass(frozen=True)
class PlayerDetection:
    detection_id: str
    team: Team
    box: Box
    foot: Point
    confidence: float
    team_confidence: float
    possesses_ball: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerDetection":
        fields = {
            "detection_id",
            "team",
            "box",
            "foot",
            "confidence",
            "team_confidence",
        }
        data = _exact_fields(data, fields, "player")
        if not isinstance(data["detection_id"], str) or not data["detection_id"]:
            raise ValueError("detection_id must be a non-empty string")
        return cls(
            data["detection_id"],
            Team(data["team"]),
            Box.from_dict(data["box"]),
            Point.from_dict(data["foot"]),
            _number(data["confidence"], "confidence"),
            _number(data["team_confidence"], "team_confidence"),
        )


@dataclass(frozen=True)
class BallDetection:
    center: Point
    box: Box | None
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BallDetection":
        data = _exact_fields(data, {"center", "box", "confidence"}, "ball")
        return cls(
            Point.from_dict(data["center"]),
            None if data["box"] is None else Box.from_dict(data["box"]),
            _number(data["confidence"], "ball confidence"),
        )


@dataclass(frozen=True)
class PossessionDetection:
    player_detection_id: str
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PossessionDetection":
        data = _exact_fields(
            data, {"player_detection_id", "confidence"}, "possession"
        )
        player_id = data["player_detection_id"]
        if not isinstance(player_id, str) or not player_id:
            raise ValueError("possession player_detection_id must be non-empty")
        return cls(player_id, _number(data["confidence"], "possession confidence"))


@dataclass(frozen=True)
class FrameDetection:
    frame_id: int
    players: tuple[PlayerDetection, ...]
    ball_detection: BallDetection | None
    possession: PossessionDetection | None
    uncertainty_notes: tuple[str, ...]

    @property
    def ball(self) -> Point | None:
        return None if self.ball_detection is None else self.ball_detection.center

    @property
    def ball_confidence(self) -> float | None:
        return None if self.ball_detection is None else self.ball_detection.confidence

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameDetection":
        fields = {"frame_id", "players", "ball", "possession", "uncertainty_notes"}
        data = _exact_fields(data, fields, "frame response")
        if (
            isinstance(data["frame_id"], bool)
            or not isinstance(data["frame_id"], int)
            or data["frame_id"] < 0
            or not isinstance(data["players"], list)
        ):
            raise ValueError("invalid frame_id or players")
        if not isinstance(data["uncertainty_notes"], list) or not all(
            isinstance(note, str) for note in data["uncertainty_notes"]
        ):
            raise ValueError("uncertainty_notes must be a list of strings")
        players = tuple(PlayerDetection.from_dict(p) for p in data["players"])
        ids = [player.detection_id for player in players]
        if len(ids) != len(set(ids)):
            raise ValueError("player detection IDs must be unique")
        ball = None if data["ball"] is None else BallDetection.from_dict(data["ball"])
        possession = (
            None
            if data["possession"] is None
            else PossessionDetection.from_dict(data["possession"])
        )
        if possession is not None:
            if possession.player_detection_id not in ids:
                raise ValueError("possession must reference a returned player")
            players = tuple(
                replace(
                    player,
                    possesses_ball=player.detection_id
                    == possession.player_detection_id,
                )
                for player in players
            )
        return cls(
            data["frame_id"],
            players,
            ball,
            possession,
            tuple(data["uncertainty_notes"]),
        )


def frame_detection_json_schema() -> dict[str, Any]:
    """Return the strict JSON Schema sent through OpenRouter structured outputs."""

    point = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "number", "minimum": 0, "maximum": 1},
            "y": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["x", "y"],
    }
    box = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            key: {"type": "number", "minimum": 0, "maximum": 1}
            for key in ("left", "top", "right", "bottom")
        },
        "required": ["left", "top", "right", "bottom"],
    }
    confidence = {"type": "number", "minimum": 0, "maximum": 1}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "frame_id": {"type": "integer", "minimum": 0},
            "players": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "detection_id": {"type": "string", "minLength": 1},
                        "team": {"type": "string", "enum": ["A", "B", "uncertain"]},
                        "box": box,
                        "foot": point,
                        "confidence": confidence,
                        "team_confidence": confidence,
                    },
                    "required": [
                        "detection_id",
                        "team",
                        "box",
                        "foot",
                        "confidence",
                        "team_confidence",
                    ],
                },
            },
            "ball": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "center": point,
                            "box": {"anyOf": [{"type": "null"}, box]},
                            "confidence": confidence,
                        },
                        "required": ["center", "box", "confidence"],
                    },
                ]
            },
            "possession": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "player_detection_id": {"type": "string", "minLength": 1},
                            "confidence": confidence,
                        },
                        "required": ["player_detection_id", "confidence"],
                    },
                ]
            },
            "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["frame_id", "players", "ball", "possession", "uncertainty_notes"],
    }
