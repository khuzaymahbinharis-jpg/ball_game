"""Strict, dependency-free response models for the future VLM boundary."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Team(str, Enum):
    A = "A"
    B = "B"
    UNKNOWN = "unknown"


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Point":
        if set(data) != {"x", "y"}:
            raise ValueError("point requires exactly x and y")
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
        required = {"left", "top", "right", "bottom"}
        if set(data) != required:
            raise ValueError("box requires exactly left, top, right, bottom")
        return cls(*(_number(data[k], k) for k in ("left", "top", "right", "bottom")))


@dataclass(frozen=True)
class PlayerDetection:
    detection_id: str
    team: Team
    box: Box
    confidence: float
    possesses_ball: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerDetection":
        allowed = {"detection_id", "team", "box", "confidence", "possesses_ball"}
        if set(data) - allowed or not {
            "detection_id",
            "team",
            "box",
            "confidence",
        } <= set(data):
            raise ValueError("invalid player fields")
        if not isinstance(data["detection_id"], str) or not data["detection_id"]:
            raise ValueError("detection_id must be a non-empty string")
        if "possesses_ball" in data and not isinstance(data["possesses_ball"], bool):
            raise ValueError("possesses_ball must be boolean")
        return cls(
            data["detection_id"],
            Team(data["team"]),
            Box.from_dict(data["box"]),
            _number(data["confidence"], "confidence"),
            data.get("possesses_ball", False),
        )


@dataclass(frozen=True)
class FrameDetection:
    frame_id: int
    players: tuple[PlayerDetection, ...]
    ball: Point | None
    ball_confidence: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameDetection":
        required = {"frame_id", "players", "ball", "ball_confidence"}
        if (
            set(data) != required
            or isinstance(data["frame_id"], bool)
            or not isinstance(data["frame_id"], int)
        ):
            raise ValueError("invalid frame response fields")
        if data["frame_id"] < 0 or not isinstance(data["players"], list):
            raise ValueError("invalid frame_id or players")
        ball = None if data["ball"] is None else Point.from_dict(data["ball"])
        confidence = (
            None
            if data["ball_confidence"] is None
            else _number(data["ball_confidence"], "ball_confidence")
        )
        if (ball is None) != (confidence is None):
            raise ValueError("ball and ball_confidence must both be present or absent")
        players = tuple(PlayerDetection.from_dict(p) for p in data["players"])
        if sum(p.possesses_ball for p in players) > 1:
            raise ValueError("at most one player may possess the ball")
        return cls(data["frame_id"], players, ball, confidence)
