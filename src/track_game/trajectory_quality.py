"""Objective motion-quality measurements for rendered tracking trajectories."""

from __future__ import annotations

from math import hypot
from statistics import median
from typing import Iterable

from .tracking import TrackedFrame


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def player_trajectory_quality(
    timeline: Iterable[TrackedFrame],
    *,
    anchor_frames: Iterable[int] = (),
    scene_cut_frames: Iterable[int] = (),
) -> dict[str, float | int]:
    """Measure visible step, acceleration, jerk, and anchor correction magnitudes.

    Only consecutive observations of the same logical track are compared. Scene
    cuts and newly created identities therefore cannot inflate the measurements.
    Values are normalized to the source frame width/height coordinate system.
    """

    anchors = set(anchor_frames)
    cuts = set(scene_cut_frames)
    history: dict[int, list[tuple[int, float, float]]] = {}
    for frame in timeline:
        for player in frame.players:
            history.setdefault(player.track_id, []).append(
                (frame.frame_id, player.foot.x, player.foot.y)
            )

    steps: list[float] = []
    anchor_steps: list[float] = []
    accelerations: list[float] = []
    jerks: list[float] = []
    for observations in history.values():
        velocities: list[tuple[int, float, float]] = []
        for left, right in zip(observations, observations[1:]):
            if right[0] != left[0] + 1 or right[0] in cuts:
                velocities = []
                continue
            dx, dy = right[1] - left[1], right[2] - left[2]
            step = hypot(dx, dy)
            steps.append(step)
            if right[0] in anchors:
                anchor_steps.append(step)
            velocities.append((right[0], dx, dy))
        for left, right in zip(velocities, velocities[1:]):
            if right[0] != left[0] + 1:
                continue
            accelerations.append(hypot(right[1] - left[1], right[2] - left[2]))
        for first, second, third in zip(velocities, velocities[1:], velocities[2:]):
            if second[0] != first[0] + 1 or third[0] != second[0] + 1:
                continue
            ax1, ay1 = second[1] - first[1], second[2] - first[2]
            ax2, ay2 = third[1] - second[1], third[2] - second[2]
            jerks.append(hypot(ax2 - ax1, ay2 - ay1))

    return {
        "consecutive_steps": len(steps),
        "anchor_steps": len(anchor_steps),
        "median_step": median(steps) if steps else 0.0,
        "p95_step": _percentile(steps, 0.95),
        "median_acceleration": median(accelerations) if accelerations else 0.0,
        "p95_acceleration": _percentile(accelerations, 0.95),
        "median_jerk": median(jerks) if jerks else 0.0,
        "p95_jerk": _percentile(jerks, 0.95),
        "median_anchor_step": median(anchor_steps) if anchor_steps else 0.0,
        "p95_anchor_step": _percentile(anchor_steps, 0.95),
    }
