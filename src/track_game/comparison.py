"""Controlled baseline/improved tracker configurations for the next experiment."""

from dataclasses import dataclass

from .config import BallTrackingConfig, PipelineConfig, SamplingConfig, TrackingConfig


@dataclass(frozen=True)
class ComparisonVariant:
    name: str
    description: str
    pipeline: PipelineConfig


def baseline_variant() -> ComparisonVariant:
    return ComparisonVariant(
        name="baseline_greedy",
        description="Greedy nearest-foot association, hard team constraint, linear ball interpolation.",
        pipeline=PipelineConfig(
            sampling=SamplingConfig(every_n_frames=15, include_last=True),
            tracking=TrackingConfig(
                association_method="greedy",
                team_constraint=True,
                max_missed_anchors=2,
            ),
            ball_tracking=BallTrackingConfig(enabled=False),
        ),
    )


def improved_variant() -> ComparisonVariant:
    return ComparisonVariant(
        name="hungarian_persistent_ball_cv",
        description=(
            "Hungarian multi-cue association, soft stabilized team evidence, "
            "lost-track reassociation, and VLM-initialized LK/Kalman ball tracking."
        ),
        pipeline=PipelineConfig(
            sampling=SamplingConfig(every_n_frames=15, include_last=True),
            tracking=TrackingConfig(
                association_method="hungarian",
                team_constraint=False,
                max_missed_anchors=2,
            ),
            ball_tracking=BallTrackingConfig(enabled=True),
        ),
    )


def comparison_variants() -> tuple[ComparisonVariant, ComparisonVariant]:
    return baseline_variant(), improved_variant()
