from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SamplingConfig:
    every_n_frames: int = 15
    include_last: bool = True

    def __post_init__(self) -> None:
        if self.every_n_frames < 1:
            raise ValueError("every_n_frames must be positive")


@dataclass(frozen=True)
class HungarianConfig:
    foot_distance_weight: float = 0.45
    iou_weight: float = 0.20
    size_change_weight: float = 0.10
    motion_weight: float = 0.25
    team_mismatch_penalty: float = 0.35
    uncertain_team_penalty: float = 0.05
    max_assignment_cost: float = 1.20
    velocity_smoothing: float = 0.65
    team_score_decay: float = 0.95
    team_switch_margin: float = 1.50

    def __post_init__(self) -> None:
        weights = (
            self.foot_distance_weight,
            self.iou_weight,
            self.size_change_weight,
            self.motion_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Hungarian cost weights cannot be negative")
        if sum(weights) <= 0:
            raise ValueError("at least one Hungarian cost weight must be positive")
        if self.team_mismatch_penalty < 0 or self.uncertain_team_penalty < 0:
            raise ValueError("team penalties cannot be negative")
        if self.max_assignment_cost <= 0:
            raise ValueError("max_assignment_cost must be positive")
        if not 0 <= self.velocity_smoothing <= 1:
            raise ValueError("velocity_smoothing must be in [0, 1]")
        if not 0 <= self.team_score_decay <= 1:
            raise ValueError("team_score_decay must be in [0, 1]")
        if self.team_switch_margin < 1:
            raise ValueError("team_switch_margin must be at least 1")


@dataclass(frozen=True)
class TrackingConfig:
    max_normalized_distance: float = 0.2
    team_constraint: bool = True
    max_missed_anchors: int = 2
    association_method: Literal["greedy", "hungarian"] = "greedy"
    hungarian: HungarianConfig = field(default_factory=HungarianConfig)

    def __post_init__(self) -> None:
        if self.max_normalized_distance <= 0:
            raise ValueError("max_normalized_distance must be positive")
        if self.max_missed_anchors < 0:
            raise ValueError("max_missed_anchors cannot be negative")


@dataclass(frozen=True)
class BallTrackingConfig:
    enabled: bool = False
    use_kalman: bool = True
    lk_window_size: int = 21
    lk_max_level: int = 3
    max_forward_backward_error_px: float = 2.0
    max_lk_error: float = 35.0
    min_tracked_points: int = 2
    max_features: int = 16
    default_roi_radius_px: int = 18
    max_displacement_fraction: float = 0.08
    max_coast_frames: int = 4
    flow_confidence_decay: float = 0.985
    coast_confidence_decay: float = 0.75
    lost_confidence_threshold: float = 0.10
    possession_prior_weight: float = 0.08
    possession_prior_max_distance: float = 0.12

    def __post_init__(self) -> None:
        if self.lk_window_size < 3 or self.lk_window_size % 2 == 0:
            raise ValueError("lk_window_size must be an odd integer of at least 3")
        if self.lk_max_level < 0 or self.min_tracked_points < 1 or self.max_features < 1:
            raise ValueError("invalid optical-flow feature settings")
        if self.default_roi_radius_px < 2 or self.max_coast_frames < 0:
            raise ValueError("invalid ball lifecycle settings")
        unit_values = (
            self.max_displacement_fraction,
            self.flow_confidence_decay,
            self.coast_confidence_decay,
            self.lost_confidence_threshold,
            self.possession_prior_weight,
            self.possession_prior_max_distance,
        )
        if any(value < 0 or value > 1 for value in unit_values):
            raise ValueError("ball tracking fractions must be in [0, 1]")
        if self.max_forward_backward_error_px <= 0 or self.max_lk_error <= 0:
            raise ValueError("optical-flow error thresholds must be positive")


@dataclass(frozen=True)
class RulerConfig:
    enabled: bool = True
    margin_px: int = 64
    tick_step: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "mock"
    model: str = "mock/deterministic-v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    reasoning_effort: str = "minimal"
    max_output_tokens: int = 4096
    # Deliberately no key field: credentials are loaded at runtime, never serialized.


@dataclass(frozen=True)
class PipelineConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    ball_tracking: BallTrackingConfig = field(default_factory=BallTrackingConfig)
    ruler: RulerConfig = field(default_factory=RulerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def as_dict(self) -> dict:
        return asdict(self)
