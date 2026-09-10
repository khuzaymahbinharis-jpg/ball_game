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
class TrackConfidenceConfig:
    enabled: bool = False
    confirmed_threshold: float = 0.60
    lost_threshold: float = 0.20
    anchor_detection_weight: float = 0.65
    assignment_weight: float = 0.35
    flow_decay: float = 0.995
    camera_fallback_decay: float = 0.90
    coast_decay: float = 0.72
    missed_anchor_decay: float = 0.60

    def __post_init__(self) -> None:
        unit_values = (
            self.confirmed_threshold,
            self.lost_threshold,
            self.anchor_detection_weight,
            self.assignment_weight,
            self.flow_decay,
            self.camera_fallback_decay,
            self.coast_decay,
            self.missed_anchor_decay,
        )
        if any(value < 0 or value > 1 for value in unit_values):
            raise ValueError("track confidence values must be in [0, 1]")
        if self.lost_threshold >= self.confirmed_threshold:
            raise ValueError("lost_threshold must be below confirmed_threshold")
        if self.anchor_detection_weight + self.assignment_weight <= 0:
            raise ValueError("anchor confidence weights must be positive")


@dataclass(frozen=True)
class BallHungarianConfig:
    """Multi-cue cost and confidence settings for VLM ball-anchor association."""

    center_distance_weight: float = 0.20
    motion_distance_weight: float = 0.55
    size_change_weight: float = 0.10
    detection_confidence_weight: float = 0.15
    distance_scale: float = 0.18
    max_assignment_cost: float = 1.20
    anchor_detection_weight: float = 0.65
    assignment_weight: float = 0.35

    def __post_init__(self) -> None:
        weights = (
            self.center_distance_weight,
            self.motion_distance_weight,
            self.size_change_weight,
            self.detection_confidence_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("ball Hungarian cost weights cannot be negative")
        if sum(weights) <= 0:
            raise ValueError("at least one ball Hungarian cost weight must be positive")
        if self.distance_scale <= 0 or self.max_assignment_cost <= 0:
            raise ValueError("ball Hungarian distance and cost limits must be positive")
        confidence_weights = (
            self.anchor_detection_weight,
            self.assignment_weight,
        )
        if any(weight < 0 for weight in confidence_weights):
            raise ValueError("ball confidence weights cannot be negative")
        if sum(confidence_weights) <= 0:
            raise ValueError("ball confidence weights must be positive")


@dataclass(frozen=True)
class PlayerCVTrackingConfig:
    enabled: bool = False
    method: Literal["linear_interpolation", "sparse_lk"] = "linear_interpolation"
    downscale_width: int = 960
    max_features_per_player: int = 20
    min_tracked_points: int = 3
    feature_refresh_below: int = 7
    quality_level: float = 0.01
    min_feature_distance_px: int = 4
    lk_window_size: int = 21
    lk_max_level: int = 3
    max_forward_backward_error_px: float = 2.5
    max_lk_error: float = 40.0
    max_displacement_fraction: float = 0.08
    max_coast_frames: int = 3
    box_padding_fraction: float = 0.08
    visual_position_gain: float = 0.28
    visual_velocity_gain: float = 0.06
    visual_velocity_decay: float = 0.88
    max_visual_correction_fraction: float = 0.012

    def __post_init__(self) -> None:
        if self.enabled and self.method != "sparse_lk":
            raise ValueError("enabled player CV tracking requires sparse_lk")
        if self.downscale_width < 160:
            raise ValueError("player tracking downscale_width is too small")
        if self.max_features_per_player < 1 or self.min_tracked_points < 1:
            raise ValueError("invalid player feature counts")
        if self.feature_refresh_below < self.min_tracked_points:
            raise ValueError("feature refresh threshold is too small")
        if self.lk_window_size < 3 or self.lk_window_size % 2 == 0:
            raise ValueError("lk_window_size must be an odd integer of at least 3")
        if self.lk_max_level < 0 or self.min_feature_distance_px < 1:
            raise ValueError("invalid player optical-flow settings")
        if not 0 < self.quality_level <= 1:
            raise ValueError("quality_level must be in (0, 1]")
        if self.max_forward_backward_error_px <= 0 or self.max_lk_error <= 0:
            raise ValueError("player optical-flow error thresholds must be positive")
        if not 0 < self.max_displacement_fraction <= 1:
            raise ValueError("max_displacement_fraction must be in (0, 1]")
        if self.max_coast_frames < 0 or not 0 <= self.box_padding_fraction <= 1:
            raise ValueError("invalid player tracking lifecycle settings")
        smoothing_values = (
            self.visual_position_gain,
            self.visual_velocity_gain,
            self.visual_velocity_decay,
            self.max_visual_correction_fraction,
        )
        if any(value < 0 or value > 1 for value in smoothing_values):
            raise ValueError("visual smoothing values must be in [0, 1]")
        if self.visual_position_gain <= 0 or self.max_visual_correction_fraction <= 0:
            raise ValueError("visual smoothing gains must be positive")


@dataclass(frozen=True)
class CameraMotionConfig:
    enabled: bool = False
    transform_type: Literal["translation", "partial_affine"] = "translation"
    downscale_width: int = 480
    max_features: int = 240
    min_tracked_points: int = 24
    quality_level: float = 0.01
    min_feature_distance_px: int = 7
    lk_window_size: int = 21
    lk_max_level: int = 3
    max_forward_backward_error_px: float = 2.0
    ransac_threshold_px: float = 2.5
    minimum_transform_confidence: float = 0.35

    def __post_init__(self) -> None:
        if self.downscale_width < 64 or self.max_features < 1:
            raise ValueError("invalid camera feature settings")
        if self.min_tracked_points < 3 or self.min_tracked_points > self.max_features:
            raise ValueError("invalid minimum camera point count")
        if not 0 < self.quality_level <= 1 or self.min_feature_distance_px < 1:
            raise ValueError("invalid camera feature quality settings")
        if self.lk_window_size < 3 or self.lk_window_size % 2 == 0:
            raise ValueError("camera LK window must be an odd integer of at least 3")
        if self.lk_max_level < 0 or self.max_forward_backward_error_px <= 0:
            raise ValueError("invalid camera optical-flow settings")
        if self.ransac_threshold_px <= 0:
            raise ValueError("ransac_threshold_px must be positive")
        if not 0 <= self.minimum_transform_confidence <= 1:
            raise ValueError("minimum_transform_confidence must be in [0, 1]")


@dataclass(frozen=True)
class SceneCutConfig:
    enabled: bool = False
    downscale_width: int = 192
    frame_difference_weight: float = 0.65
    histogram_weight: float = 0.35
    camera_failure_bonus: float = 0.08
    threshold: float = 0.35
    minimum_frame_difference: float = 0.18
    minimum_gap_frames: int = 15

    def __post_init__(self) -> None:
        if self.downscale_width < 32:
            raise ValueError("scene-cut downscale_width is too small")
        if self.frame_difference_weight < 0 or self.histogram_weight < 0:
            raise ValueError("scene-cut weights cannot be negative")
        if self.frame_difference_weight + self.histogram_weight <= 0:
            raise ValueError("at least one scene-cut weight must be positive")
        unit_values = (
            self.camera_failure_bonus,
            self.threshold,
            self.minimum_frame_difference,
        )
        if any(value < 0 or value > 1 for value in unit_values):
            raise ValueError("scene-cut thresholds must be in [0, 1]")
        if self.minimum_gap_frames < 1:
            raise ValueError("minimum_gap_frames must be positive")


@dataclass(frozen=True)
class TrackingConfig:
    max_normalized_distance: float = 0.2
    team_constraint: bool = True
    max_missed_anchors: int = 2
    association_method: Literal["greedy", "hungarian"] = "greedy"
    hungarian: HungarianConfig = field(default_factory=HungarianConfig)
    confidence: TrackConfidenceConfig = field(default_factory=TrackConfidenceConfig)

    def __post_init__(self) -> None:
        if self.max_normalized_distance <= 0:
            raise ValueError("max_normalized_distance must be positive")
        if self.max_missed_anchors < 0:
            raise ValueError("max_missed_anchors cannot be negative")


@dataclass(frozen=True)
class BallTrackingConfig:
    enabled: bool = False
    association_method: Literal["direct", "hungarian"] = "hungarian"
    hungarian: BallHungarianConfig = field(default_factory=BallHungarianConfig)
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
        if self.association_method not in ("direct", "hungarian"):
            raise ValueError("unsupported ball association method")
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
class ShotContextConfig:
    """Post-VLM geometric guard against tracking broadcast close-ups."""

    enabled: bool = True
    maximum_players_in_closeup: int = 3
    minimum_dominant_box_area: float = 0.15

    def __post_init__(self) -> None:
        if self.maximum_players_in_closeup < 1:
            raise ValueError("maximum_players_in_closeup must be positive")
        if not 0 < self.minimum_dominant_box_area <= 1:
            raise ValueError("minimum_dominant_box_area must be in (0, 1]")


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
    player_cv_tracking: PlayerCVTrackingConfig = field(default_factory=PlayerCVTrackingConfig)
    camera_motion: CameraMotionConfig = field(default_factory=CameraMotionConfig)
    scene_cut: SceneCutConfig = field(default_factory=SceneCutConfig)
    ball_tracking: BallTrackingConfig = field(default_factory=BallTrackingConfig)
    shot_context: ShotContextConfig = field(default_factory=ShotContextConfig)
    ruler: RulerConfig = field(default_factory=RulerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def as_dict(self) -> dict:
        return asdict(self)
