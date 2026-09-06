import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    status: str
    source_clip: str
    source_timestamp_seconds: float
    source_frame_number: int
    model: str
    pricing_snapshot: dict[str, Any]
    image_resolution: tuple[int, int]
    prompt_image_resolution: tuple[int, int]
    ruler_grounding: bool
    prompt_version: str
    schema_version: str
    reasoning_setting: dict[str, Any]
    vlm_calls: int
    provider_request_id: str | None = None
    returned_model: str | None = None
    returned_provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    vlm_request_latency_seconds: float | None = None
    total_wall_clock_seconds: float | None = None
    retries: int = 0
    errors: tuple[str, ...] = ()
    actual_openrouter_cost_usd: float | None = None
    estimated_cost_low_usd: float | None = None
    estimated_cost_expected_usd: float | None = None
    estimated_cost_high_usd: float | None = None
    schema_validation_success: bool | None = None
    players_returned: int | None = None
    ball_detected: bool | None = None
    possession_returned: bool | None = None
    player_localization_quality: str | None = None
    team_assignment_quality: str | None = None
    ball_localization_quality: str | None = None
    obvious_detection_failures: str | None = None
    notes: str | None = None
    anchor_frame_sampling_interval: int | None = None
    tracker_data_association_method: str | None = None
    id_switches: int | None = None
    lost_tracks: int | None = None
    recovered_tracks: int | None = None
    vlm_redetection_frequency: int | None = None
    complete_video_processing_seconds: float | None = None
    complete_video_api_cost_usd: float | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.experiment_id or self.status not in {
            "completed",
            "failed",
        }:
            raise ValueError("invalid experiment identity or status")
        if (
            self.source_timestamp_seconds < 0
            or self.source_frame_number < 0
            or self.vlm_calls < 0
            or self.retries < 0
        ):
            raise ValueError("experiment counters cannot be negative")
        if any(dimension < 1 for dimension in (*self.image_resolution, *self.prompt_image_resolution)):
            raise ValueError("image dimensions must be positive")
        measured_values = (
            self.vlm_request_latency_seconds,
            self.total_wall_clock_seconds,
            self.actual_openrouter_cost_usd,
            self.estimated_cost_low_usd,
            self.estimated_cost_expected_usd,
            self.estimated_cost_high_usd,
            self.complete_video_processing_seconds,
            self.complete_video_api_cost_usd,
        )
        if any(value is not None and value < 0 for value in measured_values):
            raise ValueError("measurements cannot be negative")


class JsonlExperimentLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: ExperimentRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        ]


@dataclass(frozen=True)
class HumanTrackingEvaluation:
    manual_id_switches: int | None = None
    fragmentation: int | None = None
    ball_misses: int | None = None
    team_assignment_mistakes: int | None = None
    marker_drift_events: int | None = None
    player_tracking_failures: int | None = None
    scene_cut_failures: int | None = None
    overall_visual_quality_notes: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        counts = (
            self.manual_id_switches,
            self.fragmentation,
            self.ball_misses,
            self.team_assignment_mistakes,
            self.marker_drift_events,
            self.player_tracking_failures,
            self.scene_cut_failures,
        )
        if any(value is not None and value < 0 for value in counts):
            raise ValueError("human evaluation counts cannot be negative")


@dataclass(frozen=True)
class TrackingMetricsRecord:
    experiment_id: str
    variant: str
    model: str
    anchor_count: int
    planned_vlm_calls: int
    attempted_vlm_calls: int
    successful_vlm_calls: int
    actual_api_cost_usd: float | None
    api_cost_complete: bool
    summed_vlm_latency_seconds: float | None
    api_batch_seconds: float | None
    local_processing_seconds: float | None
    total_processing_seconds: float | None
    unique_ids_over_time: dict[int, tuple[int, ...]]
    tracks_created: int
    tracks_expired: int
    tracks_lost: int
    tracks_recovered: int
    ball_lost_events: int | None
    ball_recovered_events: int | None
    ball_lost_frames: int | None
    uncertain_track_frames: int | None = None
    uncertainty_recoveries: int | None = None
    scene_cut_frames: tuple[int, ...] = ()
    camera_motion_successes: int | None = None
    camera_motion_failures: int | None = None
    camera_motion_seconds: float | None = None
    player_cv_tracking_seconds: float | None = None
    player_cv_total_pass_seconds: float | None = None
    scene_cut_detection_seconds: float | None = None
    ball_tracking_seconds: float | None = None
    rendering_seconds: float | None = None
    average_local_seconds_per_frame: float | None = None
    human_evaluation: HumanTrackingEvaluation = field(
        default_factory=HumanTrackingEvaluation
    )

    def __post_init__(self) -> None:
        counters = (
            self.anchor_count,
            self.planned_vlm_calls,
            self.attempted_vlm_calls,
            self.successful_vlm_calls,
            self.tracks_created,
            self.tracks_expired,
            self.tracks_lost,
            self.tracks_recovered,
        )
        if not self.experiment_id or not self.variant or any(value < 0 for value in counters):
            raise ValueError("invalid tracking metrics identity or counters")
        measurements = (
            self.actual_api_cost_usd,
            self.summed_vlm_latency_seconds,
            self.api_batch_seconds,
            self.local_processing_seconds,
            self.total_processing_seconds,
            self.ball_lost_events,
            self.ball_recovered_events,
            self.ball_lost_frames,
            self.uncertain_track_frames,
            self.uncertainty_recoveries,
            self.camera_motion_successes,
            self.camera_motion_failures,
            self.camera_motion_seconds,
            self.player_cv_tracking_seconds,
            self.player_cv_total_pass_seconds,
            self.scene_cut_detection_seconds,
            self.ball_tracking_seconds,
            self.rendering_seconds,
            self.average_local_seconds_per_frame,
        )
        if any(value is not None and value < 0 for value in measurements):
            raise ValueError("tracking measurements cannot be negative")


class TrackingMetricsWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, record: TrackingMetricsRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(record), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))
