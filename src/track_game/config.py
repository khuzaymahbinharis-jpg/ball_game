from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SamplingConfig:
    every_n_frames: int = 15
    include_last: bool = True

    def __post_init__(self) -> None:
        if self.every_n_frames < 1:
            raise ValueError("every_n_frames must be positive")


@dataclass(frozen=True)
class TrackingConfig:
    max_normalized_distance: float = 0.2
    team_constraint: bool = True
    max_missed_anchors: int = 2


@dataclass(frozen=True)
class RulerConfig:
    enabled: bool = True
    margin_px: int = 44
    tick_step: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "mock"
    model: str = "mock/deterministic-v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Deliberately no key field: secrets belong in tomorrow's provider adapter.


@dataclass(frozen=True)
class PipelineConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    ruler: RulerConfig = field(default_factory=RulerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def as_dict(self) -> dict:
        return asdict(self)
