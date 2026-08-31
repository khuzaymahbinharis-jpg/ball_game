import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    model: str
    clip: str
    frame_sampling_rate: int
    vlm_calls: int
    wall_clock_seconds: float | None = None
    vlm_latency_seconds: float | None = None
    api_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    retries: int = 0
    errors: tuple[str, ...] = ()
    tracking_config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if (
            not self.experiment_id
            or self.frame_sampling_rate < 1
            or self.vlm_calls < 0
            or self.retries < 0
        ):
            raise ValueError("invalid experiment record")
        for value in (
            self.wall_clock_seconds,
            self.vlm_latency_seconds,
            self.api_cost_usd,
        ):
            if value is not None and value < 0:
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
