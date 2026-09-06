"""Prepare and, only after approval, run the 2 FPS tracker comparison."""

import argparse
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from .ball_tracking import BallTrackingResult, VLMInitializedBallTracker, apply_ball_track
from .comparison import ComparisonVariant, comparison_variants
from .experiments import TrackingMetricsRecord, TrackingMetricsWriter
from .fullclip import full_clip_detection_prompt
from .provider import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterResult,
    OpenRouterValidationError,
    OpenRouterVLMProvider,
)
from .ruler import add_normalized_rulers
from .sampling import sample_frame_indices
from .schema import SCHEMA_VERSION, FrameDetection, frame_detection_json_schema
from .test1 import _detection_as_dict
from .tracking import TrackedFrame, build_player_tracker, interpolate_sequence
from .video import extract_video_frames, probe_video, render_tracked_video


EXPERIMENT_ID = "test-3-2fps-tracker-comparison"
FRAME_INTERVAL = 15
ANCHOR_FRAMES = tuple(sample_frame_indices(900, FRAME_INTERVAL, include_last=True))
VLM_CALLS = len(ANCHOR_FRAMES)
MAX_CONCURRENCY = 8
MAX_OUTPUT_TOKENS = 4096
REASONING_SETTING = {"effort": "minimal", "exclude": True}
PROMPT_VERSION = "fullclip-ruler-v1-unchanged"

PRIOR_VALID_ANCHOR_COSTS = (
    0.000981,
    0.00242125,
    0.0031625,
    0.003029,
    0.0025925,
    0.002381,
    0.0030815,
    0.0023615,
    0.0030755,
)
PRIOR_COST_PER_VALID_ANCHOR = sum(PRIOR_VALID_ANCHOR_COSTS) / len(
    PRIOR_VALID_ANCHOR_COSTS
)
COST_ESTIMATE = {
    "low_usd": min(PRIOR_VALID_ANCHOR_COSTS) * VLM_CALLS,
    "expected_usd": PRIOR_COST_PER_VALID_ANCHOR * VLM_CALLS,
    "high_usd": VLM_CALLS * (2588 * 0.25 / 1_000_000 + MAX_OUTPUT_TOKENS * 1.50 / 1_000_000),
    "basis": (
        "Low and expected use the observed minimum and mean across nine schema-valid "
        "Test 2 anchors. High prices 2,588 input tokens and the configured 4,096-token "
        "output cap for every call at the current listed rates."
    ),
    "prior_measured_cost_per_valid_anchor_usd": PRIOR_COST_PER_VALID_ANCHOR,
    "prior_max_observed_cost_per_valid_anchor_usd": max(PRIOR_VALID_ANCHOR_COSTS),
    "prior_cost_observations": len(PRIOR_VALID_ANCHOR_COSTS),
    "note": "The prior batch had two invalid responses whose costs were unavailable.",
}

PRICING_SNAPSHOT: dict[str, Any] = {
    "captured_at": "2026-09-05",
    "model": DEFAULT_OPENROUTER_MODEL,
    "currency": "USD",
    "listed_input_per_million_tokens": 0.25,
    "listed_output_per_million_tokens": 1.50,
    "source": "https://openrouter.ai/google/gemini-3.1-flash-lite/pricing",
}


@dataclass(frozen=True)
class ComparisonPaths:
    root: Path
    clip: Path
    experiment_dir: Path
    anchors_dir: Path
    responses_dir: Path
    prompt_template: Path
    schema: Path
    preflight: Path
    anchor_metrics: Path
    run_summary: Path
    baseline_metrics: Path
    improved_metrics: Path
    baseline_video: Path
    improved_video: Path


def comparison_paths(repository_root: str | Path) -> ComparisonPaths:
    root = Path(repository_root).resolve()
    experiment_dir = root / "experiments" / "test_3_2fps_comparison"
    artifacts = experiment_dir / "artifacts"
    return ComparisonPaths(
        root=root,
        clip=root / "clips" / "dev" / "spurs_thunder_test.mp4",
        experiment_dir=experiment_dir,
        anchors_dir=artifacts / "anchors",
        responses_dir=artifacts / "responses",
        prompt_template=experiment_dir / "prompt_template_v1.txt",
        schema=experiment_dir / "schema_v2.json",
        preflight=experiment_dir / "preflight.json",
        anchor_metrics=experiment_dir / "anchor_runs.jsonl",
        run_summary=experiment_dir / "summary.json",
        baseline_metrics=experiment_dir / "baseline_metrics.json",
        improved_metrics=experiment_dir / "improved_metrics.json",
        baseline_video=root / "outputs" / "spurs_thunder_test3_baseline.mp4",
        improved_video=root / "outputs" / "spurs_thunder_test3_improved.mp4",
    )


def _original_path(paths: ComparisonPaths, frame_id: int) -> Path:
    return paths.anchors_dir / f"frame_{frame_id:04d}_original.png"


def _ruler_path(paths: ComparisonPaths, frame_id: int) -> Path:
    return paths.anchors_dir / f"frame_{frame_id:04d}_ruler.png"


def _variant_manifest(variant: ComparisonVariant) -> dict[str, Any]:
    return {
        "name": variant.name,
        "description": variant.description,
        "sampling": asdict(variant.pipeline.sampling),
        "tracking": asdict(variant.pipeline.tracking),
        "ball_tracking": asdict(variant.pipeline.ball_tracking),
    }


def prepare_comparison(repository_root: str | Path) -> ComparisonPaths:
    """Complete all frame/prompt/schema preparation without constructing a provider."""

    paths = comparison_paths(repository_root)
    info = probe_video(paths.clip)
    if info.frame_count != 900 or abs(info.fps - 30.0) > 0.01:
        raise ValueError("comparison requires the verified 900-frame, 30 FPS clip")
    originals = extract_video_frames(paths.clip, ANCHOR_FRAMES, paths.anchors_dir)
    for frame_id, original_path in originals.items():
        ruler_path = _ruler_path(paths, frame_id)
        if ruler_path.is_file():
            continue
        with Image.open(original_path) as source:
            ruler = add_normalized_rulers(source.convert("RGB"), margin_px=64, tick_step=0.1)
        ruler.save(ruler_path, format="PNG", optimize=True)
    paths.experiment_dir.mkdir(parents=True, exist_ok=True)
    paths.prompt_template.write_text(
        full_clip_detection_prompt(0).replace("must be 0", "must be {frame_id}") + "\n",
        encoding="utf-8",
    )
    paths.schema.write_text(
        json.dumps(frame_detection_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline, improved = comparison_variants()
    preflight = {
        "experiment_id": EXPERIMENT_ID,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "experimental_purpose": (
            "Isolate association, identity persistence, and ball tracking quality while "
            "holding the VLM model, prompt, schema, rulers, clip, and 2 FPS anchors fixed."
        ),
        "model": DEFAULT_OPENROUTER_MODEL,
        "source_clip": str(paths.clip.relative_to(paths.root)),
        "clip": asdict(info),
        "frame_interval": FRAME_INTERVAL,
        "sampling_fps": info.fps / FRAME_INTERVAL,
        "anchor_frames": list(ANCHOR_FRAMES),
        "planned_vlm_calls": VLM_CALLS,
        "automatic_retries": 0,
        "max_concurrency": MAX_CONCURRENCY,
        "prompt_changed": False,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "reasoning": REASONING_SETTING,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "pricing_snapshot": PRICING_SNAPSHOT,
        "cost_estimate": COST_ESTIMATE,
        "variants": [_variant_manifest(baseline), _variant_manifest(improved)],
        "human_evaluation_fields": {
            "manual_id_switches": None,
            "fragmentation": None,
            "ball_misses": None,
            "team_assignment_mistakes": None,
            "notes": None,
        },
        "paid_requests_attempted": 0,
        "approval_gate": f"Requires new explicit approval for exactly {VLM_CALLS} calls.",
    }
    paths.preflight.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _request_one(
    provider: OpenRouterVLMProvider, paths: ComparisonPaths, frame_id: int
) -> OpenRouterResult:
    with Image.open(_ruler_path(paths, frame_id)) as image:
        return provider.request_detection(
            frame_id, image.copy(), prompt=full_clip_detection_prompt(frame_id)
        )


def _append_anchor_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _covered_timeline(timeline: list[TrackedFrame], total_frames: int) -> list[TrackedFrame]:
    if not timeline:
        raise RuntimeError("cannot build a timeline without valid anchors")
    first, last = timeline[0], timeline[-1]
    before = [
        replace(first, frame_id=frame_id, ball=None, ball_confidence=None)
        for frame_id in range(first.frame_id)
    ]
    after = [
        replace(last, frame_id=frame_id, ball=None, ball_confidence=None)
        for frame_id in range(last.frame_id + 1, total_frames)
    ]
    output = before + timeline + after
    if [frame.frame_id for frame in output] != list(range(total_frames)):
        raise RuntimeError("tracked timeline does not cover every clip frame")
    return output


def _process_variant(
    paths: ComparisonPaths,
    variant: ComparisonVariant,
    detections: list[FrameDetection],
    attempted_calls: int,
    actual_cost: float | None,
    cost_complete: bool,
    summed_latency: float | None,
    api_batch_seconds: float,
    output_video: Path,
    metrics_path: Path,
) -> TrackingMetricsRecord:
    local_started = perf_counter()
    tracker = build_player_tracker(variant.pipeline.tracking)
    tracked_anchors = [tracker.update(detection) for detection in detections]
    timeline = _covered_timeline(interpolate_sequence(tracked_anchors), 900)
    ball_result: BallTrackingResult | None = None
    if variant.pipeline.ball_tracking.enabled:
        anchor_map = {
            detection.frame_id: detection.ball_detection for detection in detections
        }
        ball_result = VLMInitializedBallTracker(
            variant.pipeline.ball_tracking
        ).track_video(paths.clip, anchor_map, timeline)
        timeline = apply_ball_track(timeline, ball_result)
    render_tracked_video(paths.clip, output_video, timeline, show_ids=True)
    local_seconds = perf_counter() - local_started
    lifecycle = tracker.lifecycle
    ball_stats = None if ball_result is None else ball_result.stats
    record = TrackingMetricsRecord(
        experiment_id=EXPERIMENT_ID,
        variant=variant.name,
        model=DEFAULT_OPENROUTER_MODEL,
        anchor_count=VLM_CALLS,
        planned_vlm_calls=VLM_CALLS,
        attempted_vlm_calls=attempted_calls,
        successful_vlm_calls=len(detections),
        actual_api_cost_usd=actual_cost,
        api_cost_complete=cost_complete,
        summed_vlm_latency_seconds=summed_latency,
        api_batch_seconds=api_batch_seconds,
        local_processing_seconds=local_seconds,
        total_processing_seconds=api_batch_seconds + local_seconds,
        unique_ids_over_time={
            frame.frame_id: tuple(player.track_id for player in frame.players)
            for frame in tracked_anchors
        },
        tracks_created=lifecycle.tracks_created,
        tracks_expired=lifecycle.tracks_expired,
        tracks_lost=lifecycle.tracks_lost,
        tracks_recovered=lifecycle.tracks_recovered,
        ball_lost_events=None if ball_stats is None else ball_stats.lost_events,
        ball_recovered_events=None if ball_stats is None else ball_stats.recovered_events,
        ball_lost_frames=None if ball_stats is None else ball_stats.lost_frames,
    )
    TrackingMetricsWriter(metrics_path).write(record)
    return record


def run_approved_comparison(
    repository_root: str | Path, approved_call_count: int
) -> ComparisonPaths:
    """Make exactly 61 approved calls once, then compare both trackers locally."""

    if approved_call_count != VLM_CALLS:
        raise PermissionError(
            f"comparison requires explicit approval for exactly {VLM_CALLS} calls"
        )
    paths = comparison_paths(repository_root)
    if paths.anchor_metrics.is_file() and paths.anchor_metrics.read_text(encoding="utf-8").strip():
        raise PermissionError("this comparison already has call records; refusing a duplicate run")
    paths = prepare_comparison(repository_root)
    provider = OpenRouterVLMProvider.from_repository_env(
        repository_root,
        model=DEFAULT_OPENROUTER_MODEL,
        reasoning_effort=REASONING_SETTING["effort"],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    preflight = json.loads(paths.preflight.read_text(encoding="utf-8"))
    preflight["paid_requests_attempted"] = VLM_CALLS
    preflight["started_at"] = datetime.now(timezone.utc).isoformat()
    paths.preflight.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    batch_started = perf_counter()
    results: dict[int, OpenRouterResult] = {}
    errors: dict[int, str] = {}
    observed_costs: list[float] = []
    observed_latencies: list[float] = []
    cost_complete = True
    futures: dict[Future[OpenRouterResult], int] = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        for frame_id in ANCHOR_FRAMES:
            futures[executor.submit(_request_one, provider, paths, frame_id)] = frame_id
        for future in as_completed(futures):
            frame_id = futures[future]
            result: OpenRouterResult | None = None
            error: Exception | None = None
            try:
                result = future.result()
                results[frame_id] = result
                response_path = paths.responses_dir / f"frame_{frame_id:04d}_response.json"
                response_path.parent.mkdir(parents=True, exist_ok=True)
                response_path.write_text(
                    json.dumps(_detection_as_dict(result.detection), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                usage, latency = result.usage, result.latency_seconds
            except Exception as caught:
                error = caught
                errors[frame_id] = f"{type(caught).__name__}: {caught}"
                if isinstance(caught, OpenRouterValidationError):
                    usage, latency = caught.usage, caught.latency_seconds
                    if caught.raw_content:
                        paths.responses_dir.mkdir(parents=True, exist_ok=True)
                        (paths.responses_dir / f"frame_{frame_id:04d}_invalid.txt").write_text(
                            caught.raw_content, encoding="utf-8"
                        )
                else:
                    usage, latency = None, None
            cost = None if usage is None else usage.cost_usd
            if cost is None:
                cost_complete = False
            else:
                observed_costs.append(cost)
            if latency is not None:
                observed_latencies.append(latency)
            _append_anchor_row(
                paths.anchor_metrics,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "frame_id": frame_id,
                    "model": DEFAULT_OPENROUTER_MODEL,
                    "schema_validation_success": result is not None,
                    "input_tokens": None if usage is None else usage.prompt_tokens,
                    "output_tokens": None if usage is None else usage.completion_tokens,
                    "reasoning_tokens": None if usage is None else usage.reasoning_tokens,
                    "actual_openrouter_cost_usd": cost,
                    "vlm_latency_seconds": latency,
                    "retry_count": 0,
                    "error": None if error is None else errors[frame_id],
                },
            )
    api_batch_seconds = perf_counter() - batch_started
    detections = [results[frame].detection for frame in sorted(results)]
    if len(detections) < 2:
        raise RuntimeError("fewer than two valid anchors; no tracker comparison can run")
    actual_cost = sum(observed_costs) if observed_costs else None
    summed_latency = sum(observed_latencies) if observed_latencies else None
    baseline, improved = comparison_variants()
    baseline_record = _process_variant(
        paths,
        baseline,
        detections,
        VLM_CALLS,
        actual_cost,
        cost_complete,
        summed_latency,
        api_batch_seconds,
        paths.baseline_video,
        paths.baseline_metrics,
    )
    improved_record = _process_variant(
        paths,
        improved,
        detections,
        VLM_CALLS,
        actual_cost,
        cost_complete,
        summed_latency,
        api_batch_seconds,
        paths.improved_video,
        paths.improved_metrics,
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed" if not errors else "completed_with_anchor_errors",
        "model": DEFAULT_OPENROUTER_MODEL,
        "planned_vlm_calls": VLM_CALLS,
        "attempted_vlm_calls": VLM_CALLS,
        "successful_vlm_calls": len(results),
        "failed_vlm_calls": len(errors),
        "errors": errors,
        "automatic_retries": 0,
        "actual_api_cost_usd": actual_cost,
        "api_cost_complete": cost_complete,
        "api_batch_seconds": api_batch_seconds,
        "baseline": asdict(baseline_record),
        "improved": asdict(improved_record),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    paths.run_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--approved-call-count", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_comparison(args.repository_root).preflight)
    else:
        print(
            run_approved_comparison(
                args.repository_root, args.approved_call_count
            ).run_summary
        )


if __name__ == "__main__":
    main()
