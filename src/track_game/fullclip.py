"""Prepare and run the first sampled-VLM experiment over the 30-second clip."""

import argparse
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from .experiments import ExperimentRecord, JsonlExperimentLogger
from .provider import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterResult,
    OpenRouterValidationError,
    OpenRouterVLMProvider,
    detection_prompt,
)
from .ruler import add_normalized_rulers
from .sampling import sample_frame_indices
from .schema import SCHEMA_VERSION, FrameDetection, frame_detection_json_schema
from .test1 import _detection_as_dict
from .tracking import NearestNeighbourTracker, interpolate_sequence
from .video import extract_video_frame, probe_video, render_tracked_video


EXPERIMENT_ID = "test-2-full-clip-3s-anchors"
FRAME_INTERVAL = 90
ANCHOR_FRAMES = tuple(sample_frame_indices(900, FRAME_INTERVAL, include_last=True))
VLM_CALLS = len(ANCHOR_FRAMES)
MAX_CONCURRENCY = 11
REASONING_SETTING = {"effort": "minimal", "exclude": True}
MAX_OUTPUT_TOKENS = 4096
PROMPT_VERSION = "fullclip-ruler-v2-closeup-filter"

PRICING_SNAPSHOT: dict[str, Any] = {
    "captured_at": "2026-09-03T02:59:35.0119534Z",
    "model": DEFAULT_OPENROUTER_MODEL,
    "currency": "USD",
    "listed_input_per_million_tokens": 0.25,
    "listed_image_input_per_million_tokens": 0.25,
    "listed_output_per_million_tokens": 1.50,
    "source": "https://openrouter.ai/google/gemini-3.1-flash-lite/pricing",
}

COST_ESTIMATE = {
    "low_usd": 0.0154,
    "expected_usd": 0.0242,
    "high_usd": 0.0880,
    "assumptions": {
        "vlm_calls": VLM_CALLS,
        "input_tokens_per_call_expected": 1403,
        "output_tokens_per_call_expected": 1235,
        "basis": "Test 1 observed usage for the same resolution, prompt, schema, and model.",
        "note": "Approximate: scene complexity and provider tokenization can change output usage.",
    },
}

LATENCY_ESTIMATE = {
    "test1_vlm_request_seconds": 5.86,
    "measured_local_render_seconds": 16.18,
    "expected_end_to_end_seconds": 24.0,
    "approximate_range_seconds": [22.0, 30.0],
    "note": "All 11 independent anchors use one bounded parallel wave; provider queueing may vary.",
}


@dataclass(frozen=True)
class FullClipPaths:
    root: Path
    clip: Path
    experiment_dir: Path
    anchors_dir: Path
    responses_dir: Path
    prompt_template: Path
    schema: Path
    manifest: Path
    metrics: Path
    summary: Path
    output_video: Path


def full_clip_paths(repository_root: str | Path) -> FullClipPaths:
    root = Path(repository_root).resolve()
    experiment_dir = root / "experiments" / "test_2_full_clip"
    artifacts = experiment_dir / "artifacts"
    return FullClipPaths(
        root=root,
        clip=root / "clips" / "dev" / "spurs_thunder_test.mp4",
        experiment_dir=experiment_dir,
        anchors_dir=artifacts / "anchors",
        responses_dir=artifacts / "responses",
        prompt_template=experiment_dir / "prompt_template_v1.txt",
        schema=experiment_dir / "schema_v2.json",
        manifest=experiment_dir / "preflight.json",
        metrics=experiment_dir / "anchor_runs.jsonl",
        summary=experiment_dir / "summary.json",
        output_video=root / "outputs" / "spurs_thunder_test2_annotated.mp4",
    )


def _original_path(paths: FullClipPaths, frame_id: int) -> Path:
    return paths.anchors_dir / f"frame_{frame_id:04d}_original.png"


def _ruler_path(paths: FullClipPaths, frame_id: int) -> Path:
    return paths.anchors_dir / f"frame_{frame_id:04d}_ruler.png"


def _update_json(path: Path, **updates: Any) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def full_clip_detection_prompt(frame_id: int) -> str:
    return detection_prompt(frame_id) + """

Clip-wide team convention required for temporal consistency:
- Team A: Oklahoma City players wearing blue uniforms.
- Team B: San Antonio players wearing black, white, or gray uniforms.
Keep this mapping even when a uniform is partly occluded. Referees are not players."""


def prepare_full_clip_test(repository_root: str | Path) -> FullClipPaths:
    """Extract anchor/ruler images and metadata without constructing a provider."""

    paths = full_clip_paths(repository_root)
    info = probe_video(paths.clip)
    if (
        info.frame_count != 900
        or abs(info.duration_seconds - 30.0) > 0.02
        or abs(info.fps - 30.0) > 0.01
    ):
        raise ValueError("full-clip test requires the verified 900-frame, 30 FPS clip")
    paths.anchors_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in ANCHOR_FRAMES:
        original_path = _original_path(paths, frame_id)
        ruler_path = _ruler_path(paths, frame_id)
        if original_path.is_file() and ruler_path.is_file():
            with Image.open(original_path) as existing_original:
                if existing_original.size != (info.width, info.height):
                    raise ValueError(f"wrong original anchor size for frame {frame_id}")
            with Image.open(ruler_path) as existing_ruler:
                if existing_ruler.size != (info.width + 64, info.height + 64):
                    raise ValueError(f"wrong ruler anchor size for frame {frame_id}")
        else:
            original = extract_video_frame(paths.clip, frame_id, original_path)
            ruler = add_normalized_rulers(original, margin_px=64, tick_step=0.1)
            ruler.save(ruler_path, format="PNG", optimize=True)
    paths.prompt_template.write_text(
        full_clip_detection_prompt(0).replace("must be 0", "must be {frame_id}")
        + "\n",
        encoding="utf-8",
    )
    paths.schema.write_text(
        json.dumps(frame_detection_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_clip": str(paths.clip.relative_to(paths.root)),
        "clip": asdict(info),
        "model": DEFAULT_OPENROUTER_MODEL,
        "anchor_frame_interval": FRAME_INTERVAL,
        "anchor_interval_seconds": FRAME_INTERVAL / info.fps,
        "anchor_frames": list(ANCHOR_FRAMES),
        "vlm_calls": VLM_CALLS,
        "max_concurrency": MAX_CONCURRENCY,
        "automatic_retries": 0,
        "ruler_grounding": True,
        "original_image_resolution": [info.width, info.height],
        "prompt_image_resolution": [info.width + 64, info.height + 64],
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "reasoning": REASONING_SETTING,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "tracker": {
            "method": "greedy nearest explicit foot point",
            "team_constraint": True,
            "max_normalized_distance": 0.2,
            "max_missed_anchors": 2,
            "interpolation": "linear",
        },
        "pricing_snapshot": PRICING_SNAPSHOT,
        "cost_estimate": COST_ESTIMATE,
        "latency_estimate": LATENCY_ESTIMATE,
        "paid_requests_attempted": 0,
        "paid_batch_succeeded": False,
        "approval_gate": f"Requires explicit approval for exactly {VLM_CALLS} calls.",
    }
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _request_one(
    provider: OpenRouterVLMProvider, paths: FullClipPaths, frame_id: int
) -> tuple[OpenRouterResult, float]:
    started = perf_counter()
    with Image.open(_ruler_path(paths, frame_id)) as image:
        result = provider.request_detection(
            frame_id, image.copy(), prompt=full_clip_detection_prompt(frame_id)
        )
    return result, perf_counter() - started


def _anchor_record(
    paths: FullClipPaths,
    frame_id: int,
    result: OpenRouterResult | None,
    wall_seconds: float | None,
    error: Exception | None,
) -> ExperimentRecord:
    detection = None if result is None else result.detection
    message = None if error is None else f"{type(error).__name__}: {error}"
    failed = error if isinstance(error, OpenRouterValidationError) else None
    usage = result.usage if result is not None else (None if failed is None else failed.usage)
    return ExperimentRecord(
        experiment_id=f"{EXPERIMENT_ID}-frame-{frame_id:04d}",
        status="completed" if result is not None else "failed",
        source_clip=str(paths.clip),
        source_timestamp_seconds=frame_id / 30.0,
        source_frame_number=frame_id,
        model=DEFAULT_OPENROUTER_MODEL,
        pricing_snapshot=PRICING_SNAPSHOT,
        image_resolution=(1920, 1080),
        prompt_image_resolution=(1984, 1144),
        ruler_grounding=True,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        reasoning_setting=REASONING_SETTING,
        vlm_calls=1,
        provider_request_id=result.request_id if result is not None else (None if failed is None else failed.request_id),
        returned_model=result.returned_model if result is not None else (None if failed is None else failed.returned_model),
        returned_provider=result.provider if result is not None else (None if failed is None else failed.provider),
        input_tokens=None if usage is None else usage.prompt_tokens,
        output_tokens=None if usage is None else usage.completion_tokens,
        reasoning_tokens=None if usage is None else usage.reasoning_tokens,
        vlm_request_latency_seconds=(
            result.latency_seconds if result is not None else (None if failed is None else failed.latency_seconds)
        ),
        total_wall_clock_seconds=wall_seconds,
        retries=0,
        errors=() if message is None else (message,),
        actual_openrouter_cost_usd=(
            None if usage is None else usage.cost_usd
        ),
        estimated_cost_low_usd=COST_ESTIMATE["low_usd"] / VLM_CALLS,
        estimated_cost_expected_usd=COST_ESTIMATE["expected_usd"] / VLM_CALLS,
        estimated_cost_high_usd=COST_ESTIMATE["high_usd"] / VLM_CALLS,
        schema_validation_success=(True if result is not None else (False if failed is not None else None)),
        players_returned=None if detection is None else len(detection.players),
        ball_detected=None if detection is None else detection.ball_detection is not None,
        possession_returned=None if detection is None else detection.possession is not None,
        anchor_frame_sampling_interval=FRAME_INTERVAL,
        tracker_data_association_method="greedy nearest explicit foot point, team constrained",
        vlm_redetection_frequency=FRAME_INTERVAL,
    )


def run_approved_full_clip_test(
    repository_root: str | Path, approved_call_count: int
) -> FullClipPaths:
    """Make the approved 11 calls once, then track and render entirely locally."""

    if approved_call_count != VLM_CALLS:
        raise PermissionError(
            f"full-clip test requires explicit approval for exactly {VLM_CALLS} calls"
        )
    paths = full_clip_paths(repository_root)
    if paths.metrics.exists() and paths.metrics.read_text(encoding="utf-8").strip():
        raise PermissionError("this batch already has call records; refusing a duplicate run")
    paths = prepare_full_clip_test(repository_root)
    provider = OpenRouterVLMProvider.from_repository_env(
        repository_root,
        model=DEFAULT_OPENROUTER_MODEL,
        reasoning_effort=REASONING_SETTING["effort"],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    _update_json(
        paths.manifest,
        paid_requests_attempted=VLM_CALLS,
        batch_attempted_at=datetime.now(timezone.utc).isoformat(),
    )
    batch_started = perf_counter()
    results: dict[int, OpenRouterResult] = {}
    errors: dict[int, str] = {}
    futures: dict[Future[tuple[OpenRouterResult, float]], int] = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        for frame_id in ANCHOR_FRAMES:
            futures[executor.submit(_request_one, provider, paths, frame_id)] = frame_id
        for future in as_completed(futures):
            frame_id = futures[future]
            try:
                result, wall_seconds = future.result()
                results[frame_id] = result
                record = _anchor_record(paths, frame_id, result, wall_seconds, None)
                response_path = paths.responses_dir / f"frame_{frame_id:04d}_response.json"
                response_path.parent.mkdir(parents=True, exist_ok=True)
                response_path.write_text(
                    json.dumps(_detection_as_dict(result.detection), indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
            except Exception as error:
                errors[frame_id] = f"{type(error).__name__}: {error}"
                if isinstance(error, OpenRouterValidationError) and error.raw_content:
                    invalid_path = paths.responses_dir / f"frame_{frame_id:04d}_invalid.txt"
                    invalid_path.parent.mkdir(parents=True, exist_ok=True)
                    invalid_path.write_text(error.raw_content, encoding="utf-8")
                record = _anchor_record(paths, frame_id, None, None, error)
            JsonlExperimentLogger(paths.metrics).append(record)
    api_batch_seconds = perf_counter() - batch_started
    if errors:
        summary = {
            "experiment_id": EXPERIMENT_ID,
            "status": "failed",
            "vlm_calls_attempted": VLM_CALLS,
            "successful_calls": len(results),
            "failed_calls": len(errors),
            "errors": errors,
            "api_batch_seconds": api_batch_seconds,
            "retries": 0,
        }
        paths.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _update_json(
            paths.manifest,
            paid_batch_succeeded=False,
            batch_finished_at=datetime.now(timezone.utc).isoformat(),
        )
        raise RuntimeError("one or more anchor requests failed; no retry was made")

    tracker = NearestNeighbourTracker(
        max_distance=0.2, team_constraint=True, max_missed=2
    )
    tracked_anchors = [tracker.update(results[frame].detection) for frame in ANCHOR_FRAMES]
    timeline = interpolate_sequence(tracked_anchors)
    render_started = perf_counter()
    video_info = render_tracked_video(paths.clip, paths.output_video, timeline, show_ids=True)
    render_seconds = perf_counter() - render_started
    total_seconds = perf_counter() - batch_started
    costs = [result.usage.cost_usd for result in results.values()]
    total_cost = None if any(cost is None for cost in costs) else sum(costs)  # type: ignore[arg-type]
    input_tokens = [result.usage.prompt_tokens for result in results.values()]
    output_tokens = [result.usage.completion_tokens for result in results.values()]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "model": DEFAULT_OPENROUTER_MODEL,
        "source_clip": str(paths.clip),
        "output_video": str(paths.output_video),
        "anchor_frames": list(ANCHOR_FRAMES),
        "anchor_frame_interval": FRAME_INTERVAL,
        "vlm_calls": VLM_CALLS,
        "max_concurrency": MAX_CONCURRENCY,
        "retries": 0,
        "api_batch_seconds": api_batch_seconds,
        "render_seconds": render_seconds,
        "total_processing_seconds": total_seconds,
        "actual_openrouter_cost_usd": total_cost,
        "input_tokens": None if any(v is None for v in input_tokens) else sum(input_tokens),
        "output_tokens": None if any(v is None for v in output_tokens) else sum(output_tokens),
        "schema_validation_success": True,
        "unique_track_ids": len(
            {player.track_id for anchor in tracked_anchors for player in anchor.players}
        ),
        "id_switches": None,
        "lost_tracks": None,
        "recovered_tracks": None,
        "human_evaluation": {
            "player_localization_quality": None,
            "team_assignment_quality": None,
            "ball_localization_quality": None,
            "obvious_detection_failures": None,
            "notes": None,
        },
        "output_video_info": asdict(video_info),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    paths.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _update_json(
        paths.manifest,
        paid_batch_succeeded=True,
        batch_finished_at=datetime.now(timezone.utc).isoformat(),
        actual_openrouter_cost_usd=total_cost,
        actual_total_processing_seconds=total_seconds,
    )
    return paths


def render_saved_full_clip_results(repository_root: str | Path) -> FullClipPaths:
    """Render existing valid anchor responses without making any provider call."""

    paths = full_clip_paths(repository_root)
    detections: dict[int, FrameDetection] = {}
    for response_path in paths.responses_dir.glob("frame_*_response.json"):
        data = json.loads(response_path.read_text(encoding="utf-8"))
        detection = FrameDetection.from_dict(data)
        if detection.frame_id not in ANCHOR_FRAMES:
            raise ValueError(f"unexpected saved anchor {detection.frame_id}")
        detections[detection.frame_id] = detection
    if ANCHOR_FRAMES[0] not in detections or ANCHOR_FRAMES[-1] not in detections:
        raise RuntimeError("cannot cover the full clip without the first and last anchors")
    if len(detections) < 2:
        raise RuntimeError("at least two valid anchors are required")

    ordered_frames = sorted(detections)
    tracker = NearestNeighbourTracker(
        max_distance=0.2, team_constraint=True, max_missed=2
    )
    tracked_anchors = [tracker.update(detections[frame]) for frame in ordered_frames]
    timeline = interpolate_sequence(tracked_anchors)
    render_started = perf_counter()
    video_info = render_tracked_video(paths.clip, paths.output_video, timeline, show_ids=True)
    render_seconds = perf_counter() - render_started

    rows = JsonlExperimentLogger(paths.metrics).read_all()
    known_costs = [
        row["actual_openrouter_cost_usd"]
        for row in rows
        if row.get("actual_openrouter_cost_usd") is not None
    ]
    input_tokens = [row["input_tokens"] for row in rows if row.get("input_tokens") is not None]
    output_tokens = [
        row["output_tokens"] for row in rows if row.get("output_tokens") is not None
    ]
    previous = (
        json.loads(paths.summary.read_text(encoding="utf-8"))
        if paths.summary.is_file()
        else {}
    )
    missing = sorted(set(ANCHOR_FRAMES) - set(ordered_frames))
    api_batch_seconds = previous.get("api_batch_seconds")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_with_anchor_errors",
        "model": DEFAULT_OPENROUTER_MODEL,
        "source_clip": str(paths.clip),
        "output_video": str(paths.output_video),
        "planned_anchor_frames": list(ANCHOR_FRAMES),
        "successful_anchor_frames": ordered_frames,
        "missing_anchor_frames": missing,
        "anchor_frame_interval": FRAME_INTERVAL,
        "vlm_calls_attempted": VLM_CALLS,
        "successful_calls": len(detections),
        "failed_calls": len(missing),
        "errors": previous.get("errors", {}),
        "retries": 0,
        "api_batch_seconds": api_batch_seconds,
        "render_seconds": render_seconds,
        "measured_processing_seconds": (
            None
            if not isinstance(api_batch_seconds, (int, float))
            else api_batch_seconds + render_seconds
        ),
        "known_minimum_openrouter_cost_usd": sum(known_costs),
        "openrouter_cost_is_complete": len(known_costs) == VLM_CALLS,
        "known_input_tokens": sum(input_tokens),
        "known_output_tokens": sum(output_tokens),
        "schema_validation_successful_calls": len(detections),
        "schema_validation_failed_calls": len(missing),
        "unique_track_ids": len(
            {player.track_id for anchor in tracked_anchors for player in anchor.players}
        ),
        "id_switches": None,
        "lost_tracks": None,
        "recovered_tracks": None,
        "human_evaluation": {
            "player_localization_quality": None,
            "team_assignment_quality": None,
            "ball_localization_quality": None,
            "obvious_detection_failures": None,
            "notes": None,
        },
        "output_video_info": asdict(video_info),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    paths.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _update_json(
        paths.manifest,
        paid_batch_succeeded=False,
        output_rendered_from_successful_anchors=True,
        missing_anchor_frames=missing,
        known_minimum_openrouter_cost_usd=sum(known_costs),
        openrouter_cost_is_complete=len(known_costs) == VLM_CALLS,
        measured_render_seconds=render_seconds,
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "render-saved"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--approved-call-count", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        paths = prepare_full_clip_test(args.repository_root)
        print(paths.manifest)
    elif args.command == "run":
        paths = run_approved_full_clip_test(
            args.repository_root, args.approved_call_count
        )
        print(paths.output_video)
    else:
        paths = render_saved_full_clip_results(args.repository_root)
        print(paths.output_video)


if __name__ == "__main__":
    main()
