"""Prepare and, only after exact approval, run the 1 FPS anchor experiment."""

import argparse
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from .ball_tracking import VLMInitializedBallTracker, apply_ball_track
from .comparison import cv_improved_variant
from .experiments import TrackingMetricsRecord, TrackingMetricsWriter
from .fullclip import full_clip_detection_prompt
from .player_cv_tracking import (
    VLMInitializedPlayerCVTracker,
    camera_log_as_dict,
    scene_cut_log_as_dict,
)
from .provider import OpenRouterResult, OpenRouterValidationError, OpenRouterVLMProvider
from .ruler import add_normalized_rulers
from .sampling import sample_frame_indices
from .schema import SCHEMA_VERSION, FrameDetection, frame_detection_json_schema
from .shot_context import trackable_ball_anchors
from .test1 import _detection_as_dict
from .tracking import build_player_tracker
from .video import extract_video_frames, probe_video, render_tracked_video


EXPERIMENT_ID = "test-5-1fps-player-cv"
MODEL = "google/gemini-3.1-flash-lite"
FRAME_INTERVAL = 30
ANCHOR_FRAMES = tuple(sample_frame_indices(900, FRAME_INTERVAL, include_last=True))
VLM_CALLS = len(ANCHOR_FRAMES)
INPUT_PRICE_PER_MILLION = 0.25
OUTPUT_PRICE_PER_MILLION = 1.50
MAX_OUTPUT_TOKENS = 4096
MAX_CONCURRENCY = 8


def experiment5_paths(repository_root: str | Path) -> dict[str, Path]:
    root = Path(repository_root).resolve()
    experiment = root / "experiments" / "test_5_1fps_player_cv"
    artifacts = experiment / "artifacts" / "anchors"
    return {
        "root": root,
        "clip": root / "clips" / "dev" / "spurs_thunder_test.mp4",
        "experiment": experiment,
        "anchors": artifacts,
        "preflight": experiment / "preflight.json",
        "prompt": experiment / "prompt_template_v1.txt",
        "schema": experiment / "schema_v2.json",
        "prior_runs": root / "experiments" / "test_3_2fps_comparison" / "anchor_runs.jsonl",
        "responses": experiment / "artifacts" / "responses",
        "runs": experiment / "anchor_runs.jsonl",
        "metrics": experiment / "metrics.json",
        "summary": experiment / "summary.json",
        "camera_log": experiment / "camera_motion.jsonl",
        "cut_log": experiment / "scene_cuts.jsonl",
        "confidence_log": experiment / "track_confidence.jsonl",
        "output": root / "outputs" / "spurs_thunder_test5_1fps_player_cv.mp4",
    }


def _measured_costs(path: Path) -> list[float]:
    return [
        row["actual_openrouter_cost_usd"]
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row.get("actual_openrouter_cost_usd") is not None
    ]


def prepare_experiment5(repository_root: str | Path) -> Path:
    paths = experiment5_paths(repository_root)
    info = probe_video(paths["clip"])
    if info.frame_count != 900 or abs(info.fps - 30.0) > 0.01:
        raise ValueError("1 FPS experiment requires the verified 900-frame clip")
    originals = extract_video_frames(paths["clip"], ANCHOR_FRAMES, paths["anchors"])
    for frame_id, original_path in originals.items():
        ruler_path = paths["anchors"] / f"frame_{frame_id:04d}_ruler.png"
        if ruler_path.is_file():
            continue
        with Image.open(original_path) as image:
            add_normalized_rulers(
                image.convert("RGB"), margin_px=64, tick_step=0.1
            ).save(ruler_path, format="PNG", optimize=True)
    paths["experiment"].mkdir(parents=True, exist_ok=True)
    paths["prompt"].write_text(
        full_clip_detection_prompt(0).replace("must be 0", "must be {frame_id}") + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(frame_detection_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    costs = _measured_costs(paths["prior_runs"])
    if len(costs) != 61:
        raise RuntimeError("expected 61 measured Test 3 costs")
    prior_average = sum(costs) / len(costs)
    high_per_anchor = (
        2588 * INPUT_PRICE_PER_MILLION / 1_000_000
        + MAX_OUTPUT_TOKENS * OUTPUT_PRICE_PER_MILLION / 1_000_000
    )
    estimate = {
        "low_usd": min(costs) * VLM_CALLS,
        "expected_usd": prior_average * VLM_CALLS,
        "high_usd": high_per_anchor * VLM_CALLS,
        "prior_measured_average_cost_per_anchor_usd": prior_average,
        "prior_measured_total_61_anchor_cost_usd": sum(costs),
        "basis": (
            "Low uses the minimum observed Test 3 anchor cost; expected uses the complete "
            "61-call measured average; high prices the observed 2,588 input-token load "
            "plus the configured 4,096-token output ceiling for every call."
        ),
    }
    preflight = {
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared_not_approved",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_clip": str(paths["clip"].relative_to(paths["root"])),
        "clip": asdict(info),
        "model": MODEL,
        "frame_interval": FRAME_INTERVAL,
        "sampling_fps": info.fps / FRAME_INTERVAL,
        "anchor_frames": list(ANCHOR_FRAMES),
        "exact_planned_call_count": VLM_CALLS,
        "current_pricing_snapshot": {
            "captured_at": "2026-09-06",
            "input_per_million_tokens_usd": INPUT_PRICE_PER_MILLION,
            "output_per_million_tokens_usd": OUTPUT_PRICE_PER_MILLION,
            "source": "https://openrouter.ai/google/gemini-3.1-flash-lite/pricing",
        },
        "cost_estimate": estimate,
        "call_reduction_from_61": 61 - VLM_CALLS,
        "expected_fractional_call_and_cost_savings": 1 - VLM_CALLS / 61,
        "purpose": (
            "Test whether the new local player CV tracker can retain visual quality while "
            "reducing VLM anchors from 61 at 2 FPS to 31 at about 1 FPS."
        ),
        "changed_variable": "VLM anchor interval: 15 frames to 30 frames",
        "held_constant": [
            "clip",
            "model",
            "ruler",
            "prompt",
            "schema",
            "player CV tracker configuration",
            "camera compensation",
            "scene-cut handling",
            "track confidence",
            "ball tracker",
        ],
        "paid_requests_attempted": 0,
        "approval_gate": f"STOP: requires explicit approval for exactly {VLM_CALLS} paid calls.",
        "schema_version": SCHEMA_VERSION,
    }
    paths["preflight"].write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths["preflight"]


def _request_one(
    provider: OpenRouterVLMProvider, paths: dict[str, Path], frame_id: int
) -> OpenRouterResult:
    ruler_path = paths["anchors"] / f"frame_{frame_id:04d}_ruler.png"
    with Image.open(ruler_path) as image:
        return provider.request_detection(
            frame_id,
            image.copy(),
            prompt=full_clip_detection_prompt(frame_id),
        )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_approved_experiment5(
    repository_root: str | Path, approved_call_count: int
) -> Path:
    """Make exactly 31 approved, non-retried calls and render the fixed CV pipeline."""

    if approved_call_count != VLM_CALLS:
        raise PermissionError(
            f"experiment requires explicit approval for exactly {VLM_CALLS} calls"
        )
    paths = experiment5_paths(repository_root)
    if paths["runs"].is_file() and paths["runs"].read_text(encoding="utf-8").strip():
        raise PermissionError("call records already exist; refusing a duplicate paid run")

    prepare_experiment5(repository_root)
    provider = OpenRouterVLMProvider.from_repository_env(
        repository_root,
        model=MODEL,
        reasoning_effort="minimal",
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    preflight.update(
        {
            "status": "approved_run_started",
            "paid_requests_attempted": VLM_CALLS,
            "approved_call_count": approved_call_count,
            "automatic_retries": 0,
            "max_concurrency": MAX_CONCURRENCY,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    paths["preflight"].write_text(
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
                paths["responses"].mkdir(parents=True, exist_ok=True)
                response_path = paths["responses"] / f"frame_{frame_id:04d}_response.json"
                response_path.write_text(
                    json.dumps(
                        _detection_as_dict(result.detection), indent=2, sort_keys=True
                    )
                    + "\n",
                    encoding="utf-8",
                )
                usage, latency = result.usage, result.latency_seconds
            except Exception as caught:
                error = caught
                errors[frame_id] = f"{type(caught).__name__}: {caught}"
                if isinstance(caught, OpenRouterValidationError):
                    usage, latency = caught.usage, caught.latency_seconds
                    if caught.raw_content:
                        paths["responses"].mkdir(parents=True, exist_ok=True)
                        invalid_path = paths["responses"] / f"frame_{frame_id:04d}_invalid.txt"
                        invalid_path.write_text(caught.raw_content, encoding="utf-8")
                else:
                    usage, latency = None, None
            cost = None if usage is None else usage.cost_usd
            if cost is None:
                cost_complete = False
            else:
                observed_costs.append(cost)
            if latency is not None:
                observed_latencies.append(latency)
            _append_jsonl(
                paths["runs"],
                {
                    "experiment_id": EXPERIMENT_ID,
                    "frame_id": frame_id,
                    "model": MODEL,
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
    detections: dict[int, FrameDetection] = {
        frame_id: result.detection for frame_id, result in results.items()
    }
    if len(detections) < 2:
        raise RuntimeError("fewer than two valid anchors; local tracking cannot run")

    variant = cv_improved_variant()
    variant = replace(
        variant,
        pipeline=replace(
            variant.pipeline,
            sampling=replace(variant.pipeline.sampling, every_n_frames=FRAME_INTERVAL),
        ),
    )
    local_started = perf_counter()
    tracker = build_player_tracker(variant.pipeline.tracking)
    player_started = perf_counter()
    player_result = VLMInitializedPlayerCVTracker(
        variant.pipeline, tracker
    ).track_video(paths["clip"], detections)
    player_pass_seconds = perf_counter() - player_started
    cut_frames = {item.frame_id for item in player_result.scene_cuts if item.is_cut}
    timeline = list(player_result.timeline)

    ball_started = perf_counter()
    ball_result = VLMInitializedBallTracker(variant.pipeline.ball_tracking).track_video(
        paths["clip"],
        trackable_ball_anchors(detections, player_result.shot_context),
        timeline,
        scene_cut_frames=cut_frames,
    )
    ball_seconds = perf_counter() - ball_started
    timeline = apply_ball_track(timeline, ball_result)
    render_started = perf_counter()
    render_tracked_video(paths["clip"], paths["output"], timeline, show_ids=True)
    render_seconds = perf_counter() - render_started
    local_seconds = perf_counter() - local_started

    _write_jsonl(paths["camera_log"], camera_log_as_dict(player_result))
    _write_jsonl(paths["cut_log"], scene_cut_log_as_dict(player_result))
    _write_jsonl(paths["confidence_log"], player_result.confidence_history)

    stats = player_result.stats
    lifecycle = tracker.lifecycle
    actual_cost = sum(observed_costs) if observed_costs else None
    summed_latency = sum(observed_latencies) if observed_latencies else None
    record = TrackingMetricsRecord(
        experiment_id=EXPERIMENT_ID,
        variant=variant.name,
        model=MODEL,
        anchor_count=VLM_CALLS,
        planned_vlm_calls=VLM_CALLS,
        attempted_vlm_calls=VLM_CALLS,
        successful_vlm_calls=len(detections),
        actual_api_cost_usd=actual_cost,
        api_cost_complete=cost_complete,
        summed_vlm_latency_seconds=summed_latency,
        api_batch_seconds=api_batch_seconds,
        local_processing_seconds=local_seconds,
        total_processing_seconds=api_batch_seconds + local_seconds,
        unique_ids_over_time={
            frame.frame_id: tuple(player.track_id for player in frame.players)
            for frame in timeline
            if frame.frame_id in detections
        },
        tracks_created=lifecycle.tracks_created,
        tracks_expired=lifecycle.tracks_expired,
        tracks_lost=lifecycle.tracks_lost,
        tracks_recovered=lifecycle.tracks_recovered,
        ball_lost_events=ball_result.stats.lost_events,
        ball_recovered_events=ball_result.stats.recovered_events,
        ball_lost_frames=ball_result.stats.lost_frames,
        uncertain_track_frames=stats.uncertain_track_frames,
        uncertainty_recoveries=stats.uncertainty_recoveries,
        scene_cut_frames=tuple(sorted(cut_frames)),
        camera_motion_successes=sum(item.success for item in player_result.camera_motion),
        camera_motion_failures=sum(
            item.frame_id > 0 and not item.success for item in player_result.camera_motion
        ),
        camera_motion_seconds=stats.camera_seconds,
        player_cv_tracking_seconds=stats.player_tracking_seconds,
        player_cv_total_pass_seconds=player_pass_seconds,
        scene_cut_detection_seconds=stats.scene_cut_seconds,
        ball_tracking_seconds=ball_seconds,
        rendering_seconds=render_seconds,
        average_local_seconds_per_frame=local_seconds / len(timeline),
    )
    TrackingMetricsWriter(paths["metrics"]).write(record)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed" if not errors else "completed_with_anchor_errors",
        "model": MODEL,
        "frame_interval": FRAME_INTERVAL,
        "planned_vlm_calls": VLM_CALLS,
        "attempted_vlm_calls": VLM_CALLS,
        "successful_vlm_calls": len(detections),
        "failed_vlm_calls": len(errors),
        "errors": errors,
        "automatic_retries": 0,
        "actual_api_cost_usd": actual_cost,
        "api_cost_complete": cost_complete,
        "summed_vlm_latency_seconds": summed_latency,
        "api_batch_seconds": api_batch_seconds,
        "metrics": asdict(record),
        "output_video": str(paths["output"].relative_to(paths["root"])),
        "machine_metrics_are_not_manual_id_switch_counts": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight.update(
        {
            "status": summary["status"],
            "successful_vlm_calls": len(detections),
            "failed_vlm_calls": len(errors),
            "actual_api_cost_usd": actual_cost,
            "completed_at": summary["completed_at"],
        }
    )
    paths["preflight"].write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--approved-call-count", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_experiment5(args.repository_root))
    else:
        print(run_approved_experiment5(args.repository_root, args.approved_call_count))


if __name__ == "__main__":
    main()
