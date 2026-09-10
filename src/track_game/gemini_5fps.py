"""Prepare and run the approved Gemini 3.8/3.7 Flash comparison at 5 FPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from .ball_tracking import VLMInitializedBallTracker, apply_ball_track
from .comparison import cv_improved_variant
from .fullclip import full_clip_detection_prompt
from .model_bakeoff import (
    BakeoffModel,
    _atomic_write_json,
    _materialize_model_artifacts,
    _percentile_95,
    _read_jsonl,
    _request_record,
    _sha256,
    _utc_now,
    _write_json,
    fetch_and_verify_catalog,
)
from .player_cv_tracking import VLMInitializedPlayerCVTracker
from .provider import OpenRouterVLMProvider
from .ruler import add_normalized_rulers
from .sampling import sample_frame_indices
from .schema import FrameDetection, frame_detection_json_schema
from .shot_context import trackable_ball_anchors
from .tracking import build_player_tracker
from .trajectory_quality import player_trajectory_quality
from .video import (
    extract_video_frames,
    mute_video_in_place,
    probe_video,
    render_tracked_video,
    video_has_audio,
)


EXPERIMENT_ID = "gemini-flash-5fps-comparison"
COMMAND_MODULE = "track_game.gemini_5fps"
FRAME_INTERVAL = 6
SAMPLING_FPS = 5.0
ANCHOR_FRAMES = tuple(sample_frame_indices(900, FRAME_INTERVAL, include_last=True))
ANCHORS_PER_MODEL = len(ANCHOR_FRAMES)
MAX_CONCURRENCY = 8
MAX_OUTPUT_TOKENS = 4096
AUTOMATIC_RETRIES = 0
REQUEST_TIMEOUT_SECONDS = 120.0

MODELS = (
    BakeoffModel(
        "gemini_3_8_flash",
        "google/gemini-3.8-flash",
        "google-ai-studio",
        include_reasoning_parameter=True,
        reasoning_setting={"effort": "low", "exclude": True},
    ),
    BakeoffModel(
        "gemini_3_7_flash",
        "google/gemini-3.7-flash",
        "google-ai-studio",
        include_reasoning_parameter=True,
        reasoning_setting={"effort": "low", "exclude": True},
    ),
)
PAID_CALLS = ANCHORS_PER_MODEL * len(MODELS)
HISTORICAL_RECORD_DIRS = {
    "gemini_3_8_flash": "gemini_3_8_flash",
    "gemini_3_7_flash": "gemini_3_7_flash_retry_120s",
}


@dataclass(frozen=True)
class FiveFpsPaths:
    root: Path
    clip: Path
    experiment: Path
    anchors: Path
    prompt: Path
    schema: Path
    catalog_snapshot: Path
    cost_preflight: Path
    manifest: Path
    approval: Path
    results_table: Path
    output_dir: Path

    def model_dir(self, model: BakeoffModel) -> Path:
        return self.experiment / model.key


def five_fps_paths(repository_root: str | Path) -> FiveFpsPaths:
    root = Path(repository_root).resolve()
    experiment = root / "experiments" / "gemini_5fps_comparison"
    return FiveFpsPaths(
        root=root,
        clip=root / "clips" / "dev" / "spurs_thunder_test.mp4",
        experiment=experiment,
        anchors=experiment / "artifacts" / "anchors",
        prompt=experiment / "prompt_template_v1.txt",
        schema=experiment / "schema_v2.json",
        catalog_snapshot=experiment / "catalog_snapshot.json",
        cost_preflight=experiment / "cost_preflight.json",
        manifest=experiment / "manifest.json",
        approval=experiment / "approval.json",
        results_table=experiment / "results_table.md",
        output_dir=root / "outputs" / "gemini_5fps_comparison",
    )


def _catalog_fingerprint(snapshot: dict[str, Any]) -> str:
    stable = [
        {
            "key": item["key"],
            "model": item["requested_slug"],
            "provider": item["chosen_provider_slug"],
            "pricing": item["pricing_per_token"],
            "reasoning": item["benchmark_reasoning_setting"],
        }
        for item in snapshot["models"]
    ]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _historical_usage(paths: FiveFpsPaths, model: BakeoffModel) -> list[dict[str, Any]]:
    directory = (
        paths.root
        / "experiments"
        / "model_bakeoff_2fps"
        / HISTORICAL_RECORD_DIRS[model.key]
        / "records"
    )
    rows = []
    for path in sorted(directory.glob("frame_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("input_tokens") is not None and row.get("output_tokens") is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no historical token usage found for {model.slug}")
    return rows


def build_cost_preflight(
    paths: FiveFpsPaths, catalog_snapshot: dict[str, Any]
) -> dict[str, Any]:
    by_key = {item["key"]: item for item in catalog_snapshot["models"]}
    estimates = []
    for model in MODELS:
        historical = _historical_usage(paths, model)
        inputs = [int(row["input_tokens"]) for row in historical]
        outputs = [int(row["output_tokens"]) for row in historical]
        price = by_key[model.key]["pricing_per_million"]
        input_rate = float(price["input"])
        output_rate = float(price["output"])

        def estimated_cost(input_tokens: float, output_tokens: float) -> float:
            return ANCHORS_PER_MODEL * (
                input_tokens * input_rate + output_tokens * output_rate
            ) / 1_000_000

        low = estimated_cost(min(inputs), min(outputs))
        expected = estimated_cost(statistics.mean(inputs), statistics.mean(outputs))
        high = estimated_cost(max(inputs), MAX_OUTPUT_TOKENS)
        estimates.append(
            {
                "key": model.key,
                "model": model.slug,
                "provider": by_key[model.key]["chosen_provider"],
                "provider_slug": by_key[model.key]["chosen_provider_slug"],
                "input_per_million_usd": input_rate,
                "output_per_million_usd": output_rate,
                "historical_2fps_requests": len(historical),
                "historical_mean_input_tokens_per_request": round(
                    statistics.mean(inputs), 3
                ),
                "historical_mean_output_tokens_per_request": round(
                    statistics.mean(outputs), 3
                ),
                "low_cost_usd": round(low, 9),
                "expected_cost_usd": round(expected, 9),
                "high_cost_usd": round(high, 9),
            }
        )
    return {
        "captured_at": catalog_snapshot["captured_at"],
        "currency": "USD",
        "sampling_fps": SAMPLING_FPS,
        "frame_interval": FRAME_INTERVAL,
        "anchor_count_per_model": ANCHORS_PER_MODEL,
        "paid_calls_requiring_approval": PAID_CALLS,
        "catalog_fingerprint": _catalog_fingerprint(catalog_snapshot),
        "models": estimates,
        "combined": {
            "low_usd": round(sum(row["low_cost_usd"] for row in estimates), 9),
            "expected_usd": round(
                sum(row["expected_cost_usd"] for row in estimates), 9
            ),
            "high_usd": round(sum(row["high_cost_usd"] for row in estimates), 9),
        },
        "scenario_definitions": {
            "low": f"{ANCHORS_PER_MODEL} requests priced at that model's observed 2 FPS minimum input and output tokens.",
            "expected": f"{ANCHORS_PER_MODEL} requests priced at that model's observed 2 FPS mean input and output tokens.",
            "high": f"{ANCHORS_PER_MODEL} requests priced at observed maximum input and the fixed 4,096 output-token ceiling.",
        },
        "warning": "These are planning estimates. Actual image tokenization, output length, and returned usage can vary.",
    }


def _prepare_anchor_images(paths: FiveFpsPaths) -> None:
    originals = extract_video_frames(paths.clip, ANCHOR_FRAMES, paths.anchors)
    for frame_id, original in originals.items():
        ruled = paths.anchors / f"frame_{frame_id:04d}_ruler.png"
        if ruled.is_file():
            continue
        with Image.open(original) as image:
            add_normalized_rulers(image, margin_px=64, tick_step=0.1).save(
                ruled, format="PNG", compress_level=1
            )


def prepare(repository_root: str | Path) -> FiveFpsPaths:
    """Create inputs and query public prices without making inference requests."""

    paths = five_fps_paths(repository_root)
    paths.experiment.mkdir(parents=True, exist_ok=True)
    mute_video_in_place(paths.clip)
    info = probe_video(paths.clip)
    if (
        info.frame_count != 900
        or abs(info.fps - 30.0) > 0.01
        or info.width != 1920
        or info.height != 1080
    ):
        raise RuntimeError(
            f"{SAMPLING_FPS:g} FPS comparison requires the verified 900-frame 1080p clip"
        )
    if video_has_audio(paths.clip):
        raise RuntimeError("source clip must be silent before test preparation")
    _prepare_anchor_images(paths)
    prompt_text = (
        full_clip_detection_prompt(0).replace("must be 0", "must be {frame_id}") + "\n"
    )
    paths.prompt.write_text(prompt_text, encoding="utf-8")
    paths.schema.write_text(
        json.dumps(frame_detection_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    catalog = fetch_and_verify_catalog(models=MODELS)
    preflight = build_cost_preflight(paths, catalog)
    _write_json(paths.catalog_snapshot, catalog)
    _write_json(paths.cost_preflight, preflight)
    snapshot_by_key = {item["key"]: item for item in catalog["models"]}
    variant = cv_improved_variant()
    for model in MODELS:
        model_dir = paths.model_dir(model)
        for name in ("records", "raw_responses", "parsed_detections"):
            (model_dir / name).mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_by_key[model.key]
        _write_json(
            model_dir / "config.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "model": model.slug,
                "provider_routing": {
                    "only": [model.provider_slug],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                "verified_provider": snapshot["chosen_provider"],
                "verified_pricing_per_million": snapshot["pricing_per_million"],
                "anchor_frames": list(ANCHOR_FRAMES),
                "sampling_fps": SAMPLING_FPS,
                "frame_interval": FRAME_INTERVAL,
                "temperature": 0,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "automatic_retries": AUTOMATIC_RETRIES,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                "max_concurrency": MAX_CONCURRENCY,
                "reasoning_setting": model.reasoning_setting,
                "player_and_ball_pipeline": variant.pipeline.as_dict(),
                "output_video": str(
                    (paths.output_dir / f"{model.key}.mp4").relative_to(paths.root)
                ),
            },
        )
    _write_json(
        paths.manifest,
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "prepared_not_approved",
            "prepared_at": _utc_now(),
            "source_clip": str(paths.clip.relative_to(paths.root)),
            "source_sha256": _sha256(paths.clip),
            "source_audio_muted": True,
            "sampling_fps": SAMPLING_FPS,
            "frame_interval": FRAME_INTERVAL,
            "anchor_frames": list(ANCHOR_FRAMES),
            "anchors_per_model": ANCHORS_PER_MODEL,
            "models": [asdict(model) for model in MODELS],
            "paid_calls_requiring_approval": PAID_CALLS,
            "cost_preflight": str(paths.cost_preflight.relative_to(paths.root)),
            "catalog_fingerprint": preflight["catalog_fingerprint"],
            "prompt_sha256": _sha256(paths.prompt),
            "schema_sha256": _sha256(paths.schema),
            "new_pipeline_changes": [
                "Hungarian multi-cue ball-anchor gating and confidence fusion",
                "larger filled supersampled player and ball markers",
            ],
            "exact_run_command": (
                f".venv\\Scripts\\python.exe -m {COMMAND_MODULE} run "
                f"--approved-call-count {PAID_CALLS}"
            ),
            "approval_gate": (
                f"STOP: requires explicit approval for exactly {PAID_CALLS} paid calls."
            ),
        },
    )
    return paths


def _record_path(paths: FiveFpsPaths, model: BakeoffModel, frame_id: int) -> Path:
    return paths.model_dir(model) / "records" / f"frame_{frame_id:04d}.json"


def _load_records(paths: FiveFpsPaths, model: BakeoffModel) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((paths.model_dir(model) / "records").glob("frame_*.json"))
    ]


def _run_model(
    paths: FiveFpsPaths, model: BakeoffModel
) -> tuple[list[dict[str, Any]], float]:
    existing = _load_records(paths, model)
    completed = {row["frame_id"] for row in existing}
    remaining = [frame_id for frame_id in ANCHOR_FRAMES if frame_id not in completed]
    if not remaining:
        return existing, sum(
            row.get("wall_seconds", 0.0)
            for row in _read_jsonl(paths.model_dir(model) / "batches.jsonl")
        )
    provider = OpenRouterVLMProvider.from_repository_env(
        paths.root,
        model=model.slug,
        reasoning_effort="minimal",
        reasoning_setting=model.reasoning_setting,
        include_reasoning_parameter=model.include_reasoning_parameter,
        provider_preferences={
            "only": [model.provider_slug],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )
    started = perf_counter()
    futures: dict[Future[dict[str, Any]], int] = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        for frame_id in remaining:
            future = executor.submit(_request_record, provider, paths, model, frame_id)
            futures[future] = frame_id
        for future in as_completed(futures):
            frame_id = futures[future]
            record = future.result()
            record["experiment_id"] = EXPERIMENT_ID
            _atomic_write_json(_record_path(paths, model, frame_id), record)
    segment_seconds = perf_counter() - started
    batch_path = paths.model_dir(model) / "batches.jsonl"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with batch_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "started_at": _utc_now(),
                    "attempted_frames": remaining,
                    "attempted_calls": len(remaining),
                    "configured_concurrency": MAX_CONCURRENCY,
                    "wall_seconds": segment_seconds,
                    "automatic_retries": AUTOMATIC_RETRIES,
                    "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                },
                sort_keys=True,
            )
            + "\n"
        )
    records = _load_records(paths, model)
    total_seconds = sum(
        row.get("wall_seconds", 0.0) for row in _read_jsonl(batch_path)
    )
    return records, total_seconds


def _summarize(
    paths: FiveFpsPaths,
    model: BakeoffModel,
    records: list[dict[str, Any]],
    batch_seconds: float,
) -> dict[str, Any]:
    valid = [row for row in records if row.get("schema_validation_success")]
    costs = [row.get("actual_openrouter_cost_usd") for row in records]
    known_costs = [float(value) for value in costs if value is not None]
    latencies = [
        float(row["request_latency_seconds"])
        for row in records
        if row.get("request_latency_seconds") is not None
    ]
    detections = [
        FrameDetection.from_dict(row["parsed_detection"])
        for row in valid
        if row.get("parsed_detection") is not None
    ]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "model_key": model.key,
        "model": model.slug,
        "calls_planned": ANCHORS_PER_MODEL,
        "calls_attempted": len(records),
        "valid_responses": len(valid),
        "invalid_responses": len(records) - len(valid),
        "schema_valid_percentage": 100.0 * len(valid) / len(records) if records else 0.0,
        "average_player_detections": (
            statistics.mean(len(item.players) for item in detections) if detections else 0.0
        ),
        "ball_returned_anchors": sum(item.ball_detection is not None for item in detections),
        "possession_returned_anchors": sum(item.possession is not None for item in detections),
        "known_cost_usd": sum(known_costs),
        "cost_complete": len(known_costs) == len(records),
        "actual_total_cost_usd": sum(known_costs) if len(known_costs) == len(records) else None,
        "total_input_tokens": sum(row.get("input_tokens") or 0 for row in records),
        "total_output_tokens": sum(row.get("output_tokens") or 0 for row in records),
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile_95(latencies),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "batch_wall_seconds": batch_seconds,
        "configured_concurrency": MAX_CONCURRENCY,
        "http_429_count": sum(bool(row.get("http_429")) for row in records),
        "request_errors": sum(row.get("error") is not None for row in records),
        "automatic_retries": AUTOMATIC_RETRIES,
        "updated_at": _utc_now(),
    }
    _write_json(paths.model_dir(model) / "summary.json", summary)
    return summary


def _detections(records: list[dict[str, Any]]) -> dict[int, FrameDetection]:
    return {
        row["frame_id"]: FrameDetection.from_dict(row["parsed_detection"])
        for row in records
        if row.get("schema_validation_success") and row.get("parsed_detection") is not None
    }


def _render_pipeline(
    paths: FiveFpsPaths,
    model: BakeoffModel,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    detections = _detections(records)
    if len(detections) < 2:
        summary["tracking"] = {
            "status": "skipped",
            "reason": "fewer than two schema-valid detector anchors",
        }
        _write_json(paths.model_dir(model) / "summary.json", summary)
        return
    variant = cv_improved_variant()
    tracker = build_player_tracker(variant.pipeline.tracking)
    local_started = perf_counter()
    player_result = VLMInitializedPlayerCVTracker(
        variant.pipeline, tracker
    ).track_video(paths.clip, detections)
    cuts = {item.frame_id for item in player_result.scene_cuts if item.is_cut}
    timeline = list(player_result.timeline)
    ball_result = VLMInitializedBallTracker(variant.pipeline.ball_tracking).track_video(
        paths.clip,
        trackable_ball_anchors(detections, player_result.shot_context),
        timeline,
        scene_cut_frames=cuts,
    )
    timeline = apply_ball_track(timeline, ball_result)
    output = paths.output_dir / f"{model.key}.mp4"
    render_tracked_video(paths.clip, output, timeline, show_ids=True)
    if video_has_audio(output):
        raise RuntimeError("rendered test output unexpectedly contains audio")
    lifecycle = tracker.lifecycle
    visible_player_marker_frames = sum(len(frame.players) for frame in timeline)
    empty_player_frames = sum(not frame.players for frame in timeline)
    trajectory_quality = player_trajectory_quality(
        timeline,
        anchor_frames=ANCHOR_FRAMES,
        scene_cut_frames=cuts,
    )
    summary["output_video"] = str(output.relative_to(paths.root))
    summary["tracking"] = {
        "status": "completed",
        "local_processing_seconds": perf_counter() - local_started,
        "tracks_created": lifecycle.tracks_created,
        "tracks_expired": lifecycle.tracks_expired,
        "tracks_lost": lifecycle.tracks_lost,
        "tracks_recovered": lifecycle.tracks_recovered,
        "scene_cut_frames": sorted(cuts),
        "closeup_suppressed_anchors": player_result.stats.closeup_suppressed_anchors,
        "visible_player_marker_frames": visible_player_marker_frames,
        "empty_player_frames": empty_player_frames,
        "trajectory_quality": trajectory_quality,
        "ball_tracking": asdict(ball_result.stats),
    }
    _write_json(paths.model_dir(model) / "summary.json", summary)


def _write_results(paths: FiveFpsPaths, summaries: dict[str, dict[str, Any]]) -> None:
    lines = [
        f"# Gemini Flash {SAMPLING_FPS:g} FPS comparison",
        "",
        "| Model | Valid | Players/frame | Ball anchors | Possession anchors | Cost | Mean latency | Batch time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        summary = summaries.get(model.key, {})
        latency = (summary.get("latency_seconds") or {}).get("mean")
        cost = summary.get("actual_total_cost_usd")
        lines.append(
            f"| {model.slug} | {summary.get('valid_responses', 0)}/{summary.get('calls_attempted', 0)} "
            f"| {summary.get('average_player_detections', 0):.3f} "
            f"| {summary.get('ball_returned_anchors', 0)} "
            f"| {summary.get('possession_returned_anchors', 0)} "
            f"| {'' if cost is None else f'${cost:.6f}'} "
            f"| {'' if latency is None else f'{latency:.3f}s'} "
            f"| {summary.get('batch_wall_seconds', 0):.3f}s |"
        )
    paths.results_table.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(repository_root: str | Path, approved_call_count: int) -> FiveFpsPaths:
    if approved_call_count != PAID_CALLS:
        raise PermissionError(
            f"{SAMPLING_FPS:g} FPS comparison requires explicit approval for exactly {PAID_CALLS} calls"
        )
    paths = five_fps_paths(repository_root)
    if not paths.manifest.is_file() or not paths.catalog_snapshot.is_file():
        raise RuntimeError("run prepare and review its cost preflight before approval")
    stored_catalog = json.loads(paths.catalog_snapshot.read_text(encoding="utf-8"))
    live_catalog = fetch_and_verify_catalog(models=MODELS)
    if _catalog_fingerprint(live_catalog) != _catalog_fingerprint(stored_catalog):
        raise PermissionError("provider or pricing changed; prepare a new preflight and reapprove")
    if video_has_audio(paths.clip):
        raise RuntimeError("source clip is not silent")
    if paths.approval.is_file():
        approval = json.loads(paths.approval.read_text(encoding="utf-8"))
        if approval.get("approved_call_count") != PAID_CALLS:
            raise PermissionError("stored approval has the wrong call count")
    else:
        _write_json(
            paths.approval,
            {
                "approved_at": _utc_now(),
                "approved_call_count": PAID_CALLS,
                "catalog_fingerprint": _catalog_fingerprint(stored_catalog),
            },
        )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["status"] = "approved_run_started"
    manifest["started_at"] = _utc_now()
    _write_json(paths.manifest, manifest)
    summaries = {}
    for model in MODELS:
        records, batch_seconds = _run_model(paths, model)
        _materialize_model_artifacts(paths, model, records)
        summary = _summarize(paths, model, records, batch_seconds)
        _render_pipeline(paths, model, records, summary)
        summaries[model.key] = summary
    _write_results(paths, summaries)
    manifest["status"] = "completed"
    manifest["completed_at"] = _utc_now()
    manifest["paid_inference_calls_attempted"] = sum(
        summary["calls_attempted"] for summary in summaries.values()
    )
    manifest["outputs"] = [summary.get("output_video") for summary in summaries.values()]
    _write_json(paths.manifest, manifest)
    return paths


def render_saved(repository_root: str | Path) -> FiveFpsPaths:
    """Re-render completed responses locally without constructing a provider."""

    paths = five_fps_paths(repository_root)
    summaries: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        records = _load_records(paths, model)
        if len(records) != ANCHORS_PER_MODEL:
            raise RuntimeError(
                f"{model.slug} has {len(records)}/{ANCHORS_PER_MODEL} saved records"
            )
        summary_path = paths.model_dir(model) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _render_pipeline(paths, model, records, summary)
        summaries[model.key] = summary
    _write_results(paths, summaries)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "render-saved"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--approved-call-count", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.repository_root).cost_preflight)
    elif args.command == "run":
        print(run(args.repository_root, args.approved_call_count).results_table)
    else:
        print(render_saved(args.repository_root).results_table)


if __name__ == "__main__":
    main()
