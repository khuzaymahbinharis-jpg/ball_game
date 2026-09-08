"""Prepare and run the controlled six-model, 2 FPS VLM detector bake-off."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import httpx
from PIL import Image, ImageDraw, ImageFont

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
from .ruler import normalized_to_pixel
from .sampling import sample_frame_indices
from .schema import FrameDetection, Team, frame_detection_json_schema
from .shot_context import trackable_ball_anchors
from .test1 import _detection_as_dict
from .tracking import build_player_tracker
from .video import extract_video_frames, probe_video, render_tracked_video


EXPERIMENT_ID = "model-bakeoff-2fps"
FRAME_INTERVAL = 15
ANCHOR_FRAMES = tuple(sample_frame_indices(900, FRAME_INTERVAL, include_last=True))
ANCHORS_PER_MODEL = len(ANCHOR_FRAMES)
MAX_CONCURRENCY = 8
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0
AUTOMATIC_RETRIES = 0
CATALOG_URL = "https://openrouter.ai/api/v1/models"
REVIEW_FRAMES = (30, 180, 270, 360, 450, 570, 690, 840)
OBSERVED_INPUT_TOKENS = 157_860
OBSERVED_OUTPUT_TOKENS = 81_058
OBSERVED_MIN_INPUT_PER_REQUEST = 2_586
OBSERVED_MIN_OUTPUT_PER_REQUEST = 220
OBSERVED_HIGH_INPUT_PER_REQUEST = 2_588
OBSERVED_CONTROL_COST_USD = 0.161052


@dataclass(frozen=True)
class BakeoffModel:
    key: str
    slug: str
    provider_slug: str
    control_reuse: bool = False
    include_reasoning_parameter: bool = False
    reasoning_setting: dict[str, Any] | None = None


MODELS = (
    BakeoffModel(
        "gemini_3_1_flash_lite",
        "google/gemini-3.1-flash-lite",
        "google-ai-studio",
        control_reuse=True,
        include_reasoning_parameter=True,
        reasoning_setting={"effort": "minimal", "exclude": True},
    ),
    BakeoffModel(
        "gemini_3_8_flash",
        "google/gemini-3.8-flash",
        "google-ai-studio",
        include_reasoning_parameter=True,
        reasoning_setting={"effort": "low", "exclude": True},
    ),
    BakeoffModel(
        "qwen3_vl_30b",
        "qwen/qwen3-vl-30b-a3b-instruct",
        "alibaba",
    ),
    BakeoffModel(
        "qwen3_vl_235b",
        "qwen/qwen3-vl-235b-a22b-instruct",
        "alibaba",
    ),
    BakeoffModel(
        "seed_2_1_turbo",
        "bytedance-seed/seed-2-1-turbo",
        "seed/fp8",
        include_reasoning_parameter=True,
        reasoning_setting={"enabled": False, "exclude": True},
    ),
    BakeoffModel(
        "glm_5_3_flash",
        "z-ai/glm-5.3-flash",
        "deepinfra/fp4",
        include_reasoning_parameter=True,
        reasoning_setting={"effort": "low", "exclude": True},
    ),
)
GEMINI_3_7_MODEL = BakeoffModel(
    "gemini_3_7_flash",
    "google/gemini-3.7-flash",
    "google-ai-studio",
    include_reasoning_parameter=True,
    reasoning_setting={"effort": "low", "exclude": True},
)
GEMINI_3_7_RETRY_MODEL = BakeoffModel(
    "gemini_3_7_flash_retry_120s",
    "google/gemini-3.7-flash",
    "google-ai-studio",
    include_reasoning_parameter=True,
    reasoning_setting={"effort": "low", "exclude": True},
)
PAID_MODELS = tuple(model for model in MODELS if not model.control_reuse)
NEW_PAID_CALLS = len(PAID_MODELS) * ANCHORS_PER_MODEL


@dataclass(frozen=True)
class BakeoffPaths:
    root: Path
    clip: Path
    experiment: Path
    test3: Path
    anchors: Path
    prior_responses: Path
    prior_runs: Path
    prompt: Path
    schema: Path
    manifest: Path
    catalog_snapshot: Path
    cost_preflight: Path
    manual_review: Path
    results_table: Path
    approval: Path
    comparison_artifacts: Path
    output_dir: Path

    def model_dir(self, model: BakeoffModel) -> Path:
        return self.experiment / model.key


def bakeoff_paths(repository_root: str | Path) -> BakeoffPaths:
    root = Path(repository_root).resolve()
    experiment = root / "experiments" / "model_bakeoff_2fps"
    test3 = root / "experiments" / "test_3_2fps_comparison"
    return BakeoffPaths(
        root=root,
        clip=root / "clips" / "dev" / "spurs_thunder_test.mp4",
        experiment=experiment,
        test3=test3,
        anchors=test3 / "artifacts" / "anchors",
        prior_responses=test3 / "artifacts" / "responses",
        prior_runs=test3 / "anchor_runs.jsonl",
        prompt=experiment / "prompt_template_v1.txt",
        schema=experiment / "schema_v2.json",
        manifest=experiment / "manifest.json",
        catalog_snapshot=experiment / "catalog_snapshot.json",
        cost_preflight=experiment / "cost_preflight.json",
        manual_review=experiment / "manual_review.md",
        results_table=experiment / "results_table.md",
        approval=experiment / "approval.json",
        comparison_artifacts=experiment / "artifacts" / "comparisons",
        output_dir=root / "outputs" / "model_bakeoff_2fps",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_has_size(path: Path, expected: tuple[int, int]) -> bool:
    with Image.open(path) as image:
        return image.size == expected


def _model_by_key(key: str) -> BakeoffModel:
    return next(model for model in MODELS if model.key == key)


def _required_parameters(model: BakeoffModel) -> set[str]:
    required = {"max_tokens", "temperature", "response_format", "structured_outputs"}
    if model.include_reasoning_parameter:
        required.add("reasoning")
        if model.reasoning_setting and "effort" in model.reasoning_setting:
            required.add("reasoning_effort")
    return required


def verify_catalog_payload(
    catalog_data: list[dict[str, Any]],
    endpoints_by_slug: dict[str, dict[str, Any]],
    *,
    captured_at: str | None = None,
    models: Iterable[BakeoffModel] = MODELS,
) -> dict[str, Any]:
    """Verify exact slugs and fixed active providers without substituting aliases."""

    catalog = {item.get("id"): item for item in catalog_data}
    verified: list[dict[str, Any]] = []
    for spec in models:
        if spec.slug not in catalog:
            raise RuntimeError(f"requested OpenRouter slug is unavailable: {spec.slug}")
        model = catalog[spec.slug]
        architecture = model.get("architecture") or {}
        if "image" not in architecture.get("input_modalities", []):
            raise RuntimeError(f"{spec.slug} does not advertise image input")
        if "text" not in architecture.get("output_modalities", []):
            raise RuntimeError(f"{spec.slug} does not advertise text output")
        supported = set(model.get("supported_parameters") or [])
        if not {"response_format", "structured_outputs"}.issubset(supported):
            raise RuntimeError(f"{spec.slug} lacks structured JSON output support")

        endpoint_data = endpoints_by_slug.get(spec.slug) or {}
        endpoints = endpoint_data.get("endpoints") or []
        compatible = [
            endpoint
            for endpoint in endpoints
            if endpoint.get("status") == 0
            and _required_parameters(spec).issubset(
                set(endpoint.get("supported_parameters") or [])
            )
        ]
        chosen = next(
            (endpoint for endpoint in compatible if endpoint.get("tag") == spec.provider_slug),
            None,
        )
        if chosen is None:
            raise RuntimeError(
                f"{spec.slug} has no active compatible {spec.provider_slug} endpoint"
            )
        pricing = chosen.get("pricing") or {}
        if pricing.get("prompt") is None or pricing.get("completion") is None:
            raise RuntimeError(f"{spec.slug} endpoint has incomplete token pricing")
        verified.append(
            {
                "key": spec.key,
                "requested_slug": spec.slug,
                "catalog_id": model.get("id"),
                "canonical_slug": model.get("canonical_slug"),
                "slug_exact_match": model.get("id") == spec.slug,
                "input_modalities": architecture.get("input_modalities"),
                "output_modalities": architecture.get("output_modalities"),
                "structured_output_supported": True,
                "chosen_provider": chosen.get("provider_name"),
                "chosen_provider_slug": chosen.get("tag"),
                "chosen_provider_status": chosen.get("status"),
                "chosen_provider_uptime_5m": chosen.get("uptime_last_5m"),
                "active_compatible_provider_count": len(compatible),
                "active_compatible_providers": [
                    {
                        "provider": endpoint.get("provider_name"),
                        "provider_slug": endpoint.get("tag"),
                    }
                    for endpoint in compatible
                ],
                "pricing_per_token": {
                    "input": pricing["prompt"],
                    "output": pricing["completion"],
                    "image": pricing.get("image"),
                },
                "pricing_per_million": {
                    "input": round(float(pricing["prompt"]) * 1_000_000, 9),
                    "output": round(float(pricing["completion"]) * 1_000_000, 9),
                    "image": (
                        None
                        if pricing.get("image") is None
                        else round(float(pricing["image"]) * 1_000_000, 9)
                    ),
                },
                "reasoning_catalog": model.get("reasoning"),
                "benchmark_reasoning_setting": spec.reasoning_setting,
                "reasoning_parameter_included": spec.include_reasoning_parameter,
            }
        )
    return {
        "source": CATALOG_URL,
        "captured_at": captured_at or _utc_now(),
        "all_requested_slugs_exact": True,
        "models": verified,
    }


def fetch_and_verify_catalog(
    timeout_seconds: float = 30.0,
    *,
    models: Iterable[BakeoffModel] = MODELS,
) -> dict[str, Any]:
    """Use only OpenRouter's public catalog APIs; this makes no inference request."""

    models = tuple(models)
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(CATALOG_URL)
        response.raise_for_status()
        catalog_data = response.json()["data"]
        by_id = {item.get("id"): item for item in catalog_data}
        endpoints: dict[str, dict[str, Any]] = {}
        for spec in models:
            model = by_id.get(spec.slug)
            if model is None:
                endpoints[spec.slug] = {}
                continue
            details = client.get("https://openrouter.ai" + model["links"]["details"])
            details.raise_for_status()
            endpoints[spec.slug] = details.json()["data"]
    return verify_catalog_payload(catalog_data, endpoints, models=models)


def _cost_estimate(model_snapshot: dict[str, Any]) -> dict[str, Any]:
    pricing = model_snapshot["pricing_per_million"]
    input_rate = float(pricing["input"])
    output_rate = float(pricing["output"])
    low = ANCHORS_PER_MODEL * (
        OBSERVED_MIN_INPUT_PER_REQUEST * input_rate
        + OBSERVED_MIN_OUTPUT_PER_REQUEST * output_rate
    ) / 1_000_000
    expected = (
        OBSERVED_INPUT_TOKENS * input_rate
        + OBSERVED_OUTPUT_TOKENS * output_rate
    ) / 1_000_000
    high = ANCHORS_PER_MODEL * (
        OBSERVED_HIGH_INPUT_PER_REQUEST * input_rate
        + MAX_OUTPUT_TOKENS * output_rate
    ) / 1_000_000
    return {
        "key": model_snapshot["key"],
        "model": model_snapshot["requested_slug"],
        "provider": model_snapshot["chosen_provider"],
        "provider_slug": model_snapshot["chosen_provider_slug"],
        "input_per_million_usd": input_rate,
        "output_per_million_usd": output_rate,
        "image_per_million_usd_if_separately_exposed": pricing["image"],
        "estimated_cost_per_frame_usd": round(expected / ANCHORS_PER_MODEL, 9),
        "low_61_frame_cost_usd": round(low, 9),
        "expected_61_frame_cost_usd": round(expected, 9),
        "high_61_frame_cost_usd": round(high, 9),
    }


def build_cost_preflight(catalog_snapshot: dict[str, Any]) -> dict[str, Any]:
    estimates = [_cost_estimate(model) for model in catalog_snapshot["models"]]
    control_key = MODELS[0].key
    new_estimates = [row for row in estimates if row["key"] != control_key]

    def totals(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "low_usd": round(sum(row["low_61_frame_cost_usd"] for row in rows), 9),
            "expected_usd": round(
                sum(row["expected_61_frame_cost_usd"] for row in rows), 9
            ),
            "high_usd": round(sum(row["high_61_frame_cost_usd"] for row in rows), 9),
        }

    return {
        "captured_at": catalog_snapshot["captured_at"],
        "currency": "USD",
        "anchor_count_per_model": ANCHORS_PER_MODEL,
        "models": estimates,
        "all_six_model_equivalent_cost": totals(estimates),
        "new_paid_five_model_cost_requiring_approval": totals(new_estimates),
        "new_paid_calls_requiring_approval": NEW_PAID_CALLS,
        "reused_control_historical_actual_cost_usd": OBSERVED_CONTROL_COST_USD,
        "observed_control_usage": {
            "input_tokens_total": OBSERVED_INPUT_TOKENS,
            "output_tokens_total": OBSERVED_OUTPUT_TOKENS,
            "input_tokens_per_request_approx": 2_588,
            "output_tokens_per_request_approx": 1_329,
            "actual_cost_usd": OBSERVED_CONTROL_COST_USD,
        },
        "scenario_definitions": {
            "low": (
                "Prices 61 requests at the control run's observed minima of 2,586 input "
                "and 220 output tokens per request. These independent minima are a proxy, "
                "not a guaranteed floor."
            ),
            "expected": (
                "Applies each selected endpoint's live rates to the control run's observed "
                "157,860 input and 81,058 output tokens."
            ),
            "high": (
                "Prices 61 requests at 2,588 input tokens and the fixed 4,096 output-token "
                "ceiling per request."
            ),
        },
        "cross_family_tokenization_warning": (
            "Image tokenization and output length can differ materially by model family. "
            "The estimates hold the observed Gemini token counts constant only as planning "
            "proxies; actual OpenRouter usage and cost will be recorded. A separately exposed "
            "image rate is reported but not added again because prompt_tokens already include "
            "the image-token charge in the prior run."
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_control_reuse(paths: BakeoffPaths) -> dict[str, Any]:
    """Prove detector-input equivalence before importing the paid Test 3 control."""

    info = probe_video(paths.clip)
    if info.frame_count != 900 or abs(info.fps - 30.0) > 0.01:
        raise RuntimeError("bake-off requires the verified 900-frame, 30 FPS clip")
    prior = json.loads((paths.test3 / "preflight.json").read_text(encoding="utf-8"))
    prompt_expected = (
        full_clip_detection_prompt(0).replace("must be 0", "must be {frame_id}") + "\n"
    )
    schema_expected = json.dumps(
        frame_detection_json_schema(), indent=2, sort_keys=True
    ) + "\n"
    prompt_actual = (paths.test3 / "prompt_template_v1.txt").read_text(encoding="utf-8")
    schema_actual = (paths.test3 / "schema_v2.json").read_text(encoding="utf-8")
    response_paths = sorted(paths.prior_responses.glob("frame_*_response.json"))
    detections = [
        FrameDetection.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in response_paths
    ]
    run_rows = _read_jsonl(paths.prior_runs)
    valid_run_frames = {
        row.get("frame_id")
        for row in run_rows
        if row.get("schema_validation_success") is True
    }
    invalid_run_frames = {
        row.get("frame_id")
        for row in run_rows
        if row.get("schema_validation_success") is False
    }
    checks = {
        "same_clip": prior.get("source_clip") == "clips\\dev\\spurs_thunder_test.mp4",
        "same_clip_geometry": (
            info.frame_count,
            info.width,
            info.height,
            round(info.fps, 6),
        )
        == (900, 1920, 1080, 30.0),
        "same_model": prior.get("model") == MODELS[0].slug,
        "same_61_frames": tuple(prior.get("anchor_frames", ())) == ANCHOR_FRAMES,
        "same_prompt_bytes": prompt_actual == prompt_expected,
        "same_schema_bytes": schema_actual == schema_expected,
        "same_ruler_source_files": all(
            (paths.anchors / f"frame_{frame_id:04d}_original.png").is_file()
            and (paths.anchors / f"frame_{frame_id:04d}_ruler.png").is_file()
            for frame_id in ANCHOR_FRAMES
        ),
        "same_original_resolution": prior.get("clip", {}).get("width") == 1920
        and prior.get("clip", {}).get("height") == 1080,
        "same_ruler_dimensions": all(
            _image_has_size(
                paths.anchors / f"frame_{frame_id:04d}_ruler.png", (1984, 1144)
            )
            for frame_id in ANCHOR_FRAMES
        ),
        "same_output_ceiling": prior.get("max_output_tokens") == MAX_OUTPUT_TOKENS,
        "same_concurrency": prior.get("max_concurrency") == MAX_CONCURRENCY,
        "same_no_retry_policy": prior.get("automatic_retries") == AUTOMATIC_RETRIES,
        "same_reasoning": prior.get("reasoning")
        == {"effort": "minimal", "exclude": True},
        "same_temperature": TEMPERATURE == 0,
        "saved_valid_responses_match_logs": {
            item.frame_id for item in detections
        }
        == valid_run_frames,
        "saved_invalid_responses_match_logs": all(
            (paths.prior_responses / f"frame_{frame_id:04d}_invalid.txt").is_file()
            for frame_id in invalid_run_frames
        ),
        "all_61_usage_rows_present": len(run_rows) == ANCHORS_PER_MODEL
        and {row.get("frame_id") for row in run_rows} == set(ANCHOR_FRAMES),
    }
    reusable = all(checks.values())
    return {
        "reusable": reusable,
        "checks": checks,
        "source_experiment": "experiments/test_3_2fps_comparison",
        "prompt_sha256": _sha256(paths.test3 / "prompt_template_v1.txt"),
        "schema_sha256": _sha256(paths.test3 / "schema_v2.json"),
        "limitations": [
            "Test 3 retained parsed response content, usage, cost, and latency, but not the full raw OpenRouter response envelope.",
            "Test 3's per-anchor log did not retain the returned provider name; provider is therefore unknown for the reused control.",
            "The control has 49 schema-valid responses and 12 schema-invalid responses; all 61 attempts are reused without retrying failures.",
        ],
        "decision": (
            "Reuse the exact paid control and make zero new Gemini 3.1 calls."
            if reusable
            else "Do not reuse; a new control would require separate approval."
        ),
    }


def _record_path(paths: BakeoffPaths, model: BakeoffModel, frame_id: int) -> Path:
    return paths.model_dir(model) / "records" / f"frame_{frame_id:04d}.json"


def _load_model_records(paths: BakeoffPaths, model: BakeoffModel) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frame_id in ANCHOR_FRAMES:
        path = _record_path(paths, model, frame_id)
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("frame_id") != frame_id or record.get("model") != model.slug:
            raise RuntimeError(f"invalid restart record: {path}")
        records.append(record)
    return records


def _import_control(paths: BakeoffPaths) -> dict[str, Any]:
    model = MODELS[0]
    rows = {row["frame_id"]: row for row in _read_jsonl(paths.prior_runs)}
    for frame_id in ANCHOR_FRAMES:
        row = rows[frame_id]
        response_path = paths.prior_responses / f"frame_{frame_id:04d}_response.json"
        invalid_path = paths.prior_responses / f"frame_{frame_id:04d}_invalid.txt"
        parsed = (
            json.loads(response_path.read_text(encoding="utf-8"))
            if response_path.is_file()
            else None
        )
        raw_content = invalid_path.read_text(encoding="utf-8") if invalid_path.is_file() else None
        record = {
            "experiment_id": EXPERIMENT_ID,
            "model_key": model.key,
            "model": model.slug,
            "frame_id": frame_id,
            "schema_validation_success": bool(row.get("schema_validation_success")),
            "parsed_detection": parsed,
            "raw_response_body": None,
            "legacy_raw_content": parsed,
            "raw_content": raw_content,
            "request_id": None,
            "returned_model": model.slug,
            "returned_provider": None,
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "reasoning_tokens": row.get("reasoning_tokens"),
            "actual_openrouter_cost_usd": row.get("actual_openrouter_cost_usd"),
            "request_latency_seconds": row.get("vlm_latency_seconds"),
            "request_wall_seconds": row.get("vlm_latency_seconds"),
            "retry_count": 0,
            "error": row.get("error"),
            "http_429": False,
            "reused_control": True,
        }
        _atomic_write_json(_record_path(paths, model, frame_id), record)
    records = _load_model_records(paths, model)
    _materialize_model_artifacts(paths, model, records)
    prior_summary = json.loads((paths.test3 / "summary.json").read_text(encoding="utf-8"))
    return _summarize_model(
        paths,
        model,
        records,
        reused_control=True,
        batch_wall_seconds=prior_summary.get("api_batch_seconds"),
    )


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _geometry_duplicate_pairs(detection: FrameDetection) -> int:
    signatures = [
        (
            player.team.value,
            asdict(player.box),
            asdict(player.foot),
        )
        for player in detection.players
    ]
    return len(signatures) - len(
        {(team, tuple(box.values()), tuple(foot.values())) for team, box, foot in signatures}
    )


def _summarize_model(
    paths: BakeoffPaths,
    model: BakeoffModel,
    records: list[dict[str, Any]],
    *,
    reused_control: bool = False,
    batch_wall_seconds: float | None = None,
) -> dict[str, Any]:
    valid_records = [row for row in records if row.get("schema_validation_success")]
    detections = [
        FrameDetection.from_dict(row["parsed_detection"]) for row in valid_records
    ]
    latencies = [
        float(row["request_wall_seconds"])
        for row in records
        if row.get("request_wall_seconds") is not None
    ]
    costs = [
        float(row["actual_openrouter_cost_usd"])
        for row in records
        if row.get("actual_openrouter_cost_usd") is not None
    ]
    inputs = [row.get("input_tokens") for row in records]
    outputs = [row.get("output_tokens") for row in records]
    returned_providers = Counter(
        row["returned_provider"]
        for row in records
        if row.get("returned_provider") is not None
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "model_key": model.key,
        "model": model.slug,
        "configured_provider_slug": model.provider_slug,
        "returned_provider_counts": dict(returned_providers),
        "reused_control": reused_control,
        "calls_planned": ANCHORS_PER_MODEL,
        "calls_attempted": len(records),
        "valid_responses": len(valid_records),
        "schema_valid_percentage": (
            100.0 * len(valid_records) / len(records) if records else None
        ),
        "invalid_responses": len(records) - len(valid_records),
        "average_player_detections": (
            statistics.mean(len(item.players) for item in detections)
            if detections
            else None
        ),
        "ball_returned_anchors": sum(
            item.ball_detection is not None for item in detections
        ),
        "possession_returned_anchors": sum(
            item.possession is not None for item in detections
        ),
        "exact_duplicate_geometry_pairs": sum(
            _geometry_duplicate_pairs(item) for item in detections
        ),
        "obviously_malformed_schema_valid_detections": 0,
        "total_input_tokens": (
            sum(inputs) if inputs and all(value is not None for value in inputs) else None
        ),
        "total_output_tokens": (
            sum(outputs) if outputs and all(value is not None for value in outputs) else None
        ),
        "actual_total_cost_usd": (
            sum(costs) if costs and len(costs) == len(records) else None
        ),
        "known_cost_usd": sum(costs),
        "cost_complete": len(costs) == len(records),
        "average_cost_per_request_usd": (
            statistics.mean(costs) if costs else None
        ),
        "latency_seconds": {
            "basis": "per-request wall time, including failed requests when recorded",
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile_95(latencies),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "batch_wall_seconds": batch_wall_seconds,
        "configured_concurrency": MAX_CONCURRENCY,
        "actual_concurrency": min(MAX_CONCURRENCY, len(records)) if records else 0,
        "http_429_count": sum(bool(row.get("http_429")) for row in records),
        "request_errors": sum(row.get("error") is not None for row in records),
        "provider_queueing_seconds": None,
        "provider_queueing_note": (
            "OpenRouter did not expose a separate queue-duration field; request wall time "
            "and batch wall time are retained instead."
        ),
        "automatic_retries": 0,
        "manual_quality": {
            "player_quality": None,
            "ball_quality": None,
            "team_quality": None,
            "possession_quality": None,
        },
        "updated_at": _utc_now(),
    }
    _write_json(paths.model_dir(model) / "summary.json", summary)
    return summary


def _materialize_model_artifacts(
    paths: BakeoffPaths, model: BakeoffModel, records: list[dict[str, Any]]
) -> None:
    model_dir = paths.model_dir(model)
    usage_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in records:
        frame_id = row["frame_id"]
        raw = row.get("raw_response_body")
        if raw is not None:
            _write_json(model_dir / "raw_responses" / f"frame_{frame_id:04d}.json", raw)
        elif row.get("legacy_raw_content") is not None:
            _write_json(
                model_dir / "raw_responses" / f"frame_{frame_id:04d}_legacy_content.json",
                row["legacy_raw_content"],
            )
        elif row.get("raw_content") is not None:
            (model_dir / "raw_responses").mkdir(parents=True, exist_ok=True)
            (model_dir / "raw_responses" / f"frame_{frame_id:04d}_invalid.txt").write_text(
                row["raw_content"], encoding="utf-8"
            )
        if row.get("parsed_detection") is not None:
            _write_json(
                model_dir / "parsed_detections" / f"frame_{frame_id:04d}.json",
                row["parsed_detection"],
            )
        usage_rows.append(
            {
                key: row.get(key)
                for key in (
                    "frame_id",
                    "model",
                    "returned_model",
                    "returned_provider",
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "actual_openrouter_cost_usd",
                )
            }
        )
        latency_rows.append(
            {
                key: row.get(key)
                for key in (
                    "frame_id",
                    "model",
                    "request_latency_seconds",
                    "request_wall_seconds",
                    "http_429",
                    "error",
                )
            }
        )
        if not row.get("schema_validation_success"):
            failures.append(
                {
                    "frame_id": frame_id,
                    "error": row.get("error"),
                    "raw_content": row.get("raw_content"),
                }
            )
    _write_jsonl(model_dir / "usage_cost.jsonl", usage_rows)
    _write_jsonl(model_dir / "latency.jsonl", latency_rows)
    _write_jsonl(model_dir / "schema_failures.jsonl", failures)


def _annotation_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def annotate_detection(image: Image.Image, detection: FrameDetection) -> Image.Image:
    """Render raw VLM detections without applying any tracker or interpolation."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    font_size = max(18, height // 45)
    font = _annotation_font(font_size)
    colors = {Team.A: "#00b7ff", Team.B: "#ff3b5c", Team.UNCERTAIN: "#aaaaaa"}
    for player in detection.players:
        box = (
            *normalized_to_pixel(player.box.left, player.box.top, width, height),
            *normalized_to_pixel(player.box.right, player.box.bottom, width, height),
        )
        color = colors[player.team]
        draw.rectangle(box, outline=color, width=max(3, height // 270))
        foot_x, foot_y = normalized_to_pixel(player.foot.x, player.foot.y, width, height)
        radius = max(5, height // 150)
        draw.ellipse(
            (foot_x - radius, foot_y - radius, foot_x + radius, foot_y + radius),
            fill=color,
            outline="black",
            width=2,
        )
        draw.text(
            (box[0], max(0, box[1] - font_size - 4)),
            f"{player.detection_id} {player.team.value}",
            fill="white",
            font=font,
            stroke_width=2,
            stroke_fill="black",
        )
        if player.possesses_ball:
            draw.polygon(
                (
                    (foot_x, foot_y - radius * 5),
                    (foot_x - radius * 2, foot_y - radius * 8),
                    (foot_x + radius * 2, foot_y - radius * 8),
                ),
                fill="#ffe600",
                outline="black",
            )
    if detection.ball_detection is not None:
        ball = detection.ball_detection
        x, y = normalized_to_pixel(ball.center.x, ball.center.y, width, height)
        radius = max(7, height // 110)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline="#ffe600",
            width=max(3, height // 270),
        )
        if ball.box is not None:
            draw.rectangle(
                (
                    *normalized_to_pixel(ball.box.left, ball.box.top, width, height),
                    *normalized_to_pixel(ball.box.right, ball.box.bottom, width, height),
                ),
                outline="#ffe600",
                width=max(2, height // 360),
            )
    return output


def _render_model_review_overlays(
    paths: BakeoffPaths, model: BakeoffModel, records: list[dict[str, Any]]
) -> None:
    by_frame = {row["frame_id"]: row for row in records}
    output_dir = paths.model_dir(model) / "review_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Review overlays are presentation artifacts, not model inputs. Re-extract their
    # source frames so a copied repository with stale per-file Windows ACLs remains
    # runnable while the immutable ruler images sent to the models stay unchanged.
    review_sources = extract_video_frames(
        paths.clip, REVIEW_FRAMES, paths.experiment / "artifacts" / "review_sources"
    )
    for frame_id in REVIEW_FRAMES:
        source_path = review_sources[frame_id]
        with Image.open(source_path) as source:
            row = by_frame.get(frame_id)
            if row and row.get("parsed_detection") is not None:
                image = annotate_detection(
                    source, FrameDetection.from_dict(row["parsed_detection"])
                )
            else:
                image = source.convert("RGB").copy()
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 80), fill="black")
                draw.text(
                    (20, 20),
                    "NO SCHEMA-VALID DETECTION",
                    fill="white",
                    font=_annotation_font(32),
                )
        image.save(output_dir / f"frame_{frame_id:04d}.jpg", quality=92)


def _render_side_by_side(paths: BakeoffPaths) -> None:
    paths.comparison_artifacts.mkdir(parents=True, exist_ok=True)
    font = _annotation_font(22)
    for frame_id in REVIEW_FRAMES:
        canvas = Image.new("RGB", (1920, 760), "#161616")
        draw = ImageDraw.Draw(canvas)
        for index, model in enumerate(MODELS):
            column, row = index % 3, index // 3
            overlay_path = (
                paths.model_dir(model) / "review_overlays" / f"frame_{frame_id:04d}.jpg"
            )
            if overlay_path.is_file():
                with Image.open(overlay_path) as overlay:
                    tile = overlay.convert("RGB").resize((640, 360))
            else:
                tile = Image.new("RGB", (640, 360), "#333333")
            canvas.paste(tile, (column * 640, row * 380 + 20))
            draw.text(
                (column * 640 + 8, row * 380),
                model.slug,
                fill="white",
                font=font,
                stroke_width=2,
                stroke_fill="black",
            )
        canvas.save(
            paths.comparison_artifacts / f"frame_{frame_id:04d}_six_models.jpg",
            quality=92,
        )


def _manual_review_markdown() -> str:
    categories = {
        30: "wide court",
        180: "crowded players",
        270: "crossing/occlusion",
        360: "small/distant players",
        450: "post-cut wide view",
        570: "ball/possession moment",
        690: "difficult tiny ball",
        840: "motion/crowding",
    }
    header = (
        "# Manual detector review\n\n"
        "Subjective fields are intentionally blank. Review the fixed-frame overlays; do not "
        "use tracker output for these ratings.\n\n"
        "| Model | Frame | Scene | Missed players | False players | Poor player localization | "
        "Foot-point quality | Wrong team | Ball correct/missed/wrong | "
        "Possession correct/wrong/uncertain | Notes |\n"
        "|---|---:|---|---:|---:|---:|---|---:|---|---|---|\n"
    )
    rows = []
    for model in MODELS:
        for frame_id in REVIEW_FRAMES:
            rows.append(
                f"| {model.slug} | {frame_id} | {categories[frame_id]} |  |  |  |  |  |  |  |  |"
            )
    return header + "\n".join(rows) + "\n"


def _results_table_markdown(
    summaries: dict[str, dict[str, Any]] | None = None,
    *,
    models: Iterable[BakeoffModel] = MODELS,
) -> str:
    summaries = summaries or {}
    lines = [
        "# Model bake-off results",
        "",
        "Manual quality columns remain blank until human detector review is entered. No overall winner is declared automatically.",
        "",
        "| Model | Valid % | Player quality* | Ball quality* | Team quality* | Possession* | Cost | Mean latency | Batch time |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for model in models:
        summary = summaries.get(model.key, {})
        valid = summary.get("schema_valid_percentage")
        cost = summary.get("actual_total_cost_usd")
        latency = (summary.get("latency_seconds") or {}).get("mean")
        batch = summary.get("batch_wall_seconds")
        lines.append(
            "| "
            + " | ".join(
                (
                    model.slug,
                    "" if valid is None else f"{valid:.2f}%",
                    "",
                    "",
                    "",
                    "",
                    "" if cost is None else f"${cost:.6f}",
                    "" if latency is None else f"{latency:.3f}s",
                    "" if batch is None else f"{batch:.3f}s",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _model_config(
    paths: BakeoffPaths,
    model: BakeoffModel,
    snapshot: dict[str, Any],
    control_audit: dict[str, Any],
) -> dict[str, Any]:
    variant = cv_improved_variant()
    player_cv_pipeline = variant.pipeline.as_dict()
    player_cv_pipeline.pop("model", None)
    return {
        "experiment_id": EXPERIMENT_ID,
        "model_key": model.key,
        "model": model.slug,
        "canonical_slug_at_preflight": snapshot["canonical_slug"],
        "control_reuse": model.control_reuse,
        "control_reuse_verified": control_audit["reusable"] if model.control_reuse else False,
        "source_clip": str(paths.clip.relative_to(paths.root)),
        "anchor_frames": list(ANCHOR_FRAMES),
        "frame_interval": FRAME_INTERVAL,
        "sampling_fps": 2.0,
        "original_resolution": [1920, 1080],
        "ruler_margin_px": 64,
        "ruler_tick_step": 0.1,
        "prompt_resolution": [1984, 1144],
        "prompt_sha256": control_audit["prompt_sha256"],
        "schema_sha256": control_audit["schema_sha256"],
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_parameter_included": model.include_reasoning_parameter,
        "reasoning_setting": model.reasoning_setting,
        "automatic_retries": AUTOMATIC_RETRIES,
        "max_concurrency": MAX_CONCURRENCY,
        "provider_routing": {
            "only": [model.provider_slug],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "verified_provider": snapshot["chosen_provider"],
        "verified_pricing_per_million": snapshot["pricing_per_million"],
        "player_cv_pipeline": player_cv_pipeline,
        "raw_response_directory": "raw_responses",
        "parsed_detection_directory": "parsed_detections",
        "record_directory": "records",
        "output_video": str(
            (paths.output_dir / f"{model.key}.mp4").relative_to(paths.root)
        ),
    }


def prepare_bakeoff(repository_root: str | Path) -> BakeoffPaths:
    """Query public catalogs and prepare everything without an inference request."""

    paths = bakeoff_paths(repository_root)
    paths.experiment.mkdir(parents=True, exist_ok=True)
    catalog = fetch_and_verify_catalog()
    control = audit_control_reuse(paths)
    if not control["reusable"]:
        raise RuntimeError("existing Gemini control is not equivalent; preparation stopped")
    cost = build_cost_preflight(catalog)
    paths.prompt.write_text(
        full_clip_detection_prompt(0).replace("must be 0", "must be {frame_id}") + "\n",
        encoding="utf-8",
    )
    paths.schema.write_text(
        json.dumps(frame_detection_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_json(paths.catalog_snapshot, catalog)
    _write_json(paths.cost_preflight, cost)
    snapshot_by_key = {row["key"]: row for row in catalog["models"]}
    for model in MODELS:
        model_dir = paths.model_dir(model)
        for directory in (
            "records",
            "raw_responses",
            "parsed_detections",
            "review_overlays",
        ):
            (model_dir / directory).mkdir(parents=True, exist_ok=True)
        _write_json(
            model_dir / "config.json",
            _model_config(paths, model, snapshot_by_key[model.key], control),
        )
    control_summary = _import_control(paths)
    paths.manual_review.write_text(_manual_review_markdown(), encoding="utf-8")
    paths.results_table.write_text(
        _results_table_markdown({MODELS[0].key: control_summary}), encoding="utf-8"
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared_not_approved",
        "prepared_at": _utc_now(),
        "strictly_2fps": True,
        "source_clip": str(paths.clip.relative_to(paths.root)),
        "anchor_frames": list(ANCHOR_FRAMES),
        "anchors_per_model": ANCHORS_PER_MODEL,
        "models": [asdict(model) for model in MODELS],
        "control_reuse": control,
        "new_paid_models": [model.slug for model in PAID_MODELS],
        "new_paid_calls_requiring_approval": NEW_PAID_CALLS,
        "new_paid_inference_calls_attempted": 0,
        "cost_preflight": str(paths.cost_preflight.relative_to(paths.root)),
        "common_configuration": {
            "frame_interval": FRAME_INTERVAL,
            "sampling_fps": 2.0,
            "max_concurrency": MAX_CONCURRENCY,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "automatic_retries": AUTOMATIC_RETRIES,
            "prompt_sha256": control["prompt_sha256"],
            "schema_sha256": control["schema_sha256"],
            "provider_fallbacks": False,
            "provider_must_support_all_parameters": True,
            "provider_selection_policy": (
                "Pin one active structured-output-compatible standard endpoint per model. "
                "Use the lowest-priced compatible non-flex endpoint; keep Gemini on standard "
                "Google AI Studio rather than discounted flex capacity so queueing does not "
                "confound the latency benchmark."
            ),
        },
        "review_frames": list(REVIEW_FRAMES),
        "exact_run_command": (
            ".venv\\Scripts\\python.exe -m track_game.model_bakeoff run --approved-call-count "
            f"{NEW_PAID_CALLS}"
        ),
        "approval_gate": (
            f"STOP: requires explicit approval for the displayed {NEW_PAID_CALLS} new paid calls."
        ),
        "overall_winner": None,
    }
    _write_json(paths.manifest, manifest)
    return paths


def _request_record(
    provider: OpenRouterVLMProvider,
    paths: BakeoffPaths,
    model: BakeoffModel,
    frame_id: int,
) -> dict[str, Any]:
    started = perf_counter()
    result: OpenRouterResult | None = None
    error: Exception | None = None
    raw_content: str | None = None
    raw_body: dict[str, Any] | None = None
    usage = None
    latency = None
    request_id = returned_model = returned_provider = None
    try:
        with Image.open(paths.anchors / f"frame_{frame_id:04d}_ruler.png") as image:
            result = provider.request_detection(
                frame_id,
                image.copy(),
                prompt=full_clip_detection_prompt(frame_id),
            )
        usage = result.usage
        latency = result.latency_seconds
        raw_body = result.raw_response_body
        request_id = result.request_id
        returned_model = result.returned_model
        returned_provider = result.provider
    except Exception as caught:
        error = caught
        if isinstance(caught, OpenRouterValidationError):
            usage = caught.usage
            latency = caught.latency_seconds
            raw_content = caught.raw_content
            raw_body = caught.raw_response_body
            request_id = caught.request_id
            returned_model = caught.returned_model
            returned_provider = caught.provider
    message = None if error is None else f"{type(error).__name__}: {error}"
    return {
        "experiment_id": EXPERIMENT_ID,
        "model_key": model.key,
        "model": model.slug,
        "frame_id": frame_id,
        "schema_validation_success": result is not None,
        "parsed_detection": (
            None if result is None else _detection_as_dict(result.detection)
        ),
        "raw_response_body": raw_body,
        "raw_content": raw_content,
        "request_id": request_id,
        "returned_model": returned_model,
        "returned_provider": returned_provider,
        "input_tokens": None if usage is None else usage.prompt_tokens,
        "output_tokens": None if usage is None else usage.completion_tokens,
        "reasoning_tokens": None if usage is None else usage.reasoning_tokens,
        "actual_openrouter_cost_usd": None if usage is None else usage.cost_usd,
        "request_latency_seconds": latency,
        "request_wall_seconds": perf_counter() - started,
        "retry_count": 0,
        "error": message,
        "http_429": bool(message and "HTTP 429" in message),
        "reused_control": False,
        "completed_at": _utc_now(),
    }


def _append_batch_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _run_detector_model(
    paths: BakeoffPaths,
    model: BakeoffModel,
    *,
    timeout_seconds: float = 45.0,
) -> tuple[list[dict[str, Any]], float]:
    existing = _load_model_records(paths, model)
    completed = {row["frame_id"] for row in existing}
    remaining = [frame for frame in ANCHOR_FRAMES if frame not in completed]
    if not remaining:
        batches = _read_jsonl(paths.model_dir(model) / "batches.jsonl")
        return existing, sum(row.get("wall_seconds", 0.0) for row in batches)

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
        timeout_seconds=timeout_seconds,
    )
    batch_started = perf_counter()
    futures: dict[Future[dict[str, Any]], int] = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        for frame_id in remaining:
            future = executor.submit(_request_record, provider, paths, model, frame_id)
            futures[future] = frame_id
        for future in as_completed(futures):
            frame_id = futures[future]
            record = future.result()
            _atomic_write_json(_record_path(paths, model, frame_id), record)
    segment_seconds = perf_counter() - batch_started
    _append_batch_log(
        paths.model_dir(model) / "batches.jsonl",
        {
            "started_at": _utc_now(),
            "attempted_frames": remaining,
            "attempted_calls": len(remaining),
            "configured_concurrency": MAX_CONCURRENCY,
            "actual_concurrency": min(MAX_CONCURRENCY, len(remaining)),
            "wall_seconds": segment_seconds,
            "automatic_retries": 0,
            "request_timeout_seconds": timeout_seconds,
        },
    )
    records = _load_model_records(paths, model)
    batches = _read_jsonl(paths.model_dir(model) / "batches.jsonl")
    return records, sum(row.get("wall_seconds", 0.0) for row in batches)


def _detections_from_records(records: list[dict[str, Any]]) -> dict[int, FrameDetection]:
    return {
        row["frame_id"]: FrameDetection.from_dict(row["parsed_detection"])
        for row in records
        if row.get("schema_validation_success")
        and row.get("parsed_detection") is not None
    }


def _run_player_cv_pipeline(
    paths: BakeoffPaths,
    model: BakeoffModel,
    detections: dict[int, FrameDetection],
    summary: dict[str, Any],
) -> None:
    if len(detections) < 2:
        summary["tracking"] = {
            "status": "skipped",
            "reason": "fewer than two schema-valid detector anchors",
        }
        _write_json(paths.model_dir(model) / "summary.json", summary)
        return
    variant = cv_improved_variant()
    local_started = perf_counter()
    tracker = build_player_tracker(variant.pipeline.tracking)
    player_started = perf_counter()
    player_result = VLMInitializedPlayerCVTracker(
        variant.pipeline, tracker
    ).track_video(paths.clip, detections)
    player_seconds = perf_counter() - player_started
    cuts = {item.frame_id for item in player_result.scene_cuts if item.is_cut}
    timeline = list(player_result.timeline)
    ball_started = perf_counter()
    ball_result = VLMInitializedBallTracker(variant.pipeline.ball_tracking).track_video(
        paths.clip,
        trackable_ball_anchors(detections, player_result.shot_context),
        timeline,
        scene_cut_frames=cuts,
    )
    ball_seconds = perf_counter() - ball_started
    timeline = apply_ball_track(timeline, ball_result)
    output_video = paths.output_dir / f"{model.key}.mp4"
    render_started = perf_counter()
    render_tracked_video(paths.clip, output_video, timeline, show_ids=True)
    render_seconds = perf_counter() - render_started
    local_seconds = perf_counter() - local_started
    model_dir = paths.model_dir(model)
    _write_jsonl(model_dir / "camera_motion.jsonl", camera_log_as_dict(player_result))
    _write_jsonl(model_dir / "scene_cuts.jsonl", scene_cut_log_as_dict(player_result))
    _write_jsonl(model_dir / "track_confidence.jsonl", player_result.confidence_history)
    stats = player_result.stats
    lifecycle = tracker.lifecycle
    detector_cost = summary.get("actual_total_cost_usd")
    record = TrackingMetricsRecord(
        experiment_id=EXPERIMENT_ID,
        variant=f"{variant.name}:{model.key}",
        model=model.slug,
        anchor_count=ANCHORS_PER_MODEL,
        planned_vlm_calls=ANCHORS_PER_MODEL,
        attempted_vlm_calls=summary["calls_attempted"],
        successful_vlm_calls=len(detections),
        actual_api_cost_usd=detector_cost,
        api_cost_complete=summary["cost_complete"],
        summed_vlm_latency_seconds=sum(
            row["request_latency_seconds"]
            for row in _load_model_records(paths, model)
            if row.get("request_latency_seconds") is not None
        ),
        api_batch_seconds=summary.get("batch_wall_seconds"),
        local_processing_seconds=local_seconds,
        total_processing_seconds=(summary.get("batch_wall_seconds") or 0) + local_seconds,
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
        scene_cut_frames=tuple(sorted(cuts)),
        camera_motion_successes=sum(item.success for item in player_result.camera_motion),
        camera_motion_failures=sum(
            item.frame_id > 0 and not item.success for item in player_result.camera_motion
        ),
        camera_motion_seconds=stats.camera_seconds,
        player_cv_tracking_seconds=stats.player_tracking_seconds,
        player_cv_total_pass_seconds=player_seconds,
        scene_cut_detection_seconds=stats.scene_cut_seconds,
        ball_tracking_seconds=ball_seconds,
        rendering_seconds=render_seconds,
        average_local_seconds_per_frame=local_seconds / len(timeline),
    )
    TrackingMetricsWriter(model_dir / "tracking_metrics.json").write(record)
    summary["tracking"] = asdict(record)
    summary["output_video"] = str(output_video.relative_to(paths.root))
    _write_json(model_dir / "summary.json", summary)


def _catalog_approval_fingerprint(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": model["key"],
            "requested_slug": model["requested_slug"],
            "canonical_slug": model["canonical_slug"],
            "chosen_provider_slug": model["chosen_provider_slug"],
            "pricing_per_token": model["pricing_per_token"],
        }
        for model in snapshot["models"]
    ]


def run_approved_bakeoff(
    repository_root: str | Path, approved_call_count: int
) -> BakeoffPaths:
    """Run at most the approved 305 new calls; completed frame records are never retried."""

    if approved_call_count != NEW_PAID_CALLS:
        raise PermissionError(
            f"bake-off requires explicit approval for exactly {NEW_PAID_CALLS} new calls"
        )
    paths = prepare_bakeoff(repository_root)
    current_catalog = json.loads(paths.catalog_snapshot.read_text(encoding="utf-8"))
    fingerprint = _catalog_approval_fingerprint(current_catalog)
    if paths.approval.is_file():
        approval = json.loads(paths.approval.read_text(encoding="utf-8"))
        if approval.get("approved_call_count") != NEW_PAID_CALLS:
            raise PermissionError("stored approval does not match this benchmark")
        if approval.get("catalog_fingerprint") != fingerprint:
            raise PermissionError("live model/provider/pricing changed; new approval is required")
    else:
        approval = {
            "approved_at": _utc_now(),
            "approved_call_count": NEW_PAID_CALLS,
            "catalog_fingerprint": fingerprint,
        }
        _write_json(paths.approval, approval)

    attempted = sum(len(_load_model_records(paths, model)) for model in PAID_MODELS)
    remaining = NEW_PAID_CALLS - attempted
    if remaining < 0:
        raise RuntimeError("more paid records exist than the approved benchmark permits")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "approved_run_started",
            "approved_call_count": approved_call_count,
            "new_paid_inference_calls_already_recorded": attempted,
            "new_paid_inference_calls_remaining": remaining,
            "run_started_at": _utc_now(),
        }
    )
    _write_json(paths.manifest, manifest)

    summaries: dict[str, dict[str, Any]] = {}
    control_records = _load_model_records(paths, MODELS[0])
    control_summary = _summarize_model(
        paths,
        MODELS[0],
        control_records,
        reused_control=True,
        batch_wall_seconds=json.loads(
            (paths.test3 / "summary.json").read_text(encoding="utf-8")
        ).get("api_batch_seconds"),
    )
    _render_model_review_overlays(paths, MODELS[0], control_records)
    _run_player_cv_pipeline(
        paths,
        MODELS[0],
        _detections_from_records(control_records),
        control_summary,
    )
    summaries[MODELS[0].key] = control_summary

    for model in PAID_MODELS:
        records, batch_seconds = _run_detector_model(paths, model)
        _materialize_model_artifacts(paths, model, records)
        summary = _summarize_model(
            paths, model, records, batch_wall_seconds=batch_seconds
        )
        _render_model_review_overlays(paths, model, records)
        _run_player_cv_pipeline(
            paths, model, _detections_from_records(records), summary
        )
        summaries[model.key] = summary

    _render_side_by_side(paths)
    paths.results_table.write_text(_results_table_markdown(summaries), encoding="utf-8")
    total_records = sum(len(_load_model_records(paths, model)) for model in PAID_MODELS)
    manifest.update(
        {
            "status": "completed" if total_records == NEW_PAID_CALLS else "partial",
            "new_paid_inference_calls_attempted": total_records,
            "new_paid_inference_calls_remaining": NEW_PAID_CALLS - total_records,
            "completed_at": _utc_now(),
            "overall_winner": None,
        }
    )
    _write_json(paths.manifest, manifest)
    return paths


def run_gemini_3_7_extension(
    repository_root: str | Path,
    approved_call_count: int,
    *,
    retry_with_long_timeout: bool = False,
) -> BakeoffPaths:
    """Run the approved Gemini 3.7 extension without disturbing the completed bake-off."""

    if approved_call_count != ANCHORS_PER_MODEL:
        raise PermissionError(
            f"Gemini 3.7 extension requires approval for exactly {ANCHORS_PER_MODEL} calls"
        )
    paths = bakeoff_paths(repository_root)
    model = GEMINI_3_7_RETRY_MODEL if retry_with_long_timeout else GEMINI_3_7_MODEL
    timeout_seconds = 120.0 if retry_with_long_timeout else 45.0
    model_dir = paths.model_dir(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    for directory in (
        "records",
        "raw_responses",
        "parsed_detections",
        "review_overlays",
    ):
        (model_dir / directory).mkdir(parents=True, exist_ok=True)

    catalog = fetch_and_verify_catalog(models=(model,))
    control = audit_control_reuse(paths)
    if not control["reusable"]:
        raise RuntimeError("existing Gemini control is not equivalent; extension stopped")
    snapshot = catalog["models"][0]
    estimate = _cost_estimate(snapshot)
    _write_json(model_dir / "catalog_snapshot.json", catalog)
    _write_json(
        model_dir / "cost_preflight.json",
        {
            "captured_at": catalog["captured_at"],
            "currency": "USD",
            "approved_calls": ANCHORS_PER_MODEL,
            "estimate": estimate,
        },
    )
    config = _model_config(paths, model, snapshot, control)
    config["request_timeout_seconds"] = timeout_seconds
    config["source_audio_policy"] = "source and rendered output must be silent"
    _write_json(model_dir / "config.json", config)

    fingerprint = _catalog_approval_fingerprint(catalog)
    approval_path = model_dir / "approval.json"
    if approval_path.is_file():
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        if approval.get("approved_call_count") != ANCHORS_PER_MODEL:
            raise PermissionError("stored Gemini 3.7 approval has the wrong call count")
        if approval.get("catalog_fingerprint") != fingerprint:
            raise PermissionError("Gemini 3.7 provider/pricing changed; new approval required")
    else:
        _write_json(
            approval_path,
            {
                "approved_at": _utc_now(),
                "approved_call_count": ANCHORS_PER_MODEL,
                "catalog_fingerprint": fingerprint,
            },
        )

    extension_manifest_path = model_dir / "manifest.json"
    existing = _load_model_records(paths, model)
    extension_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "extension": (
            "gemini-3.7-flash-retry-120s"
            if retry_with_long_timeout
            else "gemini-3.7-flash"
        ),
        "status": "approved_run_started",
        "model": model.slug,
        "provider_slug": model.provider_slug,
        "approved_call_count": ANCHORS_PER_MODEL,
        "request_timeout_seconds": timeout_seconds,
        "paid_inference_calls_already_recorded": len(existing),
        "paid_inference_calls_remaining": ANCHORS_PER_MODEL - len(existing),
        "cost_preflight": estimate,
        "started_at": _utc_now(),
    }
    _write_json(extension_manifest_path, extension_manifest)

    records, batch_seconds = _run_detector_model(
        paths, model, timeout_seconds=timeout_seconds
    )
    _materialize_model_artifacts(paths, model, records)
    summary = _summarize_model(paths, model, records, batch_wall_seconds=batch_seconds)
    _render_model_review_overlays(paths, model, records)
    _run_player_cv_pipeline(paths, model, _detections_from_records(records), summary)

    extension_manifest.update(
        {
            "status": "completed" if len(records) == ANCHORS_PER_MODEL else "partial",
            "paid_inference_calls_attempted": len(records),
            "paid_inference_calls_remaining": ANCHORS_PER_MODEL - len(records),
            "completed_at": _utc_now(),
            "output_video": summary.get("output_video"),
            "overall_winner": None,
        }
    )
    _write_json(extension_manifest_path, extension_manifest)

    display_models = MODELS + (GEMINI_3_7_MODEL,)
    summaries = {}
    for display_model in display_models:
        summary_path = paths.model_dir(display_model) / "summary.json"
        if summary_path.is_file():
            summaries[display_model.key] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
    if retry_with_long_timeout:
        summaries[GEMINI_3_7_MODEL.key] = summary
    paths.results_table.write_text(
        _results_table_markdown(summaries, models=display_models), encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run", "run-gemini-3-7", "retry-gemini-3-7"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--approved-call-count", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_bakeoff(args.repository_root).manifest)
    elif args.command == "run":
        print(
            run_approved_bakeoff(
                args.repository_root, args.approved_call_count
            ).results_table
        )
    elif args.command == "run-gemini-3-7":
        print(
            run_gemini_3_7_extension(
                args.repository_root, args.approved_call_count
            ).results_table
        )
    else:
        print(
            run_gemini_3_7_extension(
                args.repository_root,
                args.approved_call_count,
                retry_with_long_timeout=True,
            ).results_table
        )


if __name__ == "__main__":
    main()
