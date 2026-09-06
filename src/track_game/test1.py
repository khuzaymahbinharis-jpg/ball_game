"""Prepare and, only after explicit approval, run the one-call Test 1."""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from .drawing import annotate_frame
from .experiments import ExperimentRecord, JsonlExperimentLogger
from .provider import (
    DEFAULT_OPENROUTER_MODEL,
    PROMPT_VERSION,
    OpenRouterResult,
    OpenRouterVLMProvider,
    detection_prompt,
)
from .ruler import add_normalized_rulers
from .schema import SCHEMA_VERSION, FrameDetection, frame_detection_json_schema
from .tracking import NearestNeighbourTracker
from .video import extract_video_frame, probe_video


TEST_ID = "test-1-single-ruler-frame"
FRAME_NUMBER = 450
SOURCE_TIMESTAMP_SECONDS = 15.0
REASONING_SETTING = {"effort": "minimal", "exclude": True}
MAX_OUTPUT_TOKENS = 4096

# Snapshot checked before this experiment. Update and re-approve if pricing changes.
PRICING_SNAPSHOT: dict[str, Any] = {
    "captured_at": "2026-08-31T08:35:57.5229179Z",
    "model": DEFAULT_OPENROUTER_MODEL,
    "currency": "USD",
    "listed_input_per_million_tokens": 0.25,
    "listed_image_input_per_million_tokens": 0.25,
    "listed_output_per_million_tokens": 1.50,
    "higher_provider_input_per_million_tokens": 0.275,
    "higher_provider_output_per_million_tokens": 1.65,
    "source": "https://openrouter.ai/google/gemini-3.1-flash-lite/pricing",
}

COST_ESTIMATE = {
    "low_usd": 0.0014,
    "expected_usd": 0.0031,
    "high_usd": 0.0080,
    "assumptions": {
        "vlm_calls": 1,
        "prompt_image": "one 1984x1144 PNG (1920x1080 frame plus 64px rulers)",
        "image_token_estimate": "about 1,548 tokens (six 768px tiles at 258 tokens each)",
        "low_tokens": {"input": 2000, "output_including_reasoning": 600},
        "expected_tokens": {"input": 2800, "output_including_reasoning": 1600},
        "high_tokens": {"input": 4500, "output_including_reasoning": 4096},
        "note": "Approximate: provider tokenization and Gemini thinking usage are not known before the call.",
    },
}


@dataclass(frozen=True)
class Test1Paths:
    clip: Path
    original_frame: Path
    ruled_frame: Path
    prompt: Path
    schema: Path
    manifest: Path
    response: Path
    preview: Path
    metrics: Path


def test1_paths(repository_root: str | Path) -> Test1Paths:
    root = Path(repository_root).resolve()
    directory = root / "experiments" / "test_1"
    artifacts = directory / "artifacts"
    return Test1Paths(
        clip=root / "clips" / "dev" / "spurs_thunder_test.mp4",
        original_frame=artifacts / "frame_0450_original.png",
        ruled_frame=artifacts / "frame_0450_ruler.png",
        prompt=directory / "prompt_v1.txt",
        schema=directory / "schema_v2.json",
        manifest=directory / "preflight.json",
        response=artifacts / "frame_0450_response.json",
        preview=artifacts / "frame_0450_preview.png",
        metrics=directory / "runs.jsonl",
    )


def prepare_test1(repository_root: str | Path) -> Test1Paths:
    """Create every local Test 1 artifact. This function cannot call OpenRouter."""

    paths = test1_paths(repository_root)
    info = probe_video(paths.clip)
    if info.frame_count != 900 or abs(info.duration_seconds - 30.0) > 0.02:
        raise ValueError("Test 1 clip must be 900 frames and 30 seconds")
    original = extract_video_frame(paths.clip, FRAME_NUMBER, paths.original_frame)
    ruled = add_normalized_rulers(original, margin_px=64, tick_step=0.1)
    paths.ruled_frame.parent.mkdir(parents=True, exist_ok=True)
    ruled.save(paths.ruled_frame, format="PNG", optimize=True)
    paths.prompt.write_text(detection_prompt(FRAME_NUMBER) + "\n", encoding="utf-8")
    paths.schema.write_text(
        json.dumps(frame_detection_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": TEST_ID,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "paid_request_attempted": False,
        "paid_request_succeeded": False,
        "source_clip": str(paths.clip.relative_to(Path(repository_root).resolve())),
        "source_frame_number": FRAME_NUMBER,
        "source_timestamp_seconds": SOURCE_TIMESTAMP_SECONDS,
        "clip": asdict(info),
        "original_image_resolution": list(original.size),
        "prompt_image_resolution": list(ruled.size),
        "ruler_grounding": True,
        "model": DEFAULT_OPENROUTER_MODEL,
        "reasoning": REASONING_SETTING,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "pricing_snapshot": PRICING_SNAPSHOT,
        "cost_estimate": COST_ESTIMATE,
        "approval_gate": "One call only after explicit user approval; no automatic retries.",
    }
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _update_manifest(paths: Test1Paths, **updates: Any) -> None:
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest.update(updates)
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _detection_as_dict(detection: FrameDetection) -> dict[str, Any]:
    players = [
        {
            "detection_id": player.detection_id,
            "team": player.team.value,
            "box": {
                "left": player.box.left,
                "top": player.box.top,
                "right": player.box.right,
                "bottom": player.box.bottom,
            },
            "foot": {"x": player.foot.x, "y": player.foot.y},
            "confidence": player.confidence,
            "team_confidence": player.team_confidence,
        }
        for player in detection.players
    ]
    ball = None
    if detection.ball_detection is not None:
        box = detection.ball_detection.box
        ball = {
            "center": {
                "x": detection.ball_detection.center.x,
                "y": detection.ball_detection.center.y,
            },
            "box": (
                None
                if box is None
                else {
                    "left": box.left,
                    "top": box.top,
                    "right": box.right,
                    "bottom": box.bottom,
                }
            ),
            "confidence": detection.ball_detection.confidence,
        }
    possession = None
    if detection.possession is not None:
        possession = {
            "player_detection_id": detection.possession.player_detection_id,
            "confidence": detection.possession.confidence,
        }
    return {
        "frame_id": detection.frame_id,
        "players": players,
        "ball": ball,
        "possession": possession,
        "uncertainty_notes": list(detection.uncertainty_notes),
    }


def render_detection_preview(
    original: Image.Image, detection: FrameDetection
) -> Image.Image:
    """Map normalized response coordinates onto the untouched source frame."""

    tracked = NearestNeighbourTracker().update(detection)
    return annotate_frame(original, tracked, show_ids=True)


def _record_for_success(
    paths: Test1Paths, result: OpenRouterResult, wall_seconds: float
) -> ExperimentRecord:
    detection = result.detection
    return ExperimentRecord(
        experiment_id=TEST_ID,
        status="completed",
        source_clip=str(paths.clip),
        source_timestamp_seconds=SOURCE_TIMESTAMP_SECONDS,
        source_frame_number=FRAME_NUMBER,
        model=DEFAULT_OPENROUTER_MODEL,
        pricing_snapshot=PRICING_SNAPSHOT,
        image_resolution=(1920, 1080),
        prompt_image_resolution=(1984, 1144),
        ruler_grounding=True,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        reasoning_setting=REASONING_SETTING,
        vlm_calls=1,
        provider_request_id=result.request_id,
        returned_model=result.returned_model,
        returned_provider=result.provider,
        input_tokens=result.usage.prompt_tokens,
        output_tokens=result.usage.completion_tokens,
        reasoning_tokens=result.usage.reasoning_tokens,
        vlm_request_latency_seconds=result.latency_seconds,
        total_wall_clock_seconds=wall_seconds,
        retries=0,
        actual_openrouter_cost_usd=result.usage.cost_usd,
        estimated_cost_low_usd=COST_ESTIMATE["low_usd"],
        estimated_cost_expected_usd=COST_ESTIMATE["expected_usd"],
        estimated_cost_high_usd=COST_ESTIMATE["high_usd"],
        schema_validation_success=True,
        players_returned=len(detection.players),
        ball_detected=detection.ball_detection is not None,
        possession_returned=detection.possession is not None,
    )


def _record_for_failure(
    paths: Test1Paths, error: Exception, wall_seconds: float
) -> ExperimentRecord:
    message = f"{type(error).__name__}: {error}"
    return ExperimentRecord(
        experiment_id=TEST_ID,
        status="failed",
        source_clip=str(paths.clip),
        source_timestamp_seconds=SOURCE_TIMESTAMP_SECONDS,
        source_frame_number=FRAME_NUMBER,
        model=DEFAULT_OPENROUTER_MODEL,
        pricing_snapshot=PRICING_SNAPSHOT,
        image_resolution=(1920, 1080),
        prompt_image_resolution=(1984, 1144),
        ruler_grounding=True,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        reasoning_setting=REASONING_SETTING,
        vlm_calls=1,
        total_wall_clock_seconds=wall_seconds,
        retries=0,
        errors=(message,),
        estimated_cost_low_usd=COST_ESTIMATE["low_usd"],
        estimated_cost_expected_usd=COST_ESTIMATE["expected_usd"],
        estimated_cost_high_usd=COST_ESTIMATE["high_usd"],
        schema_validation_success=(False if "schema" in message.lower() else None),
    )


def run_approved_test1(
    repository_root: str | Path, approved_call_count: int
) -> Test1Paths:
    """Run exactly one paid request after the caller supplies the approval count."""

    if approved_call_count != 1:
        raise PermissionError("Test 1 requires explicit approval for exactly one call")
    paths = prepare_test1(repository_root)
    provider = OpenRouterVLMProvider.from_repository_env(
        repository_root,
        model=DEFAULT_OPENROUTER_MODEL,
        reasoning_effort=REASONING_SETTING["effort"],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    _update_manifest(
        paths,
        paid_request_attempted=True,
        request_attempted_at=datetime.now(timezone.utc).isoformat(),
    )
    started = perf_counter()
    try:
        with Image.open(paths.ruled_frame) as ruled:
            result = provider.request_detection(FRAME_NUMBER, ruled.copy())
    except Exception as error:
        wall_seconds = perf_counter() - started
        JsonlExperimentLogger(paths.metrics).append(
            _record_for_failure(paths, error, wall_seconds)
        )
        _update_manifest(
            paths,
            paid_request_succeeded=False,
            request_finished_at=datetime.now(timezone.utc).isoformat(),
        )
        raise
    wall_seconds = perf_counter() - started
    paths.response.write_text(
        json.dumps(_detection_as_dict(result.detection), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with Image.open(paths.original_frame) as original:
        preview = render_detection_preview(original.copy(), result.detection)
    preview.save(paths.preview, format="PNG", optimize=True)
    JsonlExperimentLogger(paths.metrics).append(
        _record_for_success(paths, result, wall_seconds)
    )
    _update_manifest(
        paths,
        paid_request_succeeded=True,
        request_finished_at=datetime.now(timezone.utc).isoformat(),
        actual_cost_usd=result.usage.cost_usd,
        actual_input_tokens=result.usage.prompt_tokens,
        actual_output_tokens=result.usage.completion_tokens,
        actual_vlm_latency_seconds=result.latency_seconds,
        actual_total_wall_clock_seconds=wall_seconds,
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run"), help="prepare is always zero-cost"
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--approved-call-count",
        type=int,
        default=0,
        help="run requires exactly 1 after explicit user approval",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        paths = prepare_test1(args.repository_root)
    else:
        paths = run_approved_test1(args.repository_root, args.approved_call_count)
    print(paths.manifest)


if __name__ == "__main__":
    main()
