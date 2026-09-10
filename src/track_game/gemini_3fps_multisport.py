"""Run Gemini 3.8 Flash on five 30-second sports clips at 3 FPS."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from . import gemini_5fps as engine
from . import model_bakeoff
from .comparison import cv_improved_variant
from .provider import detection_prompt


EXPERIMENT_ID = "gemini-3.8-flash-3fps-multisport"
SAMPLING_FPS = 3.0
FRAME_INTERVAL = 10
ANCHOR_FRAMES = tuple(range(0, 900, FRAME_INTERVAL)) + (899,)
ANCHORS_PER_CLIP = len(ANCHOR_FRAMES)
MAX_CONCURRENCY = 2
LOCAL_RENDER_WORKERS = 2
MODEL = engine.MODELS[0]


@dataclass(frozen=True)
class ClipSpec:
    key: str
    filename: str
    sport: str


CLIPS = (
    ClipSpec("football_amateur", "football_amateur_1080.mp4", "association football"),
    ClipSpec("allstars_fr_eng", "allstars_fr_eng.mp4", "association football"),
    ClipSpec("basketball", "basketball.mp4", "basketball"),
    ClipSpec("football_cuts", "football_cuts_1080.mp4", "association football"),
    ClipSpec("volleyball", "volleyball_1080.mp4", "volleyball"),
)
PAID_CALLS = ANCHORS_PER_CLIP * len(CLIPS)


def configure_engine() -> None:
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.COMMAND_MODULE = "track_game.gemini_3fps_multisport"
    engine.FRAME_INTERVAL = FRAME_INTERVAL
    engine.SAMPLING_FPS = SAMPLING_FPS
    engine.ANCHOR_FRAMES = ANCHOR_FRAMES
    engine.ANCHORS_PER_MODEL = ANCHORS_PER_CLIP
    engine.MAX_CONCURRENCY = MAX_CONCURRENCY
    engine.MODELS = (MODEL,)
    engine.PAID_CALLS = PAID_CALLS


def experiment_root(repository_root: str | Path) -> Path:
    return Path(repository_root).resolve() / "experiments" / "gemini_3fps_multisport"


def clip_paths(repository_root: str | Path, spec: ClipSpec) -> engine.FiveFpsPaths:
    root = Path(repository_root).resolve()
    experiment = experiment_root(root) / spec.key
    return engine.FiveFpsPaths(
        root=root,
        clip=root / "clips" / "multisport" / spec.filename,
        experiment=experiment,
        anchors=experiment / "artifacts" / "anchors",
        prompt=experiment / "prompt_template_v1.txt",
        schema=experiment / "schema_v2.json",
        catalog_snapshot=experiment_root(root) / "catalog_snapshot.json",
        cost_preflight=experiment_root(root) / "cost_preflight.json",
        manifest=experiment / "manifest.json",
        approval=experiment_root(root) / "approval.json",
        results_table=experiment / "results_table.md",
        output_dir=root / "outputs" / "gemini_3fps_multisport" / spec.key,
    )


def multisport_detection_prompt(frame_id: int, sport: str) -> str:
    """Adapt the existing strict detector prompt without hard-coded NBA teams."""

    prompt = detection_prompt(frame_id)
    prompt = prompt.replace("basketball video frame", f"{sport} video frame")
    prompt = prompt.replace("on-court player", "active player")
    prompt = prompt.replace("For the basketball", "For the game ball")
    return prompt + """

Clip-wide team convention required for temporal consistency:
- Team A is the visually lighter/brighter uniform group.
- Team B is the visually darker/more saturated uniform group.
Use uniform appearance, not screen position, to preserve this convention.
Officials are not players. For sports without individual ball possession, return
possession=null unless one returned player clearly controls the ball."""


def _historical_rows(root: Path) -> list[dict[str, Any]]:
    records = (
        root
        / "experiments"
        / "gemini_4fps_fresh_improved"
        / MODEL.key
        / "records"
    )
    rows = []
    for path in sorted(records.glob("frame_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("input_tokens") is not None and row.get("output_tokens") is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("no Gemini 3.8 measured token history is available")
    return rows


def build_cost_preflight(root: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    rows = _historical_rows(root)
    inputs = [float(row["input_tokens"]) for row in rows]
    outputs = [float(row["output_tokens"]) for row in rows]
    costs = [
        float(row["actual_openrouter_cost_usd"])
        for row in rows
        if row.get("actual_openrouter_cost_usd") is not None
    ]
    pricing = catalog["models"][0]["pricing_per_million"]
    input_rate = float(pricing["input"])
    output_rate = float(pricing["output"])

    def token_cost(calls: int, input_tokens: float, output_tokens: float) -> float:
        return calls * (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    expected_per_call = token_cost(
        1, statistics.mean(inputs), statistics.mean(outputs)
    )
    return {
        "captured_at": catalog["captured_at"],
        "catalog_fingerprint": engine._catalog_fingerprint(catalog),
        "model": MODEL.slug,
        "provider": catalog["models"][0]["chosen_provider"],
        "pricing_per_million_usd": pricing,
        "clip_count": len(CLIPS),
        "calls_per_clip": ANCHORS_PER_CLIP,
        "paid_calls_requiring_approval": PAID_CALLS,
        "expected_cost_per_call_usd": round(expected_per_call, 12),
        "per_clip": {
            "expected_usd": round(expected_per_call * ANCHORS_PER_CLIP, 9),
            "observed_low_usd": round(min(costs) * ANCHORS_PER_CLIP, 9),
            "observed_high_usd": round(max(costs) * ANCHORS_PER_CLIP, 9),
            "absolute_ceiling_usd": round(
                token_cost(ANCHORS_PER_CLIP, max(inputs), engine.MAX_OUTPUT_TOKENS), 9
            ),
        },
        "combined": {
            "expected_usd": round(expected_per_call * PAID_CALLS, 9),
            "observed_low_usd": round(min(costs) * PAID_CALLS, 9),
            "observed_high_usd": round(max(costs) * PAID_CALLS, 9),
            "absolute_ceiling_usd": round(
                token_cost(PAID_CALLS, max(inputs), engine.MAX_OUTPUT_TOKENS), 9
            ),
        },
        "basis": "Gemini 3.8 Flash measured 4 FPS records on the prior 1080p clip.",
    }


def fetch_key_usage(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    load_dotenv(root / ".env", override=False)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    response = httpx.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json().get("data", {})
    return {
        key: data.get(key)
        for key in (
            "usage",
            "usage_daily",
            "usage_weekly",
            "usage_monthly",
            "limit",
            "limit_remaining",
            "is_free_tier",
            "byok_usage",
        )
    }


def prepare(
    repository_root: str | Path, source_directory: str | Path
) -> engine.FiveFpsPaths:
    """Copy verified muted clips and prepare anchors without inference calls."""

    configure_engine()
    root = Path(repository_root).resolve()
    source_dir = Path(source_directory).resolve()
    top = experiment_root(root)
    top.mkdir(parents=True, exist_ok=True)
    catalog = engine.fetch_and_verify_catalog(models=(MODEL,))
    preflight = build_cost_preflight(root, catalog)
    engine._write_json(top / "catalog_snapshot.json", catalog)
    engine._write_json(top / "cost_preflight.json", preflight)

    prepared = []
    for spec in CLIPS:
        paths = clip_paths(root, spec)
        source = source_dir / spec.filename
        if not source.is_file():
            raise FileNotFoundError(source)
        if engine.video_has_audio(source):
            engine.mute_video_in_place(source)
        paths.clip.parent.mkdir(parents=True, exist_ok=True)
        if paths.clip.is_file() and engine._sha256(paths.clip) != engine._sha256(source):
            if list((paths.model_dir(MODEL) / "records").glob("frame_*.json")):
                raise RuntimeError(f"cannot replace {paths.clip}; detector records exist")
            shutil.copy2(source, paths.clip)
        elif not paths.clip.is_file():
            shutil.copy2(source, paths.clip)
        engine.mute_video_in_place(paths.clip)
        info = engine.probe_video(paths.clip)
        if (
            info.frame_count != 900
            or abs(info.duration_seconds - 30.0) > 0.02
            or abs(info.fps - 30.0) > 0.01
        ):
            raise RuntimeError(f"{spec.filename} is not a verified 900-frame clip")

        paths.experiment.mkdir(parents=True, exist_ok=True)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        engine._prepare_anchor_images(paths)
        prompt = multisport_detection_prompt(0, spec.sport).replace(
            "must be 0", "must be {frame_id}"
        )
        paths.prompt.write_text(prompt + "\n", encoding="utf-8")
        paths.schema.write_text(
            json.dumps(engine.frame_detection_json_schema(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        for name in ("records", "raw_responses", "parsed_detections"):
            (paths.model_dir(MODEL) / name).mkdir(parents=True, exist_ok=True)
        engine._write_json(
            paths.model_dir(MODEL) / "config.json",
            {
                "experiment_id": f"{EXPERIMENT_ID}-{spec.key}",
                "model": MODEL.slug,
                "sport": spec.sport,
                "source_resolution": [info.width, info.height],
                "anchor_frames": list(ANCHOR_FRAMES),
                "sampling_fps": SAMPLING_FPS,
                "automatic_retries": engine.AUTOMATIC_RETRIES,
                "request_timeout_seconds": engine.REQUEST_TIMEOUT_SECONDS,
                "max_concurrency": MAX_CONCURRENCY,
                "reasoning_setting": MODEL.reasoning_setting,
                "provider_routing": {
                    "only": [MODEL.provider_slug],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                "player_and_ball_pipeline": cv_improved_variant().pipeline.as_dict(),
            },
        )
        engine._write_json(
            paths.manifest,
            {
                "experiment_id": f"{EXPERIMENT_ID}-{spec.key}",
                "status": "prepared_not_approved",
                "sport": spec.sport,
                "source_clip": str(paths.clip.relative_to(root)),
                "source_sha256": engine._sha256(paths.clip),
                "source_audio_muted": not engine.video_has_audio(paths.clip),
                "source_resolution": [info.width, info.height],
                "sampling_fps": SAMPLING_FPS,
                "anchor_frames": list(ANCHOR_FRAMES),
                "paid_calls": ANCHORS_PER_CLIP,
            },
        )
        prepared.append(spec.key)

    engine._write_json(
        top / "manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "prepared_not_approved",
            "clips": prepared,
            "model": asdict(MODEL),
            "paid_calls_requiring_approval": PAID_CALLS,
            "balance_checks": "before and after every clip",
            "automatic_retries": engine.AUTOMATIC_RETRIES,
        },
    )
    return clip_paths(root, CLIPS[0])


def _write_balance_event(top: Path, phase: str, spec: ClipSpec, usage: dict[str, Any]) -> None:
    path = top / "key_usage_checks.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "captured_at": engine._utc_now(),
                    "phase": phase,
                    "clip": spec.key,
                    **usage,
                },
                sort_keys=True,
            )
            + "\n"
        )


def _write_combined_results(root: Path, summaries: dict[str, dict[str, Any]]) -> None:
    top = experiment_root(root)
    lines = [
        "# Gemini 3.8 Flash multisport comparison at 3 FPS",
        "",
        "| Clip | Sport | Valid | Cost | Mean latency | Output |",
        "|---|---|---:|---:|---:|---|",
    ]
    for spec in CLIPS:
        summary = summaries[spec.key]
        cost = summary.get("actual_total_cost_usd")
        if cost is None:
            cost = summary.get("known_cost_usd")
        latency = (summary.get("latency_seconds") or {}).get("mean")
        lines.append(
            f"| {spec.filename} | {spec.sport} | "
            f"{summary.get('valid_responses', 0)}/{summary.get('calls_attempted', 0)} | "
            f"${float(cost or 0):.6f} | "
            f"{'' if latency is None else f'{latency:.3f}s'} | "
            f"{summary.get('output_video', '')} |"
        )
    (top / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(repository_root: str | Path, approved_call_count: int) -> Path:
    """Run at most the explicitly approved calls, checking balance per clip."""

    configure_engine()
    if approved_call_count != PAID_CALLS:
        raise PermissionError(f"requires approval for exactly {PAID_CALLS} calls")
    root = Path(repository_root).resolve()
    top = experiment_root(root)
    stored_catalog = json.loads((top / "catalog_snapshot.json").read_text(encoding="utf-8"))
    live_catalog = engine.fetch_and_verify_catalog(models=(MODEL,))
    if engine._catalog_fingerprint(live_catalog) != engine._catalog_fingerprint(
        stored_catalog
    ):
        raise PermissionError("provider or pricing changed; prepare and approve again")
    preflight = json.loads((top / "cost_preflight.json").read_text(encoding="utf-8"))
    expected_per_call = float(preflight["expected_cost_per_call_usd"])
    engine._write_json(
        top / "approval.json",
        {
            "approved_at": engine._utc_now(),
            "approved_call_count": approved_call_count,
            "catalog_fingerprint": engine._catalog_fingerprint(stored_catalog),
        },
    )

    summaries: dict[str, dict[str, Any]] = {}
    for spec in CLIPS:
        paths = clip_paths(root, spec)
        if engine.video_has_audio(paths.clip):
            raise RuntimeError(f"source clip is not silent: {paths.clip}")
        existing = engine._load_records(paths, MODEL)
        if len(existing) > ANCHORS_PER_CLIP:
            raise RuntimeError(f"too many records for {spec.key}")
        remaining = ANCHORS_PER_CLIP - len(existing)
        before = fetch_key_usage(root)
        _write_balance_event(top, "before", spec, before)
        balance = before.get("limit_remaining")
        if balance is not None and float(balance) < expected_per_call * remaining:
            raise RuntimeError(
                f"remaining key balance ${float(balance):.6f} is below the expected "
                f"${expected_per_call * remaining:.6f} for {spec.key}"
            )

        engine.EXPERIMENT_ID = f"{EXPERIMENT_ID}-{spec.key}"
        model_bakeoff.full_clip_detection_prompt = (
            lambda frame_id, sport=spec.sport: multisport_detection_prompt(frame_id, sport)
        )
        records, batch_seconds = engine._run_model(paths, MODEL)
        engine._materialize_model_artifacts(paths, MODEL, records)
        summary = engine._summarize(paths, MODEL, records, batch_seconds)
        summaries[spec.key] = summary
        after = fetch_key_usage(root)
        _write_balance_event(top, "after", spec, after)
        clip_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        clip_manifest["status"] = "inference_completed"
        clip_manifest["calls_attempted"] = len(records)
        clip_manifest["valid_responses"] = summary["valid_responses"]
        clip_manifest["key_usage_before"] = before
        clip_manifest["key_usage_after"] = after
        engine._write_json(paths.manifest, clip_manifest)

    def render_one(spec: ClipSpec) -> tuple[str, dict[str, Any]]:
        paths = clip_paths(root, spec)
        records = engine._load_records(paths, MODEL)
        summary = summaries[spec.key]
        engine._render_pipeline(paths, MODEL, records, summary)
        engine._write_results(paths, {MODEL.key: summary})
        clip_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        clip_manifest["status"] = "completed"
        clip_manifest["completed_at"] = engine._utc_now()
        clip_manifest["output_video"] = summary.get("output_video")
        engine._write_json(paths.manifest, clip_manifest)
        return spec.key, summary

    with ThreadPoolExecutor(max_workers=LOCAL_RENDER_WORKERS) as executor:
        futures = {executor.submit(render_one, spec): spec.key for spec in CLIPS}
        for future in as_completed(futures):
            key, summary = future.result()
            summaries[key] = summary

    _write_combined_results(root, summaries)
    manifest = json.loads((top / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest["completed_at"] = engine._utc_now()
    manifest["paid_inference_calls_attempted"] = sum(
        summary["calls_attempted"] for summary in summaries.values()
    )
    manifest["known_cost_usd"] = sum(
        float(summary.get("known_cost_usd") or 0) for summary in summaries.values()
    )
    engine._write_json(top / "manifest.json", manifest)
    return top / "results_table.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=Path(os.environ.get("USERPROFILE", "")) / "Downloads",
    )
    parser.add_argument("--approved-call-count", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        paths = prepare(args.repository_root, args.source_directory)
        print(paths.cost_preflight)
    else:
        print(run(args.repository_root, args.approved_call_count))


if __name__ == "__main__":
    main()
