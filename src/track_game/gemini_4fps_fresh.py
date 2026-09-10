"""Run a fresh Gemini 3.8/3.7 Flash comparison at 4 FPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import gemini_5fps as engine


EXPERIMENT_ID = "gemini-flash-4fps-fresh-improved"
NOMINAL_FRAME_INTERVAL = 7.5
SAMPLING_FPS = 4.0
# A 30 FPS source cannot represent a 7.5-frame interval directly. Half-up
# rounding alternates 8/7-frame gaps, then include_last preserves prior tests'
# explicit final-frame policy.
ANCHOR_FRAMES = tuple((sample * 15 + 1) // 2 for sample in range(120)) + (899,)
PAID_CALLS = len(ANCHOR_FRAMES) * len(engine.MODELS)
RETRY_FAILED_CALLS = 205
RETRY_MAX_CONCURRENCY = 2
EXPECTED_FAILED_BY_MODEL = {
    "gemini_3_8_flash": 121,
    "gemini_3_7_flash": 84,
}


def comparison_paths(repository_root: str | Path) -> engine.FiveFpsPaths:
    root = Path(repository_root).resolve()
    experiment = root / "experiments" / "gemini_4fps_fresh_improved"
    return engine.FiveFpsPaths(
        root=root,
        clip=root / "clips" / "dev" / "spurs_thunder_test.mp4",
        experiment=experiment,
        anchors=experiment / "artifacts" / "anchors",
        prompt=experiment / "prompt_template_v2.txt",
        schema=experiment / "schema_v2.json",
        catalog_snapshot=experiment / "catalog_snapshot.json",
        cost_preflight=experiment / "cost_preflight.json",
        manifest=experiment / "manifest.json",
        approval=experiment / "approval.json",
        results_table=experiment / "results_table.md",
        output_dir=root / "outputs" / "gemini_4fps_fresh_improved",
    )


def _fresh_two_fps_usage(
    paths: engine.FiveFpsPaths, model: engine.BakeoffModel
) -> list[dict[str, Any]]:
    directory = (
        paths.root
        / "experiments"
        / "gemini_2fps_fresh_improved"
        / model.key
        / "records"
    )
    rows = []
    for path in sorted(directory.glob("frame_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("input_tokens") is not None and row.get("output_tokens") is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no fresh 2 FPS token usage found for {model.slug}")
    return rows


def configure_engine() -> None:
    """Configure the shared restartable comparison engine for this isolated run."""

    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.COMMAND_MODULE = "track_game.gemini_4fps_fresh"
    engine.FRAME_INTERVAL = NOMINAL_FRAME_INTERVAL
    engine.SAMPLING_FPS = SAMPLING_FPS
    engine.ANCHOR_FRAMES = ANCHOR_FRAMES
    engine.ANCHORS_PER_MODEL = len(ANCHOR_FRAMES)
    engine.PAID_CALLS = PAID_CALLS
    engine.five_fps_paths = comparison_paths
    engine._historical_usage = _fresh_two_fps_usage


def _failed_frame_ids(
    paths: engine.FiveFpsPaths, model: engine.BakeoffModel
) -> list[int]:
    """Return only saved frames that did not produce schema-valid detections."""

    return sorted(
        int(row["frame_id"])
        for row in engine._load_records(paths, model)
        if not row.get("schema_validation_success")
    )


def _archive_failed_records(
    paths: engine.FiveFpsPaths,
    model: engine.BakeoffModel,
    frame_ids: list[int],
) -> None:
    """Move first-attempt failures aside so the restartable engine replaces them."""

    backup_root = paths.experiment / "first_attempt_failures" / model.key
    for folder in ("records", "raw_responses", "parsed_detections"):
        (backup_root / folder).mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        filename = f"frame_{frame_id:04d}.json"
        for folder in ("records", "raw_responses", "parsed_detections"):
            source = paths.model_dir(model) / folder / filename
            if not source.is_file():
                continue
            destination = backup_root / folder / filename
            if destination.exists():
                raise RuntimeError(f"retry archive already contains {destination}")
            source.replace(destination)


def retry_failed(repository_root: str | Path, approved_call_count: int) -> engine.FiveFpsPaths:
    """Replace exactly the failed first-attempt records without touching successes."""

    if approved_call_count != RETRY_FAILED_CALLS:
        raise PermissionError(
            f"failed-frame replacement requires approval for exactly {RETRY_FAILED_CALLS} calls"
        )
    paths = comparison_paths(repository_root)
    if not paths.catalog_snapshot.is_file():
        raise RuntimeError("the original prepared catalog snapshot is missing")
    stored_catalog = json.loads(paths.catalog_snapshot.read_text(encoding="utf-8"))
    live_catalog = engine.fetch_and_verify_catalog(models=engine.MODELS)
    if engine._catalog_fingerprint(live_catalog) != engine._catalog_fingerprint(
        stored_catalog
    ):
        raise PermissionError("provider or pricing changed; replacement calls are not approved")
    if engine.video_has_audio(paths.clip):
        raise RuntimeError("source clip is not silent")

    failed_by_model = {
        model.key: _failed_frame_ids(paths, model) for model in engine.MODELS
    }
    observed = {key: len(value) for key, value in failed_by_model.items()}
    if observed != EXPECTED_FAILED_BY_MODEL:
        raise RuntimeError(
            f"failed-frame counts changed; expected {EXPECTED_FAILED_BY_MODEL}, observed {observed}"
        )
    if sum(observed.values()) != approved_call_count:
        raise RuntimeError("failed-frame total does not match approved replacement calls")

    engine._write_json(
        paths.experiment / "retry_approval.json",
        {
            "approved_at": engine._utc_now(),
            "approved_call_count": approved_call_count,
            "catalog_fingerprint": engine._catalog_fingerprint(stored_catalog),
            "configured_concurrency": RETRY_MAX_CONCURRENCY,
            "automatic_retries": engine.AUTOMATIC_RETRIES,
            "failed_frames_by_model": failed_by_model,
        },
    )
    for model in engine.MODELS:
        _archive_failed_records(paths, model, failed_by_model[model.key])

    engine.MAX_CONCURRENCY = RETRY_MAX_CONCURRENCY
    summaries: dict[str, dict[str, Any]] = {}
    for model in engine.MODELS:
        records, batch_seconds = engine._run_model(paths, model)
        engine._materialize_model_artifacts(paths, model, records)
        summary = engine._summarize(paths, model, records, batch_seconds)
        summary["replacement_calls_attempted"] = len(failed_by_model[model.key])
        engine._render_pipeline(paths, model, records, summary)
        summaries[model.key] = summary
    engine._write_results(paths, summaries)

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["status"] = "completed_after_failed_frame_replacement"
    manifest["replacement_completed_at"] = engine._utc_now()
    manifest["replacement_calls_attempted"] = approved_call_count
    manifest["total_call_attempts_including_replacements"] = PAID_CALLS + approved_call_count
    manifest["outputs"] = [
        summary.get("output_video") for summary in summaries.values()
    ]
    engine._write_json(paths.manifest, manifest)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "retry-failed", "render-saved")
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--approved-call-count", type=int, default=0)
    args = parser.parse_args()
    configure_engine()
    if args.command == "prepare":
        print(engine.prepare(args.repository_root).cost_preflight)
    elif args.command == "run":
        print(engine.run(args.repository_root, args.approved_call_count).results_table)
    elif args.command == "retry-failed":
        print(
            retry_failed(args.repository_root, args.approved_call_count).results_table
        )
    else:
        print(engine.render_saved(args.repository_root).results_table)


if __name__ == "__main__":
    main()
