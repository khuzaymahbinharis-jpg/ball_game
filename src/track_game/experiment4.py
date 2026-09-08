"""Zero-cost 2 FPS current-improved versus player-CV tracking ablation."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from .ball_tracking import VLMInitializedBallTracker, apply_ball_track
from .comparison import ComparisonVariant, cv_improved_variant, improved_variant
from .experiments import TrackingMetricsRecord, TrackingMetricsWriter
from .player_cv_tracking import (
    VLMInitializedPlayerCVTracker,
    camera_log_as_dict,
    scene_cut_log_as_dict,
)
from .schema import FrameDetection
from .shot_context import trackable_ball_anchors
from .tracking import TrackedFrame, build_player_tracker, interpolate_sequence
from .video import probe_video, render_tracked_video


EXPERIMENT_ID = "test-4-player-cv-tracking-ablation"
MODEL = "google/gemini-3.1-flash-lite"
PLANNED_ANCHORS = 61


@dataclass(frozen=True)
class Experiment4Paths:
    root: Path
    clip: Path
    saved_responses: Path
    prior_summary: Path
    experiment_dir: Path
    preflight: Path
    current_metrics: Path
    cv_metrics: Path
    summary: Path
    camera_log: Path
    cut_log: Path
    confidence_log: Path
    current_video: Path
    cv_video: Path


def experiment4_paths(repository_root: str | Path) -> Experiment4Paths:
    root = Path(repository_root).resolve()
    experiment_dir = root / "experiments" / "test_4_player_cv_tracking_ablation"
    return Experiment4Paths(
        root=root,
        clip=root / "clips" / "dev" / "spurs_thunder_test.mp4",
        saved_responses=root / "experiments" / "test_3_2fps_comparison" / "artifacts" / "responses",
        prior_summary=root / "experiments" / "test_3_2fps_comparison" / "summary.json",
        experiment_dir=experiment_dir,
        preflight=experiment_dir / "preflight.json",
        current_metrics=experiment_dir / "current_improved_metrics.json",
        cv_metrics=experiment_dir / "cv_improved_metrics.json",
        summary=experiment_dir / "summary.json",
        camera_log=experiment_dir / "camera_motion.jsonl",
        cut_log=experiment_dir / "scene_cuts.jsonl",
        confidence_log=experiment_dir / "track_confidence.jsonl",
        current_video=root / "outputs" / "spurs_thunder_test4_current_improved.mp4",
        cv_video=root / "outputs" / "spurs_thunder_test4_player_cv_improved.mp4",
    )


def _load_saved_detections(paths: Experiment4Paths) -> dict[int, FrameDetection]:
    detections: dict[int, FrameDetection] = {}
    for response_path in paths.saved_responses.glob("frame_*_response.json"):
        detection = FrameDetection.from_dict(
            json.loads(response_path.read_text(encoding="utf-8"))
        )
        detections[detection.frame_id] = detection
    if len(detections) < 2:
        raise RuntimeError("Test 4 requires saved Test 3 VLM responses")
    return detections


def _covered_timeline(timeline: list[TrackedFrame], total_frames: int) -> list[TrackedFrame]:
    if not timeline:
        raise RuntimeError("cannot cover an empty timeline")
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
        raise RuntimeError("timeline does not cover the complete clip")
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _variant_manifest(variant: ComparisonVariant) -> dict[str, Any]:
    return {
        "name": variant.name,
        "description": variant.description,
        "pipeline": variant.pipeline.as_dict(),
    }


def prepare_experiment4(repository_root: str | Path) -> Experiment4Paths:
    paths = experiment4_paths(repository_root)
    info = probe_video(paths.clip)
    detections = _load_saved_detections(paths)
    current, cv = improved_variant(), cv_improved_variant()
    paths.experiment_dir.mkdir(parents=True, exist_ok=True)
    paths.preflight.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "prepared_at": datetime.now(timezone.utc).isoformat(),
                "source_clip": str(paths.clip.relative_to(paths.root)),
                "clip": asdict(info),
                "model": MODEL,
                "planned_anchor_count": PLANNED_ANCHORS,
                "saved_valid_anchor_count": len(detections),
                "saved_anchor_frames": sorted(detections),
                "new_paid_requests": 0,
                "shared_detection_source": str(paths.saved_responses.relative_to(paths.root)),
                "controlled_variables": [
                    "clip",
                    "model",
                    "ruler",
                    "prompt",
                    "schema",
                    "saved VLM responses",
                    "Hungarian association",
                    "soft team history",
                    "persistent logical identities",
                    "existing ball tracker",
                ],
                "changed_variables": [
                    "player sparse-LK tracking",
                    "camera-motion compensation",
                    "scene-cut spatial reset",
                    "persistent track confidence",
                ],
                "variants": [_variant_manifest(current), _variant_manifest(cv)],
                "human_evaluation": {
                    "manual_id_switches": None,
                    "marker_drift_events": None,
                    "player_tracking_failures": None,
                    "scene_cut_failures": None,
                    "ball_failures": None,
                    "overall_visual_quality_notes": None,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _prior_api_metrics(paths: Experiment4Paths) -> dict[str, Any]:
    prior = json.loads(paths.prior_summary.read_text(encoding="utf-8"))
    return {
        "attempted": prior["attempted_vlm_calls"],
        "actual_cost": prior["actual_api_cost_usd"],
        "cost_complete": prior["api_cost_complete"],
        "summed_latency": prior["baseline"]["summed_vlm_latency_seconds"],
        "batch_seconds": prior["api_batch_seconds"],
    }


def render_saved_ablation(repository_root: str | Path) -> Experiment4Paths:
    """Render both variants from saved detections; this function has no provider path."""

    paths = prepare_experiment4(repository_root)
    detections = _load_saved_detections(paths)
    ordered = [detections[index] for index in sorted(detections)]
    api = _prior_api_metrics(paths)
    current_variant, cv_variant = improved_variant(), cv_improved_variant()

    current_started = perf_counter()
    current_tracker = build_player_tracker(current_variant.pipeline.tracking)
    current_anchors = [current_tracker.update(item) for item in ordered]
    current_timeline = _covered_timeline(interpolate_sequence(current_anchors), 900)
    current_ball_started = perf_counter()
    current_ball = VLMInitializedBallTracker(
        current_variant.pipeline.ball_tracking
    ).track_video(
        paths.clip,
        {frame: detection.ball_detection for frame, detection in detections.items()},
        current_timeline,
    )
    current_ball_seconds = perf_counter() - current_ball_started
    current_timeline = apply_ball_track(current_timeline, current_ball)
    current_render_started = perf_counter()
    render_tracked_video(paths.clip, paths.current_video, current_timeline, show_ids=True)
    current_render_seconds = perf_counter() - current_render_started
    current_seconds = perf_counter() - current_started

    cv_started = perf_counter()
    cv_tracker = build_player_tracker(cv_variant.pipeline.tracking)
    player_pass_started = perf_counter()
    player_result = VLMInitializedPlayerCVTracker(
        cv_variant.pipeline, cv_tracker
    ).track_video(paths.clip, detections)
    player_pass_seconds = perf_counter() - player_pass_started
    cut_frames = {
        item.frame_id for item in player_result.scene_cuts if item.is_cut
    }
    cv_timeline = list(player_result.timeline)
    cv_ball_started = perf_counter()
    cv_ball = VLMInitializedBallTracker(cv_variant.pipeline.ball_tracking).track_video(
        paths.clip,
        trackable_ball_anchors(detections, player_result.shot_context),
        cv_timeline,
        scene_cut_frames=cut_frames,
    )
    cv_ball_seconds = perf_counter() - cv_ball_started
    cv_timeline = apply_ball_track(cv_timeline, cv_ball)
    cv_render_started = perf_counter()
    render_tracked_video(paths.clip, paths.cv_video, cv_timeline, show_ids=True)
    cv_render_seconds = perf_counter() - cv_render_started
    cv_seconds = perf_counter() - cv_started

    _write_jsonl(paths.camera_log, camera_log_as_dict(player_result))
    _write_jsonl(paths.cut_log, scene_cut_log_as_dict(player_result))
    _write_jsonl(paths.confidence_log, player_result.confidence_history)

    def record(
        variant: ComparisonVariant,
        tracker: Any,
        anchors: list[TrackedFrame],
        local_seconds: float,
        render_seconds: float,
        ball_stats: Any,
        *,
        cv_enabled: bool,
        ball_seconds: float,
    ) -> TrackingMetricsRecord:
        lifecycle = tracker.lifecycle
        stats = player_result.stats if cv_enabled else None
        camera_successes = None if not cv_enabled else sum(
            item.success for item in player_result.camera_motion
        )
        camera_failures = None if not cv_enabled else sum(
            item.frame_id > 0 and not item.success for item in player_result.camera_motion
        )
        return TrackingMetricsRecord(
            experiment_id=EXPERIMENT_ID,
            variant=variant.name,
            model=MODEL,
            anchor_count=PLANNED_ANCHORS,
            planned_vlm_calls=PLANNED_ANCHORS,
            attempted_vlm_calls=api["attempted"],
            successful_vlm_calls=len(detections),
            actual_api_cost_usd=api["actual_cost"],
            api_cost_complete=api["cost_complete"],
            summed_vlm_latency_seconds=api["summed_latency"],
            api_batch_seconds=api["batch_seconds"],
            local_processing_seconds=local_seconds,
            total_processing_seconds=api["batch_seconds"] + local_seconds,
            unique_ids_over_time={
                frame.frame_id: tuple(player.track_id for player in frame.players)
                for frame in anchors
            },
            tracks_created=lifecycle.tracks_created,
            tracks_expired=lifecycle.tracks_expired,
            tracks_lost=lifecycle.tracks_lost,
            tracks_recovered=lifecycle.tracks_recovered,
            ball_lost_events=ball_stats.lost_events,
            ball_recovered_events=ball_stats.recovered_events,
            ball_lost_frames=ball_stats.lost_frames,
            uncertain_track_frames=None if stats is None else stats.uncertain_track_frames,
            uncertainty_recoveries=None if stats is None else stats.uncertainty_recoveries,
            scene_cut_frames=tuple(sorted(cut_frames)) if cv_enabled else (),
            camera_motion_successes=camera_successes,
            camera_motion_failures=camera_failures,
            camera_motion_seconds=None if stats is None else stats.camera_seconds,
            player_cv_tracking_seconds=None if stats is None else stats.player_tracking_seconds,
            player_cv_total_pass_seconds=None if stats is None else player_pass_seconds,
            scene_cut_detection_seconds=None if stats is None else stats.scene_cut_seconds,
            ball_tracking_seconds=ball_seconds,
            rendering_seconds=render_seconds,
            average_local_seconds_per_frame=local_seconds / 900,
        )

    current_record = record(
        current_variant,
        current_tracker,
        current_anchors,
        current_seconds,
        current_render_seconds,
        current_ball.stats,
        cv_enabled=False,
        ball_seconds=current_ball_seconds,
    )
    cv_anchor_frames = [
        frame for frame in cv_timeline if frame.frame_id in detections
    ]
    cv_record = record(
        cv_variant,
        cv_tracker,
        cv_anchor_frames,
        cv_seconds,
        cv_render_seconds,
        cv_ball.stats,
        cv_enabled=True,
        ball_seconds=cv_ball_seconds,
    )
    TrackingMetricsWriter(paths.current_metrics).write(current_record)
    TrackingMetricsWriter(paths.cv_metrics).write(cv_record)
    paths.summary.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "new_paid_requests": 0,
                "saved_anchor_responses_reused": len(detections),
                "current_improved": asdict(current_record),
                "cv_improved": asdict(cv_record),
                "detailed_logs": {
                    "camera_motion": str(paths.camera_log.relative_to(paths.root)),
                    "scene_cuts": str(paths.cut_log.relative_to(paths.root)),
                    "track_confidence": str(paths.confidence_log.relative_to(paths.root)),
                },
                "machine_metrics_are_not_manual_id_switch_counts": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "render-saved"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    paths = (
        prepare_experiment4(args.repository_root)
        if args.command == "prepare"
        else render_saved_ablation(args.repository_root)
    )
    print(paths.summary if args.command == "render-saved" else paths.preflight)


if __name__ == "__main__":
    main()
