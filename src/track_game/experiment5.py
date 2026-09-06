"""Prepare the gated 1 FPS VLM-anchor experiment without calling OpenRouter."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .fullclip import full_clip_detection_prompt
from .ruler import add_normalized_rulers
from .sampling import sample_frame_indices
from .schema import SCHEMA_VERSION, frame_detection_json_schema
from .video import extract_video_frames, probe_video


EXPERIMENT_ID = "test-5-1fps-player-cv"
MODEL = "google/gemini-3.1-flash-lite"
FRAME_INTERVAL = 30
ANCHOR_FRAMES = tuple(sample_frame_indices(900, FRAME_INTERVAL, include_last=True))
VLM_CALLS = len(ANCHOR_FRAMES)
INPUT_PRICE_PER_MILLION = 0.25
OUTPUT_PRICE_PER_MILLION = 1.50
MAX_OUTPUT_TOKENS = 4096


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(prepare_experiment5(args.repository_root))


if __name__ == "__main__":
    main()
