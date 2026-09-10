"""Run a fresh Gemini 3.8/3.7 Flash comparison at 2 FPS."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import gemini_5fps as engine
from .sampling import sample_frame_indices


EXPERIMENT_ID = "gemini-flash-2fps-fresh-improved"
FRAME_INTERVAL = 15
SAMPLING_FPS = 2.0
ANCHOR_FRAMES = tuple(sample_frame_indices(900, FRAME_INTERVAL, include_last=True))
PAID_CALLS = len(ANCHOR_FRAMES) * len(engine.MODELS)


def comparison_paths(repository_root: str | Path) -> engine.FiveFpsPaths:
    root = Path(repository_root).resolve()
    experiment = root / "experiments" / "gemini_2fps_fresh_improved"
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
        output_dir=root / "outputs" / "gemini_2fps_fresh_improved",
    )


def configure_engine() -> None:
    """Configure the shared restartable comparison engine for this isolated run."""

    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.COMMAND_MODULE = "track_game.gemini_2fps_fresh"
    engine.FRAME_INTERVAL = FRAME_INTERVAL
    engine.SAMPLING_FPS = SAMPLING_FPS
    engine.ANCHOR_FRAMES = ANCHOR_FRAMES
    engine.ANCHORS_PER_MODEL = len(ANCHOR_FRAMES)
    engine.PAID_CALLS = PAID_CALLS
    engine.five_fps_paths = comparison_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "render-saved"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--approved-call-count", type=int, default=0)
    args = parser.parse_args()
    configure_engine()
    if args.command == "prepare":
        print(engine.prepare(args.repository_root).cost_preflight)
    elif args.command == "run":
        print(engine.run(args.repository_root, args.approved_call_count).results_table)
    else:
        print(engine.render_saved(args.repository_root).results_table)


if __name__ == "__main__":
    main()
