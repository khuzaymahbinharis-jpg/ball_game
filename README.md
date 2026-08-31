# Track the Game

A deliberately small foundation for Zeta Solutions internship Task 02. It turns sampled, **VLM-provided** sports detections into persistent player tracks and FIFA-style Pillow overlays. Tonight's implementation is offline: the deterministic provider does not inspect the image and no API, key, model, or paid service is used.

## Architecture

```text
video -> Pillow frames -> anchor sampler -> optional prompt-only rulers
      -> structured VLM provider -> team-aware association -> interpolation
      -> Pillow markers -> encoded video
```

The separation is an important constraint. A future VLM is responsible for all semantic interpretation: identifying players, teams, the ball, and possession. `NearestNeighbourTracker` only associates already-localized detections. There is no object detector, segmentation model, or classical semantic detector.

### Package map

- `schema.py`: strict normalized response boundary (`FrameDetection`, player boxes, ball and confidence).
- `ruler.py`: optional visible 0–1 axes for the prompt image and coordinate conversion against the untouched frame.
- `sampling.py`: deterministic anchor selection, including the last frame when configured.
- `provider.py`: provider protocol plus deterministic mock.
- `tracking.py`: greedy nearest-foot association constrained by team, explicit integer track IDs, missed-anchor tolerance, and linear interpolation.
- `drawing.py`: colored under-player ellipses, possession indicator, highlighted ball, and optional IDs.
- `video.py`: local `ffmpeg` decode/encode boundary.
- `experiments.py`: append-only JSONL records whose unknown real measurements remain `null`.
- `pipeline.py`: in-memory mock end-to-end composition.

## Setup and tests

Python 3.11+, Pillow, and pytest are required. `ffmpeg`/`ffprobe` are needed only for real local video I/O.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

An offline smoke test can construct Pillow frames and call `MockPipeline.process_frames`. Inputs are copied before rulers and drawing; final annotations therefore never contain prompt rulers.

## Configuration and ablations

`PipelineConfig` groups sampling, ruler, model, and tracking settings. Record each run with `ExperimentRecord` in `experiments/*.jsonl` (ignored by Git). For controlled experiments, copy a baseline configuration and change exactly one field: model, `every_n_frames`, `ruler.enabled`, matching distance, or team constraint.

## Tomorrow: adding real inference safely

1. Add a separate OpenRouter provider implementing `VLMProvider`. It should accept an explicitly supplied credential at runtime, request JSON matching `FrameDetection`, validate it, retry only bounded transient/validation failures, and never log the key.
2. Keep the configured Gemini-family model name in `ModelConfig`; do not move HTTP logic into tracking or the pipeline.
3. Batch/parallelize independent anchor requests with bounded concurrency, then reorder by `frame_id` before tracking.
4. Decode a short local clip through `read_video_frames`, run the pipeline, then use `write_video_frames`. Preserve audio in a later, explicit muxing step.
5. Log observed latency, token usage, price, retries, and errors from provider metadata. Never estimate missing values: leave them `null`.
6. Inspect identity switches, occlusion, camera cuts, ball misses, team confusion, and interpolation drift. Only after the basic method is measured should Hungarian assignment, motion prediction, optical flow, or correlation tracking be considered.

Current limitations are intentional: greedy matching has no velocity model; tracks visible in only one endpoint disappear between anchors; linear interpolation cannot follow nonlinear motion or cuts; the mock provider demonstrates plumbing rather than visual quality; video encoding currently omits audio. No benchmark, latency, cost, or quality claim is made.

