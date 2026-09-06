# Track the Game

A deliberately small foundation for Zeta Solutions internship Task 02. It turns sampled, **VLM-provided** sports detections into persistent player tracks and FIFA-style Pillow overlays. Semantic interpretation stays at the VLM boundary; the optional improved path uses only post-detection data association and VLM-initialized classical optical flow.

## Architecture

```text
video -> Pillow frames -> anchor sampler -> optional prompt-only rulers
      -> structured VLM provider -> team-aware anchor association
      -> VLM-initialized player/ball flow + camera motion + cut guards
      -> confidence-aware Pillow markers -> encoded video
```

The separation is a hard constraint. A VLM through OpenRouter is responsible for all semantic interpretation: identifying players, teams, the ball, and possession. Classical CV may follow only player/ball regions already localized by the VLM, perform geometric association, or estimate non-semantic global image motion and scene changes. It cannot discover a new player or ball, assign a team from jersey pixels, or replace the VLM detector. There is no object detector, segmentation model, or classical semantic detector.

### Package map

- `schema.py`: strict normalized response boundary plus the JSON Schema sent as OpenRouter structured output.
- `ruler.py`: optional visible 0–1 axes for the prompt image and coordinate conversion against the untouched frame.
- `sampling.py`: deterministic anchor selection, including the last frame when configured.
- `provider.py`: deterministic mock plus a one-request OpenRouter adapter with no automatic retries.
- `tracking.py`: configurable greedy or Hungarian association, persistent IDs, soft stabilized team evidence, and active/lost/expired lifecycle accounting.
- `camera_motion.py`: generic Shi–Tomasi/LK keypoints, robust translation or partial-affine estimation, confidence, and identity fallback.
- `player_cv_tracking.py`: one batched sparse-LK call per frame over features inside VLM player boxes, camera fallback, bounded coasting, anchor correction, and confidence logging. It has no path for discovering a player.
- `scene_cut.py`: non-semantic frame-difference and grayscale-histogram cut scoring.
- `ball_tracking.py`: VLM-initialized pyramidal LK tracking with optional Kalman smoothing, bounded prediction, and explicit lost/recovered states.
- `comparison.py`: fixed 2 FPS baseline and improved configurations for controlled A/B runs.
- `drawing.py`: colored under-player ellipses, possession indicator, highlighted ball, and optional IDs.
- `video.py`: local ffmpeg trim, probe, frame extraction, decode, and encode boundary.
- `experiments.py`: append-only JSONL records for usage, cost, latency, validation, and later tracking evaluation.
- `test1.py`: zero-cost artifact preparation, explicit one-call approval gate, response validation, and preview rendering.
- `pipeline.py`: in-memory mock end-to-end composition.

## Setup and tests

Python 3.11+ is required. The project dependency set includes Pillow, HTTP/environment support, and an isolated ffmpeg binary for reproducible local video work.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. An offline smoke test can construct Pillow frames and call `MockPipeline.process_frames`. Inputs are copied before rulers and drawing; final annotations therefore never contain prompt rulers.

## Test 1: one ruler-grounded frame

The local source was trimmed from 3:11 through 3:41 and normalized to exactly 30.000 seconds, 30 FPS, and 900 frames at `clips/dev/spurs_thunder_test.mp4`. Clips and generated image artifacts are ignored by Git.

Prepare or refresh all zero-cost Test 1 artifacts:

```bash
python -m track_game.test1 prepare
```

This writes the untouched midpoint frame (frame 450), the 64-pixel top/left ruler image, versioned prompt, strict JSON Schema, and preflight manifest under `experiments/test_1/`. The prompt explicitly defines all normalized coordinates relative to the untouched 1920×1080 content, not the 1984×1144 ruler canvas.

Before an approved real run, create a repository-root `.env` containing:

```dotenv
OPENROUTER_API_KEY=...
```

`.env` is ignored by Git and the application never serializes or logs the key. Preparing artifacts is not approval to run. After a fresh model-pricing preflight and explicit approval for one request, the single-call command is:

```bash
python -m track_game.test1 run --approved-call-count 1
```

The runner refuses any approved count other than exactly one and never retries a failed request automatically. It sends only the ruler-enhanced frame, validates the structured response, renders the result on the original frame with Pillow, and appends returned usage/cost fields to `experiments/test_1/runs.jsonl`. Test 1 returned seven players, a ball, and a possession player with valid schema; its measured cost was $0.00220325.

## Configuration and next steps

`PipelineConfig` groups sampling, ruler, model, and tracking settings. Controlled experiments should change exactly one field at a time. After Test 1, first evaluate its response and preview; then use observed token usage, latency, cost, and grounding failures to decide the next single-variable experiment. Do not jump directly to a full-video run.

The tracking controls are independent: `association_method` selects greedy or Hungarian matching; `team_constraint` selects hard or soft team treatment; `player_cv_tracking.enabled` selects sparse-LK player following instead of interpolation; and camera motion, scene cuts, track confidence, and the existing ball tracker each have their own `enabled` flag. There is deliberately no opaque all-in-one smart-tracker switch.

## Prepared full-clip Test 2

Test 2 samples 11 ruler-enhanced anchors at frames 0, 90, …, 810, and 899 (three-second spacing plus the exact final frame). It fixes Team A as Oklahoma City blue and Team B as San Antonio black/white across every independent request, associates explicit foot points with the baseline tracker, linearly interpolates all 900 frames, draws compact transparent layers with Pillow, and uses ffmpeg only for compositing/encoding and audio preservation.

```bash
python -m track_game.fullclip prepare
```

Preparation is zero-cost. The paid runner requires a separate preflight approval for exactly 11 calls and refuses a duplicate batch when call records already exist:

```bash
python -m track_game.fullclip run --approved-call-count 11
```

The approved Test 2 batch made exactly 11 calls with no retries. Nine anchors passed strict validation; frames 630 and 810 failed validation. A degraded output was rendered from the nine valid saved anchors, spanning all 900 frames and preserving audio. The known cost from successful responses is $0.02308575; total cost is incomplete because the two invalid responses' usage was not retained by the original adapter. Measured API batch plus final local render time was 21.74 seconds.

Current limitations are intentional: greedy matching has no velocity model; tracks visible in only one endpoint disappear between anchors; player interpolation cannot follow nonlinear motion or cuts; and optical flow declares the ball lost when its VLM-localized patch cannot be tracked reliably. Test 1 is an interface/grounding result, not a tracking benchmark.

## Prepared 2 FPS tracker comparison

The next experiment holds the clip, model, ruler prompt, schema, and 15-frame sampling interval fixed. One shared set of 61 VLM anchors feeds both the baseline (greedy, hard team constraint, linear ball interpolation) and improved path (Hungarian multi-cue association, soft team history, lost-track reassociation, and VLM-initialized LK/Kalman ball tracking). This avoids paying for duplicate detections.

```bash
python -m track_game.experiment3 prepare
```

Preparation is local and cannot call OpenRouter. The paid command is separately gated and must not be run until its current model, exact 61-call count, pricing estimate, and experiment purpose have been shown and explicitly approved:

```bash
python -m track_game.experiment3 run --approved-call-count 61
```

Per-anchor usage is append-only. Each variant writes machine metrics for IDs over time, track creation/loss/recovery/expiration, ball loss/recovery, cost, and timing. Manual ID switches, fragmentation, ball misses, team mistakes, and reviewer notes remain `null` until a person evaluates the output.

## Zero-cost player CV tracking ablation

Test 4 reuses the saved Test 3 VLM response JSON and never constructs an OpenRouter provider. Its control is the existing improved tracker: Hungarian association, soft team history, persistent logical tracks, linear between-anchor player interpolation, and the current VLM-initialized ball tracker. The new variant changes only player-local sparse LK, global camera translation, scene-cut spatial resets, and track confidence.

```bash
python -m track_game.experiment4 render-saved
```

Both variants render locally. Detailed per-frame camera, cut, and confidence JSONL logs remain ignored; compact preflight and metrics JSON are committed. Fewer machine-created IDs must not be described as fewer real ID switches without manual video review.

## Prepared 1 FPS experiment

Test 5 prepares exactly 31 anchors at frames `0, 30, 60, …, 870, 899`, preserving the same Gemini 3.1 Flash Lite model, ruler, prompt, schema, player CV configuration, camera/cut/confidence behavior, and ball tracker. Preparation extracts local ruler images and writes a cost preflight but contains no paid-run command.

```bash
python -m track_game.experiment5 prepare
```

The 31-call batch must not be run until the current pricing, measured prior per-anchor cost, low/expected/high total, purpose, and changed variable are presented and the user explicitly approves exactly 31 calls.
