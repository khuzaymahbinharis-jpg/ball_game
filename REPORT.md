# Track the Game — Report

> Working report. Results-dependent sections remain intentionally unfilled until controlled experiments run on the official/local clips. No mock result is evidence of detection quality, cost, or speed.

## 1. Method

The planned pipeline samples anchor frames from the source video, optionally adds prompt-only normalized rulers, obtains strict structured semantic detections from a VLM, associates player identities across anchors, interpolates locations between anchors, and renders sports-game-style overlays on untouched source frames with Pillow.

The initial association baseline is greedy nearest-neighbour matching between explicit player ground points, gated by normalized distance and optionally constrained by VLM team label. Track IDs are owned by the tracker. Linear interpolation supplies positions between anchor observations. This baseline was selected for interpretability and will be compared with more sophisticated methods only if real failure evidence warrants them.

### Prepared Test 1 configuration

The local development source was trimmed from 3:11 through 3:41, re-encoded to 30 FPS, and verified as 900 frames over 30.000 seconds. Test 1 uses the midpoint at clip frame 450 (15.000 seconds), preserving a 1920×1080 original PNG and creating a separate 1984×1144 PNG with 64-pixel normalized rulers on the top and left.

The prepared model is `google/gemini-3.1-flash-lite` through OpenRouter with minimal reasoning, one request, no automatic retry, strict JSON Schema output, and a 4,096-token output ceiling. The response includes explicit player boxes and foot points, A/B/uncertain team labels, ball center and optional box, possession identity, and uncertainty notes. Coordinates are mapped directly from normalized ruler values to the untouched frame dimensions for the Pillow preview.

### Prepared full-clip Test 2 configuration

The first full-clip test uses 11 VLM anchors at three-second intervals (frames 0, 90, …, 810, plus frame 899). It keeps the Test 1 model, resolution, rulers, schema, and minimal reasoning. The prompt now fixes Team A as Oklahoma City blue and Team B as San Antonio black/white for cross-frame consistency. Eleven independent requests will run in one bounded parallel wave with no retry; tracking and all 900-frame rendering remain local.

An initial full-resolution Python pixel-stream render took 59.4 seconds. A Pillow-generated 480×270 transparent annotation layer, ffmpeg scaling/compositing, ultrafast H.264, and source-audio preservation reduced the verified local render to 16.18 seconds while retaining a 1920×1080, 900-frame, 30-second output. This is a local infrastructure measurement, not the real full-clip experiment result.

## 2. Example successes and failures

Test 1 completed one approved real request and produced a validated response plus Pillow preview. The model returned seven players, detected the ball, and returned a possession reference. Human localization/team/ball quality fields remain deliberately unscored pending review; the interface result alone is not an accuracy claim. Do not treat the deterministic mock as detection evidence.

Later video testing should record identity continuity, crossings, occlusions, camera motion/cuts, small or hidden balls, team ambiguity, and possession changes. Include failures and avoid selecting only favourable clips.

## 3. Analysis / ablations / cost / latency

No benchmark has been run. Planned controlled comparisons change one variable at a time:

- ruler grounding enabled versus disabled;
- model choice within the approved Gemini-family/OpenRouter setup;
- anchor sampling interval;
- matching threshold and team constraint;
- baseline association versus a justified motion/assignment enhancement.

For every run, record model, clip, source timestamp/frame, image size, prompt/schema version, reasoning setting, call count, measured wall time and VLM latency, provider-reported token usage and cost, retries/errors, schema validation, detection counts, and complete tracking configuration when applicable. Human review fields cover player localization, team assignment, ball localization, obvious failures, and notes. Missing measurements remain unavailable rather than estimated.

Test 1's pre-run pricing snapshot listed $0.25 per million input/image-input tokens and $1.50 per million output tokens for the selected model. Based on one 1984×1144 ruler image, the versioned prompt/schema, and uncertain provider image/reasoning tokenization, the approximate one-call range was $0.0014–$0.0080, with $0.0031 expected.

The single request was served by Google AI Studio. It used 1,403 input tokens and 1,235 output tokens, with zero reported reasoning tokens. VLM latency was 5.860 seconds and total harness wall time was 7.129 seconds. OpenRouter reported an actual cost of $0.00220325, within the preflight range and below the expected estimate. Schema validation succeeded with zero retries and no recorded error.

Test 2's 11-call estimate used the current unchanged $0.25/M input and $1.50/M output list rates plus Test 1's observed usage. The approximate batch cost was $0.0154 low, $0.0242 expected, and $0.0880 high.

The approved batch attempted exactly 11 calls in 11.13 seconds with no retries. Nine responses passed strict schema validation; frames 630 and 810 returned schema-invalid output. Successful responses account for 23,289 known input tokens, 11,509 known output tokens, and $0.02308575 known cost. This is a minimum rather than the complete bill because usage from the two invalid responses was not retained by the original adapter.

Because the first and last anchors succeeded, the system rendered a clearly labeled degraded result from the nine valid anchors without making further requests. The corrected interpolation preserves every detection at exact anchor frames and interpolates across the two missing anchors. Final local rendering took 10.61 seconds, putting measured API-batch-plus-render processing at 21.74 seconds. The output is 1920×1080, 30 FPS, 900 frames, and 30.000 seconds with source audio. It contains 21 unique baseline track IDs; ID-switch, lost-track, and human quality fields remain unscored.
