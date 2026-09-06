# Test 5: 1 FPS anchor experiment

Status: **completed with 27/31 schema-valid anchors**.

- Model: `google/gemini-3.1-flash-lite`
- Exact frames: `0, 30, 60, …, 870, 899`
- Exact call count: **31**
- Current listed pricing captured 2026-09-06: **$0.25/M input, $1.50/M output**
- Measured prior average: **$0.00264020 per anchor** across the complete 61-call Test 3 batch
- Estimated total: **$0.030287 low / $0.081846 expected / $0.210521 high**
- Reduction from 2 FPS: **30 calls, or 49.18% expected call/cost savings**

Purpose: determine whether the stronger local player tracker can retain comparable visual quality while halving the VLM anchor rate. The only intended experimental change is the anchor interval from 15 to 30 frames. The paid runner is exact-count gated, performs no automatic retries, and refuses duplicate execution after call records exist.

## Result

- Exactly **31** calls attempted; **0** retries.
- **27** responses passed schema validation; frames 570, 630, 780, and 810 failed validation and were retained in the detailed ignored logs.
- Actual API cost: **$0.08316075**, versus the **$0.081846** preflight expectation.
- Actual cost reduction versus the 61-call Test 3 run: **48.36%**.
- API batch: **37.25 s**; complete local player tracking, ball tracking, and rendering: **112.99 s**.
- Output: `outputs/spurs_thunder_test5_1fps_player_cv.mp4` (kept local/ignored).
- Machine metrics are recorded in `metrics.json` and `summary.json`; they are not manual ID-switch counts.
