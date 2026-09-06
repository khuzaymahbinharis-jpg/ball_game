# Test 5: prepared 1 FPS anchor experiment

Status: **prepared, not approved, zero paid calls attempted**.

- Model: `google/gemini-3.1-flash-lite`
- Exact frames: `0, 30, 60, …, 870, 899`
- Exact call count: **31**
- Current listed pricing captured 2026-09-06: **$0.25/M input, $1.50/M output**
- Measured prior average: **$0.00264020 per anchor** across the complete 61-call Test 3 batch
- Estimated total: **$0.030287 low / $0.081846 expected / $0.210521 high**
- Reduction from 2 FPS: **30 calls, or 49.18% expected call/cost savings**

Purpose: determine whether the stronger local player tracker can retain comparable visual quality while halving the VLM anchor rate. The only intended experimental change is the anchor interval from 15 to 30 frames. No paid runner is exposed by `experiment5`; an explicit approval for exactly 31 calls is required before one may be added or executed.
