# Gemini Flash 5 FPS comparison

Status: **completed on 2026-09-08 UTC; all 302 approved inference calls attempted exactly once**.

This experiment compares `google/gemini-3.8-flash` and `google/gemini-3.7-flash` on the same silent 30-second source clip at 5 FPS. The fixed sample contains 151 frames per model (`0, 6, 12, …, 894, 899`), for exactly 302 paid calls. Automatic retries and provider fallbacks are disabled; both models are pinned to the compatible Google AI Studio endpoint.

The live catalog snapshot captured on 2026-09-08 UTC reported $0.75 per million input tokens and $3.75 per million output tokens for both endpoints. Estimates use each model's own completed 2 FPS run as the token-usage proxy:

| Model | Low | Expected | High |
|---|---:|---:|---:|
| Gemini 3.8 Flash | $0.2605 | $0.9198 | $2.4849 |
| Gemini 3.7 Flash | $0.2593 | $0.9484 | $2.4849 |
| **Combined** | **$0.5198** | **$1.8682** | **$4.9699** |

Low uses the prior per-request token minima, expected uses prior per-request means, and high uses the prior maximum input with the fixed 4,096-token output ceiling. These are planning estimates, not guaranteed bounds. Exact calculations are in `cost_preflight.json`.

The run must not start without explicit approval for exactly 302 calls:

```powershell
.venv\Scripts\python.exe -m track_game.gemini_5fps run --approved-call-count 302
```

## Completed results

| Model | Schema-valid | Players/frame | Ball anchors | Possession anchors | Actual cost | Mean latency | Batch time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | 148/151 (98.01%) | 8.176 | 109 | 100 | $0.933797 | 7.764s | 225.746s |
| Gemini 3.7 Flash | 143/151 (94.70%) | 8.098 | 100 | 91 | $0.965046 | 7.765s | 232.120s |
| **Combined** | **291/302** |  |  |  | **$1.898843** |  |  |

Both outputs contain exactly 900 frames at 30 FPS and 1920×1080, last 30.0 seconds, and contain no audio stream. Gemini 3.8 produced more schema-valid responses, player detections, ball anchors, and possession anchors while costing slightly less. These machine metrics do not replace manual visual review.
