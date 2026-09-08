# Controlled 2 FPS VLM model bake-off

Status: **completed on 2026-09-07; all 305 approved new inference calls attempted**.

All six requested slugs were exact matches in OpenRouter's live catalog. Each accepts image input, returns text, advertises structured outputs/`response_format`, and had at least one active compatible provider. The benchmark pins the provider shown below, disables provider fallbacks, requires parameter support, uses concurrency 8, temperature 0, a 4,096-token ceiling, and never retries. Qwen3-VL-235B used Alibaba after DeepInfra FP8 became inactive at run time; that provider change and the revised $2.285 ceiling were explicitly approved before inference.

| Model | Pinned provider | Input / M | Output / M | Separate image / M | Expected / frame | Low / 61 | Expected / 61 | High / 61 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `google/gemini-3.1-flash-lite` | Google AI Studio | $0.25 | $1.50 | $0.25 | $0.00264 | $0.0596 | $0.1611 | $0.4143 |
| `google/gemini-3.8-flash` | Google AI Studio | $0.75 | $3.75 | $0.75 | $0.00692 | $0.1686 | $0.4224 | $1.0554 |
| `qwen/qwen3-vl-30b-a3b-instruct` | Alibaba | $0.13 | $0.52 | — | $0.00103 | $0.0275 | $0.0627 | $0.1504 |
| `qwen/qwen3-vl-235b-a22b-instruct` | Alibaba | $0.26 | $1.04 | — | $0.00205 | $0.0550 | $0.1253 | $0.3009 |
| `bytedance-seed/seed-2-1-turbo` | Seed FP8 | $0.50 | $2.50 | — | $0.00462 | $0.1124 | $0.2816 | $0.7036 |
| `z-ai/glm-5.3-flash` | DeepInfra FP4 | $0.075 | $0.25 | — | $0.000526 | $0.0152 | $0.0321 | $0.0743 |

The existing Gemini 3.1 control is reusable because its clip, exact 61 frames, source/ruler images, prompt bytes, schema bytes, token ceiling, temperature, reasoning, concurrency, and no-retry policy match. It contains 61 attempts: 49 schema-valid and 12 schema-invalid. Reusing those failures is necessary for an honest reliability comparison. The legacy run retained parsed content and invalid raw content, but not complete OpenRouter response envelopes or returned provider names.

Five new models therefore require **305 paid calls**. Planning estimates for those new calls are approximately **$0.38 low / $0.92 expected / $2.28 high**. The six-model equivalent total, including the already-paid control, is approximately **$0.44 / $1.09 / $2.70**.

These are planning proxies, not promises: image tokenization and output length vary across model families. The expected case applies each selected endpoint's live price to the control's observed 157,860 input and 81,058 output tokens. See `cost_preflight.json` for exact calculations and scenario definitions.

The completed run recorded **$0.75623447** of known new-call cost. Cost metadata was present for 254/305 calls; the one Qwen3-VL-30B HTTP 429 and all 50 failed GLM calls returned no cost field, so the artifact-derived billing total is intentionally marked incomplete. See `results_table.md` and each model's `summary.json` for validity, latency, provider, and cost details. Manual visual-quality columns remain unscored, and no overall winner is declared automatically.

An explicitly approved 61-call Gemini 3.7 Flash extension was run on the same anchors and configuration. All 61 requests failed without an HTTP response body, request ID, provider metadata, or billable-usage metadata; therefore no schema-valid detections or output video were produced. The public catalog still reported the pinned Google AI Studio endpoint active after the run, so this is recorded as an operational endpoint/transport failure rather than evidence about Gemini 3.7's detection quality. The artifact-derived known cost is $0, but billing completeness cannot be proven without returned usage metadata.

A second explicitly approved attempt used the same model, Google AI Studio endpoint, frames, prompt, schema, and concurrency with only the request timeout increased from 45 to 120 seconds. It returned 60/61 schema-valid detections, cost $0.383148, averaged 10.781 seconds per request, and completed the concurrent API batch in 87.542 seconds. This confirms the first attempt was operationally invalid. The source clip and rendered retry output are both video-only files with no audio stream.

After explicit approval for the displayed 305-call ceiling, run:

```powershell
.venv\Scripts\python.exe -m track_game.model_bakeoff run --approved-call-count 305
```

The runner is restartable at per-frame records, never retries a completed or failed frame, renders all schema-valid detector outputs through the unchanged Player-CV-improved pipeline, and creates fixed-frame manual comparison sheets. It does not declare an overall winner before manual detector review.
