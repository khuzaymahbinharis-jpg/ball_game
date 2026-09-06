# Controlled 2 FPS VLM model bake-off

Status: **prepared, not approved; zero new inference calls attempted**.

All six requested slugs were exact matches in OpenRouter's live catalog on 2026-09-06. Each accepts image input, returns text, advertises structured outputs/`response_format`, and had at least one active compatible provider. The benchmark pins the provider shown below, disables provider fallbacks, requires parameter support, uses concurrency 8, temperature 0, a 4,096-token ceiling, and never retries.

| Model | Pinned provider | Input / M | Output / M | Separate image / M | Expected / frame | Low / 61 | Expected / 61 | High / 61 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `google/gemini-3.1-flash-lite` | Google AI Studio | $0.25 | $1.50 | $0.25 | $0.00264 | $0.0596 | $0.1611 | $0.4143 |
| `google/gemini-3.8-flash` | Google AI Studio | $0.75 | $3.75 | $0.75 | $0.00692 | $0.1686 | $0.4224 | $1.0554 |
| `qwen/qwen3-vl-30b-a3b-instruct` | Alibaba | $0.13 | $0.52 | — | $0.00103 | $0.0275 | $0.0627 | $0.1504 |
| `qwen/qwen3-vl-235b-a22b-instruct` | DeepInfra FP8 | $0.20 | $0.88 | — | $0.00169 | $0.0434 | $0.1029 | $0.2514 |
| `bytedance-seed/seed-2-1-turbo` | Seed FP8 | $0.50 | $2.50 | — | $0.00462 | $0.1124 | $0.2816 | $0.7036 |
| `z-ai/glm-5.3-flash` | DeepInfra FP4 | $0.075 | $0.25 | — | $0.000526 | $0.0152 | $0.0321 | $0.0743 |

The existing Gemini 3.1 control is reusable because its clip, exact 61 frames, source/ruler images, prompt bytes, schema bytes, token ceiling, temperature, reasoning, concurrency, and no-retry policy match. It contains 61 attempts: 49 schema-valid and 12 schema-invalid. Reusing those failures is necessary for an honest reliability comparison. The legacy run retained parsed content and invalid raw content, but not complete OpenRouter response envelopes or returned provider names.

Five new models therefore require **305 paid calls**. Planning estimates for those new calls are approximately **$0.37 low / $0.90 expected / $2.24 high**. The six-model equivalent total, including the already-paid control, is approximately **$0.43 / $1.06 / $2.65**.

These are planning proxies, not promises: image tokenization and output length vary across model families. The expected case applies each selected endpoint's live price to the control's observed 157,860 input and 81,058 output tokens. See `cost_preflight.json` for exact calculations and scenario definitions.

After explicit approval for the displayed 305-call ceiling, run:

```powershell
.venv\Scripts\python.exe -m track_game.model_bakeoff run --approved-call-count 305
```

The runner is restartable at per-frame records, never retries a completed or failed frame, renders all schema-valid detector outputs through the unchanged Player-CV-improved pipeline, and creates fixed-frame manual comparison sheets. It does not declare an overall winner before manual detector review.
