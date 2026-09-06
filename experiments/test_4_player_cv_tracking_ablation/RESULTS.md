# Test 4: player CV tracking ablation

This experiment made zero new OpenRouter requests. Both variants reused the 49 schema-valid detections from the same 61-attempt Test 3 batch; model, clip, prompt, ruler, schema, Hungarian association, soft team history, logical-track persistence, and ball-tracker implementation were held constant.

## Variants

- **Current improved:** linear player interpolation between VLM anchors.
- **Player CV improved:** VLM-initialized batched sparse LK, global camera translation, scene-cut spatial resets, and persistent confidence.

## Machine measurements

| Measurement | Current improved | Player CV improved |
|---|---:|---:|
| Local processing, including output render | 31.10 s | 51.03 s |
| Average local time/frame | 34.55 ms | 56.70 ms |
| Complete player-CV video pass | — | 18.46 s |
| Camera estimation inside CV pass | — | 4.06 s |
| Player sparse-LK updates inside CV pass | — | 5.65 s |
| Scene-cut scoring inside CV pass | — | 0.23 s |
| Ball tracking | 14.90 s | 14.35 s |
| H.264/Pillow rendering | 16.15 s | 18.22 s |
| Successful camera transforms | — | 859/899 |
| Detected cut frames | — | 15, 369, 442 |
| Uncertain track-frames | — | 392 |
| Recoveries from uncertainty | — | 31 |
| Logical tracks created | 27 | 29 |

Track counts are lifecycle diagnostics, not manually verified identity-switch counts. The cut-aware version deliberately starts cautious new spatial references after broadcast cuts, so it can create more logical IDs while avoiding confident nonsense across unrelated shots.

## Visual spot-check

Synchronized checks at frames 100, 200, 300, 368–370, 441–443, 600, 800, and 890 show materially less straight-line screen-space drift. The clearest case is frame 368: the current interpolation leaves a player marker gliding near the lower edge, while sparse LK keeps markers with the moving players. Frames 369 and 442 are correctly cleared at the broadcast cuts until the next saved VLM anchor.

This is sufficient visual improvement to justify the separately gated 1 FPS experiment. It is not yet sufficient to claim fewer identity switches without manual full-video scoring. The 51-second local path also exceeds the 25-second target; most wall time is full-video passes and rendering rather than camera estimation or cut scoring.
