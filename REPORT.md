# Track the Game — Report

> Working skeleton. Results-dependent sections are intentionally unfilled until controlled experiments are run on the official/local clips. No mock result is evidence of detection quality, cost, or speed.

## 1. Method

The planned pipeline samples anchor frames from the source video, optionally adds prompt-only normalized rulers, obtains strict structured semantic detections from a VLM, associates player identities across anchors, interpolates locations between anchors, and renders sports-game-style overlays on untouched source frames with Pillow.

The initial association baseline is greedy nearest-neighbour matching between bottom-centre player positions, gated by normalized distance and optionally constrained by VLM team label. Track IDs are owned by the tracker. Linear interpolation supplies positions between anchor observations. This baseline was selected for interpretability and will be compared with more sophisticated methods only if real failure evidence warrants them.

**To add after real testing:** exact prompt and model; clip selection; sampling setup; provider concurrency/retry policy; encoding details; and any method changes supported by observations.

## 2. Example successes and failures

No real inference has been run. Add representative frames and factual observations here after testing, including identity continuity, crossings, occlusions, camera motion/cuts, small or hidden balls, team ambiguity, and possession changes. Include failures as well as successes and avoid selecting only favourable clips.

## 3. Analysis / ablations / cost / latency

No benchmarks have been run. Planned controlled comparisons change one variable at a time:

- ruler grounding enabled versus disabled;
- model choice within the approved Gemini-family/OpenRouter setup;
- anchor sampling interval;
- matching threshold and team constraint;
- baseline association versus a justified motion/assignment enhancement.

For every run, record model, clip, sampling rate, call count, measured wall time and VLM latency, provider-reported token usage and cost, retries/errors, and complete tracking configuration. Report missing measurements as unavailable, not estimates. Summarize all five eventual clips and relate accuracy/identity failures to the under-$1 and target-time constraints only after actual measurements exist.

