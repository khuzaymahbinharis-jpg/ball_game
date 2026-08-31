# Project guardrails

This repository implements Zeta Solutions Task 02, **Track the Game**.

- Semantic image interpretation (players, ball, team, and possession) belongs exclusively to a VLM through OpenRouter. Classical CV is permitted only after VLM localisation, for tracking/data association.
- Do not introduce YOLO, SAM, pretrained detectors, or semantic detection by classical CV.
- During the mock phase, never call OpenRouter, inspect secrets, or run paid services. Tests must use deterministic mock responses.
- Keep normalized coordinates relative to the untouched source frame. Rulers are a configurable prompt-only overlay.
- Use Pillow for rulers and final FIFA-style annotations.
- Keep experiments reproducible and vary one parameter at a time. Never invent costs, timings, successes, or benchmark results.
- Prefer simple, explainable tracking and strict structured data. Preserve player identity explicitly.

