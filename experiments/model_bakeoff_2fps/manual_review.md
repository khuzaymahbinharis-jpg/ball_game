# Manual detector review

Subjective fields are intentionally blank. Review the fixed-frame overlays; do not use tracker output for these ratings.

| Model | Frame | Scene | Missed players | False players | Poor player localization | Foot-point quality | Wrong team | Ball correct/missed/wrong | Possession correct/wrong/uncertain | Notes |
|---|---:|---|---:|---:|---:|---|---:|---|---|---|
| google/gemini-3.1-flash-lite | 30 | wide court |  |  |  |  |  |  |  |  |
| google/gemini-3.1-flash-lite | 180 | crowded players |  |  |  |  |  |  |  |  |
| google/gemini-3.1-flash-lite | 270 | crossing/occlusion |  |  |  |  |  |  |  |  |
| google/gemini-3.1-flash-lite | 360 | small/distant players |  |  |  |  |  |  |  |  |
| google/gemini-3.1-flash-lite | 450 | post-cut wide view |  |  |  |  |  |  |  |  |
| google/gemini-3.1-flash-lite | 570 | ball/possession moment |  |  |  |  |  |  |  |  |
| google/gemini-3.1-flash-lite | 690 | difficult tiny ball |  |  |  |  |  |  |  |  |
| google/gemini-3.1-flash-lite | 840 | motion/crowding |  |  |  |  |  |  |  |  |
| google/gemini-3.8-flash | 30 | wide court |  |  |  |  |  |  |  |  |
| google/gemini-3.8-flash | 180 | crowded players |  |  |  |  |  |  |  |  |
| google/gemini-3.8-flash | 270 | crossing/occlusion |  |  |  |  |  |  |  |  |
| google/gemini-3.8-flash | 360 | small/distant players |  |  |  |  |  |  |  |  |
| google/gemini-3.8-flash | 450 | post-cut wide view |  |  |  |  |  |  |  |  |
| google/gemini-3.8-flash | 570 | ball/possession moment |  |  |  |  |  |  |  |  |
| google/gemini-3.8-flash | 690 | difficult tiny ball |  |  |  |  |  |  |  |  |
| google/gemini-3.8-flash | 840 | motion/crowding |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-30b-a3b-instruct | 30 | wide court |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-30b-a3b-instruct | 180 | crowded players |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-30b-a3b-instruct | 270 | crossing/occlusion |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-30b-a3b-instruct | 360 | small/distant players |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-30b-a3b-instruct | 450 | post-cut wide view |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-30b-a3b-instruct | 570 | ball/possession moment |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-30b-a3b-instruct | 690 | difficult tiny ball |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-30b-a3b-instruct | 840 | motion/crowding |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-235b-a22b-instruct | 30 | wide court |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-235b-a22b-instruct | 180 | crowded players |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-235b-a22b-instruct | 270 | crossing/occlusion |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-235b-a22b-instruct | 360 | small/distant players |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-235b-a22b-instruct | 450 | post-cut wide view |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-235b-a22b-instruct | 570 | ball/possession moment |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-235b-a22b-instruct | 690 | difficult tiny ball |  |  |  |  |  |  |  |  |
| qwen/qwen3-vl-235b-a22b-instruct | 840 | motion/crowding |  |  |  |  |  |  |  |  |
| bytedance-seed/seed-2-1-turbo | 30 | wide court |  |  |  |  |  |  |  |  |
| bytedance-seed/seed-2-1-turbo | 180 | crowded players |  |  |  |  |  |  |  |  |
| bytedance-seed/seed-2-1-turbo | 270 | crossing/occlusion |  |  |  |  |  |  |  |  |
| bytedance-seed/seed-2-1-turbo | 360 | small/distant players |  |  |  |  |  |  |  |  |
| bytedance-seed/seed-2-1-turbo | 450 | post-cut wide view |  |  |  |  |  |  |  |  |
| bytedance-seed/seed-2-1-turbo | 570 | ball/possession moment |  |  |  |  |  |  |  |  |
| bytedance-seed/seed-2-1-turbo | 690 | difficult tiny ball |  |  |  |  |  |  |  |  |
| bytedance-seed/seed-2-1-turbo | 840 | motion/crowding |  |  |  |  |  |  |  |  |
| z-ai/glm-5.3-flash | 30 | wide court |  |  |  |  |  |  |  |  |
| z-ai/glm-5.3-flash | 180 | crowded players |  |  |  |  |  |  |  |  |
| z-ai/glm-5.3-flash | 270 | crossing/occlusion |  |  |  |  |  |  |  |  |
| z-ai/glm-5.3-flash | 360 | small/distant players |  |  |  |  |  |  |  |  |
| z-ai/glm-5.3-flash | 450 | post-cut wide view |  |  |  |  |  |  |  |  |
| z-ai/glm-5.3-flash | 570 | ball/possession moment |  |  |  |  |  |  |  |  |
| z-ai/glm-5.3-flash | 690 | difficult tiny ball |  |  |  |  |  |  |  |  |
| z-ai/glm-5.3-flash | 840 | motion/crowding |  |  |  |  |  |  |  |  |
