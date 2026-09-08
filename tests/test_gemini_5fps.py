from copy import deepcopy

from track_game.gemini_5fps import (
    ANCHOR_FRAMES,
    ANCHORS_PER_MODEL,
    PAID_CALLS,
    _catalog_fingerprint,
)


def test_five_fps_sampling_is_exact_and_includes_final_frame():
    assert ANCHORS_PER_MODEL == 151
    assert ANCHOR_FRAMES[:3] == (0, 6, 12)
    assert ANCHOR_FRAMES[-2:] == (894, 899)
    assert PAID_CALLS == 302


def test_catalog_fingerprint_ignores_capture_time_but_not_price():
    snapshot = {
        "captured_at": "first",
        "models": [
            {
                "key": "model",
                "requested_slug": "vendor/model",
                "chosen_provider_slug": "provider",
                "pricing_per_token": {"input": "0.1", "output": "0.2"},
                "benchmark_reasoning_setting": None,
            }
        ],
    }
    updated = deepcopy(snapshot)
    updated["captured_at"] = "second"
    assert _catalog_fingerprint(snapshot) == _catalog_fingerprint(updated)
    updated["models"][0]["pricing_per_token"] = {"input": "0.2", "output": "0.2"}
    assert _catalog_fingerprint(snapshot) != _catalog_fingerprint(updated)
