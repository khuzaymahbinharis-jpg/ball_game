from track_game.gemini_4fps_fresh import (
    ANCHOR_FRAMES,
    EXPECTED_FAILED_BY_MODEL,
    PAID_CALLS,
    RETRY_FAILED_CALLS,
    RETRY_MAX_CONCURRENCY,
)


def test_four_fps_sampling_alternates_integer_frame_gaps_and_includes_last():
    gaps = [right - left for left, right in zip(ANCHOR_FRAMES, ANCHOR_FRAMES[1:])]

    assert len(ANCHOR_FRAMES) == 121
    assert len(set(ANCHOR_FRAMES)) == 121
    assert ANCHOR_FRAMES[:4] == (0, 8, 15, 23)
    assert ANCHOR_FRAMES[-2:] == (893, 899)
    assert set(gaps).issubset({6, 7, 8})
    assert PAID_CALLS == 242


def test_failed_frame_retry_budget_is_exact():
    assert EXPECTED_FAILED_BY_MODEL == {
        "gemini_3_8_flash": 121,
        "gemini_3_7_flash": 84,
    }
    assert sum(EXPECTED_FAILED_BY_MODEL.values()) == RETRY_FAILED_CALLS
    assert RETRY_FAILED_CALLS == 205
    assert RETRY_MAX_CONCURRENCY == 2
