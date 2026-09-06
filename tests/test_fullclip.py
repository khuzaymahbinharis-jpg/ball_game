import pytest

from track_game.fullclip import (
    ANCHOR_FRAMES,
    MAX_CONCURRENCY,
    VLM_CALLS,
    full_clip_detection_prompt,
    run_approved_full_clip_test,
)


def test_full_clip_sampling_is_eleven_calls_covering_both_ends():
    assert VLM_CALLS == 11
    assert MAX_CONCURRENCY == 11
    assert ANCHOR_FRAMES == (0, 90, 180, 270, 360, 450, 540, 630, 720, 810, 899)


def test_full_clip_prompt_locks_team_labels_across_anchors():
    prompt = full_clip_detection_prompt(90)
    assert "Team A: Oklahoma City" in prompt
    assert "Team B: San Antonio" in prompt
    assert "must be 90" in prompt


def test_full_clip_runner_requires_exact_approved_count(tmp_path):
    with pytest.raises(PermissionError, match="exactly 11"):
        run_approved_full_clip_test(tmp_path, approved_call_count=0)
    with pytest.raises(PermissionError, match="exactly 11"):
        run_approved_full_clip_test(tmp_path, approved_call_count=10)
