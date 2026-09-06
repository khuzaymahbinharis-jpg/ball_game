import pytest

from track_game.experiment5 import (
    ANCHOR_FRAMES,
    VLM_CALLS,
    run_approved_experiment5,
)


def test_one_fps_plan_has_exact_final_frame_and_31_calls():
    assert VLM_CALLS == 31
    assert ANCHOR_FRAMES[:3] == (0, 30, 60)
    assert ANCHOR_FRAMES[-2:] == (870, 899)
    assert len(set(ANCHOR_FRAMES)) == VLM_CALLS


def test_paid_runner_requires_exact_approval_before_touching_repository(tmp_path):
    with pytest.raises(PermissionError, match="exactly 31 calls"):
        run_approved_experiment5(tmp_path, 30)
