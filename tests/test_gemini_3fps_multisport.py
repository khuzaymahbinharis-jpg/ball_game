import pytest

from track_game import gemini_3fps_multisport as module


def test_multisport_three_fps_schedule_and_budget_are_exact():
    assert module.ANCHOR_FRAMES[:3] == (0, 10, 20)
    assert module.ANCHOR_FRAMES[-2:] == (890, 899)
    assert len(module.ANCHOR_FRAMES) == 91
    assert len(module.CLIPS) == 5
    assert module.PAID_CALLS == 455
    assert module.MAX_CONCURRENCY == 2


def test_multisport_prompt_removes_basketball_only_wording():
    prompt = module.multisport_detection_prompt(20, "volleyball")

    assert "volleyball video frame" in prompt
    assert "basketball video frame" not in prompt
    assert "For the game ball" in prompt
    assert "Team A is the visually lighter/brighter uniform group" in prompt


def test_multisport_run_rejects_wrong_approval_before_repository_access():
    with pytest.raises(PermissionError, match="exactly 455"):
        module.run("missing", 454)
