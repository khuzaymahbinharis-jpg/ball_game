import pytest

from track_game.experiment3 import (
    ANCHOR_FRAMES,
    COST_ESTIMATE,
    VLM_CALLS,
    run_approved_comparison,
)


def test_next_experiment_keeps_two_fps_sampling_and_final_frame():
    assert VLM_CALLS == 61
    assert ANCHOR_FRAMES[:3] == (0, 15, 30)
    assert ANCHOR_FRAMES[-2:] == (885, 899)


def test_cost_estimate_is_derived_from_prior_observed_anchor_costs():
    assert COST_ESTIMATE["low_usd"] == pytest.approx(0.059841)
    assert COST_ESTIMATE["expected_usd"] == pytest.approx(0.15647, abs=1e-6)
    assert COST_ESTIMATE["high_usd"] == pytest.approx(0.414251)


def test_paid_comparison_requires_exact_new_approval(tmp_path):
    with pytest.raises(PermissionError, match="exactly 61"):
        run_approved_comparison(tmp_path, approved_call_count=0)
