from track_game.comparison import baseline_variant, improved_variant


def test_comparison_holds_sampling_constant_and_switches_tracking_only():
    baseline = baseline_variant().pipeline
    improved = improved_variant().pipeline
    assert baseline.sampling == improved.sampling
    assert baseline.sampling.every_n_frames == 15
    assert baseline.tracking.association_method == "greedy"
    assert baseline.tracking.team_constraint is True
    assert baseline.ball_tracking.enabled is False
    assert improved.tracking.association_method == "hungarian"
    assert improved.ball_tracking.enabled is True
