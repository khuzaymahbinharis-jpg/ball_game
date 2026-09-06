from track_game.comparison import baseline_variant, cv_improved_variant, improved_variant


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


def test_new_features_are_independently_toggled_for_ablation():
    current = improved_variant().pipeline
    upgraded = cv_improved_variant().pipeline
    assert current.tracking.association_method == upgraded.tracking.association_method
    assert current.ball_tracking == upgraded.ball_tracking
    assert not current.player_cv_tracking.enabled
    assert not current.camera_motion.enabled
    assert not current.scene_cut.enabled
    assert not current.tracking.confidence.enabled
    assert upgraded.player_cv_tracking.enabled
    assert upgraded.camera_motion.enabled
    assert upgraded.scene_cut.enabled
    assert upgraded.tracking.confidence.enabled
