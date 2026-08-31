from track_game.experiments import ExperimentRecord, JsonlExperimentLogger


def test_jsonl_logger_round_trip(tmp_path):
    logger = JsonlExperimentLogger(tmp_path / "nested" / "runs.jsonl")
    logger.append(
        ExperimentRecord(
            "mock-1", "mock", "synthetic", 15, 3, tracking_config={"distance": 0.2}
        )
    )
    row = logger.read_all()[0]
    assert row["api_cost_usd"] is None
    assert row["tracking_config"] == {"distance": 0.2}
