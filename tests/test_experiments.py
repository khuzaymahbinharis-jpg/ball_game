from track_game.experiments import (
    ExperimentRecord,
    JsonlExperimentLogger,
    TrackingMetricsRecord,
    TrackingMetricsWriter,
)


def test_jsonl_logger_round_trip(tmp_path):
    logger = JsonlExperimentLogger(tmp_path / "nested" / "runs.jsonl")
    logger.append(
        ExperimentRecord(
            experiment_id="mock-1",
            status="completed",
            source_clip="synthetic.mp4",
            source_timestamp_seconds=1.0,
            source_frame_number=30,
            model="mock",
            pricing_snapshot={"input": 0},
            image_resolution=(160, 90),
            prompt_image_resolution=(224, 154),
            ruler_grounding=True,
            prompt_version="mock-v1",
            schema_version="mock-v1",
            reasoning_setting={"effort": "minimal"},
            vlm_calls=1,
        )
    )
    row = logger.read_all()[0]
    assert row["actual_openrouter_cost_usd"] is None
    assert row["source_frame_number"] == 30


def test_tracking_metrics_preserve_unknown_human_fields_as_null(tmp_path):
    path = tmp_path / "metrics.json"
    record = TrackingMetricsRecord(
        experiment_id="comparison",
        variant="improved",
        model="mock",
        anchor_count=3,
        planned_vlm_calls=3,
        attempted_vlm_calls=0,
        successful_vlm_calls=0,
        actual_api_cost_usd=None,
        api_cost_complete=False,
        summed_vlm_latency_seconds=None,
        api_batch_seconds=None,
        local_processing_seconds=0.25,
        total_processing_seconds=0.25,
        unique_ids_over_time={0: (1, 2), 15: (1, 2)},
        tracks_created=2,
        tracks_expired=0,
        tracks_lost=0,
        tracks_recovered=0,
        ball_lost_events=1,
        ball_recovered_events=1,
        ball_lost_frames=2,
    )
    TrackingMetricsWriter(path).write(record)
    saved = TrackingMetricsWriter(path).read()
    assert saved["human_evaluation"]["manual_id_switches"] is None
    assert saved["unique_ids_over_time"]["0"] == [1, 2]
