import pytest

Image = pytest.importorskip("PIL.Image")
ImageChops = pytest.importorskip("PIL.ImageChops")
from track_game.config import PipelineConfig, SamplingConfig  # noqa: E402
from track_game.pipeline import MockPipeline  # noqa: E402


def test_mock_pipeline_runs_end_to_end_and_draws_without_mutation():
    frames = [Image.new("RGB", (160, 90), "#206020") for _ in range(5)]
    originals = [frame.copy() for frame in frames]
    pipeline = MockPipeline(PipelineConfig(sampling=SamplingConfig(every_n_frames=2)))
    annotated, timeline = pipeline.process_frames(frames, show_ids=True)
    assert len(annotated) == len(timeline) == 5
    assert len({frame.players[0].track_id for frame in timeline}) == 1
    assert ImageChops.difference(frames[0], originals[0]).getbbox() is None
    assert ImageChops.difference(annotated[0], originals[0]).getbbox() is not None
