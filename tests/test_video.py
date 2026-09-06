from PIL import Image

from track_game.tracking import TrackedFrame
from track_game.video import (
    extract_video_frames,
    probe_video,
    render_tracked_video,
    write_video_frames,
)


def test_streaming_video_render_preserves_frame_count(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    frames = [Image.new("RGB", (64, 36), color) for color in ("red", "green", "blue")]
    write_video_frames(frames, source, fps=3.0)
    timeline = [TrackedFrame(i, (), None, None) for i in range(3)]
    info = render_tracked_video(source, output, timeline, show_ids=True)
    assert output.is_file()
    assert info.frame_count == probe_video(source).frame_count == 3


def test_batch_frame_extraction_preserves_requested_frame_names(tmp_path):
    source = tmp_path / "source.mp4"
    frames = [Image.new("RGB", (64, 36), (index * 30, 0, 0)) for index in range(5)]
    write_video_frames(frames, source, fps=5.0)
    paths = extract_video_frames(source, [0, 2, 4], tmp_path / "anchors")
    assert set(paths) == {0, 2, 4}
    assert all(path.is_file() for path in paths.values())
