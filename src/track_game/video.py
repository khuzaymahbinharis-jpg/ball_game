"""Small ffmpeg boundary; frame semantics remain outside this module."""

import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def read_video_frames(path: str | Path) -> tuple[list[Image.Image], float]:
    """Decode via installed ffmpeg/ffprobe. Intended for tomorrow's local clips."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    numerator, denominator = map(int, probe.stdout.strip().split("/"))
    fps = numerator / denominator
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                str(Path(directory) / "%08d.png"),
            ],
            check=True,
        )
        frames = [
            Image.open(frame).convert("RGB").copy()
            for frame in sorted(Path(directory).glob("*.png"))
        ]
    return frames, fps


def write_video_frames(
    frames: list[Image.Image], output: str | Path, fps: float
) -> None:
    """Encode annotated frames as H.264 MP4; audio muxing can be added later."""
    if not frames or fps <= 0:
        raise ValueError("frames must be non-empty and fps positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        for index, frame in enumerate(frames, 1):
            frame.convert("RGB").save(Path(directory) / f"{index:08d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(directory) / "%08d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )
