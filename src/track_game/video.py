"""Local ffmpeg boundary; frame semantics remain outside this module."""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from .drawing import annotation_overlay
from .tracking import TrackedFrame


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    duration_seconds: float
    fps: float
    width: int
    height: int


def resolve_ffmpeg() -> str:
    installed = shutil.which("ffmpeg")
    return installed or imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame_count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
    reader = imageio_ffmpeg.read_frames(str(path))
    metadata = next(reader)
    reader.close()
    width, height = metadata["size"]
    fps = float(metadata["fps"])
    return VideoInfo(frame_count, float(duration), fps, width, height)


def trim_video_clip(
    source: str | Path,
    output: str | Path,
    start_seconds: float,
    duration_seconds: float,
    fps: float = 30.0,
) -> VideoInfo:
    """Accurately decode/re-encode one clip with a deterministic frame count."""

    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if start_seconds < 0 or duration_seconds <= 0 or fps <= 0:
        raise ValueError("invalid trim timing")
    expected_frames = round(duration_seconds * fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            resolve_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(source),
            "-t",
            f"{duration_seconds:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(expected_frames),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ],
        check=True,
    )
    info = probe_video(output)
    if info.frame_count != expected_frames or abs(info.duration_seconds - duration_seconds) > 0.02:
        raise RuntimeError("trimmed clip did not match requested duration and frame count")
    return info


def extract_video_frame(
    path: str | Path, frame_index: int, output: str | Path
) -> Image.Image:
    """Extract one exact zero-based decoded frame and preserve it as a PNG."""

    path, output = Path(path), Path(output)
    if not path.is_file():
        raise FileNotFoundError(path)
    if frame_index < 0:
        raise ValueError("frame_index cannot be negative")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            resolve_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-y",
            str(output),
        ],
        check=True,
    )
    if not output.is_file():
        raise RuntimeError(f"frame {frame_index} was not present in {path}")
    with Image.open(output) as image:
        return image.convert("RGB").copy()


def extract_video_frames(
    path: str | Path, frame_indices: list[int] | tuple[int, ...], output_dir: str | Path
) -> dict[int, Path]:
    """Extract several exact decoded frames in one ffmpeg pass."""

    path, output_dir = Path(path), Path(output_dir)
    indices = sorted(set(frame_indices))
    if not path.is_file():
        raise FileNotFoundError(path)
    if not indices or indices[0] < 0:
        raise ValueError("frame_indices must contain non-negative values")
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {index: output_dir / f"frame_{index:04d}_original.png" for index in indices}
    missing = [index for index in indices if not destinations[index].is_file()]
    if not missing:
        return destinations
    expression = "+".join(f"eq(n\\,{index})" for index in missing)
    with tempfile.TemporaryDirectory(dir=output_dir) as directory:
        template = Path(directory) / "%08d.png"
        subprocess.run(
            [
                resolve_ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vf",
                f"select={expression}",
                "-vsync",
                "0",
                "-y",
                str(template),
            ],
            check=True,
        )
        extracted = sorted(Path(directory).glob("*.png"))
        if len(extracted) != len(missing):
            raise RuntimeError("not every requested video frame was present")
        for frame_index, extracted_path in zip(missing, extracted):
            extracted_path.replace(destinations[frame_index])
    return destinations


def read_video_frames(path: str | Path) -> tuple[list[Image.Image], float]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    info = probe_video(path)
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                resolve_ffmpeg(),
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
    return frames, info.fps


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
                resolve_ffmpeg(),
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


def render_tracked_video(
    source: str | Path,
    output: str | Path,
    timeline: list[TrackedFrame],
    show_ids: bool = True,
) -> VideoInfo:
    """Draw Pillow overlays, composite with ffmpeg, and preserve source audio."""

    source, output = Path(source), Path(output)
    info = probe_video(source)
    by_frame = {frame.frame_id: frame for frame in timeline}
    expected_ids = set(range(info.frame_count))
    if set(by_frame) != expected_ids:
        raise ValueError("tracked timeline must contain every source frame exactly once")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        overlay_directory = Path(directory) / "overlays"
        overlay_directory.mkdir()
        for frame_id in range(info.frame_count):
            overlay = annotation_overlay(
                by_frame[frame_id],
                (info.width, info.height),
                overlay_size=(480, 270),
                show_ids=show_ids,
            )
            overlay.save(
                overlay_directory / f"{frame_id:08d}.png",
                format="PNG",
                compress_level=1,
            )
        subprocess.run(
            [
                resolve_ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-framerate",
                str(info.fps),
                "-i",
                str(overlay_directory / "%08d.png"),
                "-filter_complex",
                (
                    f"[1:v]scale={info.width}:{info.height}:flags=bilinear[overlay];"
                    "[0:v][overlay]overlay=0:0:format=auto[video]"
                ),
                "-map",
                "[video]",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-frames:v",
                str(info.frame_count),
                "-shortest",
                "-movflags",
                "+faststart",
                "-y",
                str(output),
            ],
            check=True,
        )
    rendered_info = probe_video(output)
    if rendered_info.frame_count != info.frame_count:
        raise RuntimeError("annotated video frame count differs from source")
    return rendered_info
