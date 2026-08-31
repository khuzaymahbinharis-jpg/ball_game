def sample_frame_indices(
    total_frames: int, every_n_frames: int, include_last: bool = True
) -> list[int]:
    if total_frames < 0 or every_n_frames < 1:
        raise ValueError("invalid frame count or interval")
    indices = list(range(0, total_frames, every_n_frames))
    if include_last and total_frames and indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices
