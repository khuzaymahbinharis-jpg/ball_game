from PIL import Image

from .config import PipelineConfig
from .drawing import annotate_frame
from .provider import MockVLMProvider, VLMProvider
from .ruler import add_normalized_rulers
from .sampling import sample_frame_indices
from .tracking import NearestNeighbourTracker, TrackedFrame, interpolate_sequence


class MockPipeline:
    def __init__(
        self, config: PipelineConfig | None = None, provider: VLMProvider | None = None
    ):
        self.config = config or PipelineConfig()
        self.provider = provider or MockVLMProvider()

    def process_frames(
        self, frames: list[Image.Image], show_ids: bool = False
    ) -> tuple[list[Image.Image], list[TrackedFrame]]:
        if not frames:
            return [], []
        indices = sample_frame_indices(
            len(frames),
            self.config.sampling.every_n_frames,
            self.config.sampling.include_last,
        )
        c = self.config.tracking
        tracker = NearestNeighbourTracker(
            c.max_normalized_distance, c.team_constraint, c.max_missed_anchors
        )
        anchors = []
        for frame_id in indices:
            source = frames[frame_id]
            prompt_image = (
                add_normalized_rulers(
                    source, self.config.ruler.margin_px, self.config.ruler.tick_step
                )
                if self.config.ruler.enabled
                else source.copy()
            )
            anchors.append(tracker.update(self.provider.detect(frame_id, prompt_image)))
        timeline = interpolate_sequence(anchors)
        by_id = {frame.frame_id: frame for frame in timeline}
        annotated = [
            annotate_frame(image, by_id[i], show_ids)
            for i, image in enumerate(frames)
            if i in by_id
        ]
        return annotated, timeline
