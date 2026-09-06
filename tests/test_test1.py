import pytest
from PIL import Image, ImageChops

from track_game.provider import MockVLMProvider
from track_game.test1 import render_detection_preview, run_approved_test1


def test_paid_test_requires_exactly_one_explicitly_approved_call(tmp_path):
    with pytest.raises(PermissionError, match="exactly one"):
        run_approved_test1(tmp_path, approved_call_count=0)
    with pytest.raises(PermissionError, match="exactly one"):
        run_approved_test1(tmp_path, approved_call_count=2)


def test_preview_uses_original_frame_and_draws_mock_detection():
    original = Image.new("RGB", (320, 180), "#206020")
    unchanged = original.copy()
    detection = MockVLMProvider().detect(0, original)
    preview = render_detection_preview(original, detection)
    assert ImageChops.difference(original, unchanged).getbbox() is None
    assert ImageChops.difference(preview, original).getbbox() is not None
