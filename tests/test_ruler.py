import pytest

Image = pytest.importorskip("PIL.Image")
from track_game.ruler import (  # noqa: E402
    add_normalized_rulers,
    normalized_to_pixel,
    pixel_to_normalized,
)


def test_ruler_does_not_mutate_source_and_adds_margins():
    source = Image.new("RGB", (100, 50), "green")
    ruled = add_normalized_rulers(source, 40, 0.1)
    assert source.size == (100, 50)
    assert ruled.size == (140, 90)
    assert ruled.getpixel((40, 40)) == (0, 128, 0)


def test_coordinate_round_trip_uses_original_dimensions():
    pixel = normalized_to_pixel(0.5, 1, 101, 51)
    assert pixel == (50, 50)
    assert pixel_to_normalized(*pixel, 101, 51) == (0.5, 1)
