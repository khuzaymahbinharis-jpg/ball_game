from PIL import Image, ImageDraw


def add_normalized_rulers(
    image: Image.Image, margin_px: int = 44, tick_step: float = 0.1
) -> Image.Image:
    """Return a prompt-only copy with rulers; never mutates the source image."""
    if margin_px < 20 or not 0 < tick_step <= 1:
        raise ValueError("invalid ruler configuration")
    source = image.convert("RGB")
    out = Image.new(
        "RGB", (source.width + margin_px, source.height + margin_px), "white"
    )
    out.paste(source, (margin_px, margin_px))
    draw = ImageDraw.Draw(out)
    steps = round(1 / tick_step)
    for i in range(steps + 1):
        value = min(1.0, i * tick_step)
        x = margin_px + round(value * (source.width - 1))
        y = margin_px + round(value * (source.height - 1))
        draw.line((x, margin_px - 8, x, margin_px), fill="black", width=1)
        draw.text((x - 8, 2), f"{value:.1f}", fill="black")
        draw.line((margin_px - 8, y, margin_px, y), fill="black", width=1)
        draw.text((1, y - 6), f"{value:.1f}", fill="black")
    return out


def normalized_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    if not 0 <= x <= 1 or not 0 <= y <= 1 or width < 1 or height < 1:
        raise ValueError("invalid normalized coordinate or dimensions")
    return round(x * (width - 1)), round(y * (height - 1))


def pixel_to_normalized(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    if width < 2 or height < 2 or not 0 <= x < width or not 0 <= y < height:
        raise ValueError("invalid pixel coordinate or dimensions")
    return x / (width - 1), y / (height - 1)
