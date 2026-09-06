from PIL import Image, ImageDraw, ImageFont


def _ruler_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def add_normalized_rulers(
    image: Image.Image, margin_px: int = 64, tick_step: float = 0.1
) -> Image.Image:
    """Return a prompt-only copy with rulers; never mutates the source image."""
    if margin_px < 40 or not 0 < tick_step <= 1:
        raise ValueError("invalid ruler configuration")
    source = image.convert("RGB")
    out = Image.new(
        "RGB", (source.width + margin_px, source.height + margin_px), "white"
    )
    out.paste(source, (margin_px, margin_px))
    draw = ImageDraw.Draw(out)
    font = _ruler_font(max(14, margin_px // 3))
    draw.line(
        (margin_px, margin_px - 2, out.width - 1, margin_px - 2),
        fill="black",
        width=2,
    )
    draw.line(
        (margin_px - 2, margin_px, margin_px - 2, out.height - 1),
        fill="black",
        width=2,
    )
    steps = round(1 / tick_step)
    for i in range(steps + 1):
        value = i / steps
        x = margin_px + round(value * (source.width - 1))
        y = margin_px + round(value * (source.height - 1))
        label = f"{value:.1f}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        draw.line((x, margin_px - 10, x, margin_px - 2), fill="black", width=2)
        draw.text(
            (max(1, min(out.width - label_width - 1, x - label_width / 2)), 4),
            label,
            fill="black",
            font=font,
        )
        draw.line((margin_px - 10, y, margin_px - 2, y), fill="black", width=2)
        draw.text(
            (2, max(margin_px, min(out.height - label_height - 1, y - label_height / 2))),
            label,
            fill="black",
            font=font,
        )
    return out


def normalized_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    if not 0 <= x <= 1 or not 0 <= y <= 1 or width < 1 or height < 1:
        raise ValueError("invalid normalized coordinate or dimensions")
    return round(x * (width - 1)), round(y * (height - 1))


def pixel_to_normalized(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    if width < 2 or height < 2 or not 0 <= x < width or not 0 <= y < height:
        raise ValueError("invalid pixel coordinate or dimensions")
    return x / (width - 1), y / (height - 1)
