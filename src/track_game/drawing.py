from PIL import Image, ImageColor, ImageDraw, ImageFont

from .ruler import normalized_to_pixel
from .schema import Team
from .tracking import TrackedFrame, TrackedPlayer

TEAM_COLORS = {Team.A: "#00b7ff", Team.B: "#ff3b5c", Team.UNCERTAIN: "#aaaaaa"}
PLAYER_RADIUS_FRACTION = 0.026
BALL_RADIUS_FRACTION = 0.014
MARKER_FILL_ALPHA = 58
OVERLAY_SUPERSAMPLE = 3
MARKER_HALF_WIDTH_RATIO = 2.2
MARKER_HALF_HEIGHT_RATIO = 0.58
MARKER_Y_OFFSET_RATIO = 0.12


def _player_marker_geometry(
    size: tuple[int, int], foot_y: float
) -> tuple[int, int, int, int]:
    """Return perspective-aware ellipse geometry in source-frame pixels."""

    # The reference style uses a wider ellipse for near-side players while
    # keeping distant markers compact.  Scaling from the normalized foot
    # position approximates that field/court perspective without covering the
    # player's body.
    perspective = 0.72 + 0.80 * max(0.0, min(1.0, foot_y))
    base_radius = max(12, round(min(size) * PLAYER_RADIUS_FRACTION))
    radius = max(10, round(base_radius * perspective))
    half_width = max(18, round(radius * MARKER_HALF_WIDTH_RATIO))
    half_height = max(6, round(radius * MARKER_HALF_HEIGHT_RATIO))
    y_offset = max(1, round(radius * MARKER_Y_OFFSET_RATIO))
    return radius, half_width, half_height, y_offset


def _annotation_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_player_marker(
    draw: ImageDraw.ImageDraw,
    player: TrackedPlayer,
    size: tuple[int, int],
    show_id: bool,
) -> None:
    x, y = normalized_to_pixel(player.foot.x, player.foot.y, *size)
    radius, half_width, half_height, y_offset = _player_marker_geometry(
        size, player.foot.y
    )
    color = TEAM_COLORS[player.team] if player.tracking_state == "confirmed" else "#b8b8b8"
    rgb = ImageColor.getrgb(color)
    fill = rgb + (MARKER_FILL_ALPHA,) if draw.mode == "RGBA" else rgb
    bounds = (
        x - half_width,
        y + y_offset - half_height,
        x + half_width,
        y + y_offset + half_height,
    )
    line_width = max(2, round(radius * 0.11))
    draw.ellipse(
        bounds,
        fill=fill,
        outline="black",
        width=line_width + 2,
    )
    draw.ellipse(
        bounds,
        outline=color,
        width=line_width,
    )
    if player.possesses_ball:
        draw.polygon(
            (
                (x, y - radius * 3),
                (x - radius, y - radius * 5),
                (x + radius, y - radius * 5),
            ),
            fill="#ffe600",
            outline="black",
        )
    if show_id:
        draw.text(
            (x + half_width + 2, y - radius),
            f"P{player.track_id}{'?' if player.tracking_state == 'uncertain' else ''}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )


def annotate_frame(
    image: Image.Image, frame: TrackedFrame, show_ids: bool = False
) -> Image.Image:
    output = image.convert("RGBA")
    overlay = annotation_overlay(frame, output.size, output.size, show_ids)
    return Image.alpha_composite(output, overlay).convert("RGB")


def annotation_overlay(
    frame: TrackedFrame,
    source_size: tuple[int, int],
    overlay_size: tuple[int, int] = (960, 540),
    show_ids: bool = True,
) -> Image.Image:
    """Draw a compact transparent Pillow layer for later video compositing."""

    source_width, source_height = source_size
    width, height = overlay_size
    if source_width < 1 or source_height < 1 or width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    render_width = width * OVERLAY_SUPERSAMPLE
    render_height = height * OVERLAY_SUPERSAMPLE
    scale = min(render_width / source_width, render_height / source_height)
    overlay = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _annotation_font(max(12, round(18 * scale)))
    for player in frame.players:
        x, y = normalized_to_pixel(
            player.foot.x, player.foot.y, render_width, render_height
        )
        color = TEAM_COLORS[player.team] if player.tracking_state == "confirmed" else "#b8b8b8"
        fill = ImageColor.getrgb(color) + (MARKER_FILL_ALPHA,)
        full_radius, full_half_width, full_half_height, full_y_offset = (
            _player_marker_geometry(source_size, player.foot.y)
        )
        radius = max(3, round(full_radius * scale))
        half_width = max(3, round(full_half_width * scale))
        half_height = max(2, round(full_half_height * scale))
        y_offset = max(1, round(full_y_offset * scale))
        line_width = max(2, round(max(2, full_radius * 0.11) * scale))
        bounds = (
            x - half_width,
            y + y_offset - half_height,
            x + half_width,
            y + y_offset + half_height,
        )
        draw.ellipse(
            bounds,
            fill=fill,
            outline=(0, 0, 0, 205),
            width=line_width + max(1, round(1.5 * scale)),
        )
        draw.ellipse(
            bounds,
            outline=ImageColor.getrgb(color) + (255,),
            width=line_width,
        )
        if player.possesses_ball:
            draw.polygon(
                (
                    (x, y - radius * 3),
                    (x - radius, y - radius * 5),
                    (x + radius, y - radius * 5),
                ),
                fill="#ffe600",
                outline="black",
            )
        if show_ids:
            draw.text(
                (x + half_width + 1, y - radius),
                f"P{player.track_id}{'?' if player.tracking_state == 'uncertain' else ''}",
                fill="white",
                font=font,
                stroke_width=1,
                stroke_fill="black",
            )
    if frame.ball is not None:
        x, y = normalized_to_pixel(
            frame.ball.x, frame.ball.y, render_width, render_height
        )
        full_ball_radius = max(8, round(min(source_size) * BALL_RADIUS_FRACTION))
        ball_radius = max(3, round(full_ball_radius * scale))
        draw.ellipse(
            (x - ball_radius, y - ball_radius, x + ball_radius, y + ball_radius),
            fill=(255, 242, 0, 235),
            outline=(0, 0, 0, 255),
            width=max(2, round(3 * scale)),
        )
        draw.arc(
            (
                x - ball_radius * 2,
                y - ball_radius * 2,
                x + ball_radius * 2,
                y + ball_radius * 2,
            ),
            0,
            360,
            fill="white",
            width=max(2, round(3 * scale)),
        )
    return overlay.resize(overlay_size, Image.Resampling.LANCZOS)
