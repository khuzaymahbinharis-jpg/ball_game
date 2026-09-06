from PIL import Image, ImageDraw, ImageFont

from .ruler import normalized_to_pixel
from .schema import Team
from .tracking import TrackedFrame, TrackedPlayer

TEAM_COLORS = {Team.A: "#00b7ff", Team.B: "#ff3b5c", Team.UNCERTAIN: "#aaaaaa"}


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
    radius = max(8, round(min(size) * 0.018))
    color = TEAM_COLORS[player.team]
    draw.ellipse(
        (x - radius * 2, y - radius // 2, x + radius * 2, y + radius // 2),
        outline=color,
        width=max(2, radius // 3),
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
            (x + radius * 2 + 2, y - radius),
            f"P{player.track_id}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )


def annotate_frame(
    image: Image.Image, frame: TrackedFrame, show_ids: bool = False
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for player in frame.players:
        draw_player_marker(draw, player, output.size, show_ids)
    if frame.ball is not None:
        x, y = normalized_to_pixel(frame.ball.x, frame.ball.y, *output.size)
        radius = max(5, round(min(output.size) * 0.01))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#fff200",
            outline="black",
            width=2,
        )
        draw.arc(
            (x - radius * 2, y - radius * 2, x + radius * 2, y + radius * 2),
            0,
            360,
            fill="white",
            width=2,
        )
    return output


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
    scale = min(width / source_width, height / source_height)
    overlay = Image.new("RGBA", overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    full_radius = max(8, round(min(source_size) * 0.018))
    radius = max(2, round(full_radius * scale))
    line_width = max(1, round(max(2, full_radius // 3) * scale))
    font = _annotation_font(max(8, round(18 * scale)))
    for player in frame.players:
        x, y = normalized_to_pixel(player.foot.x, player.foot.y, width, height)
        color = TEAM_COLORS[player.team]
        draw.ellipse(
            (x - radius * 2, y - radius // 2, x + radius * 2, y + radius // 2),
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
        if show_ids:
            draw.text(
                (x + radius * 2 + 1, y - radius),
                f"P{player.track_id}",
                fill="white",
                font=font,
                stroke_width=1,
                stroke_fill="black",
            )
    if frame.ball is not None:
        x, y = normalized_to_pixel(frame.ball.x, frame.ball.y, width, height)
        full_ball_radius = max(5, round(min(source_size) * 0.01))
        ball_radius = max(2, round(full_ball_radius * scale))
        draw.ellipse(
            (x - ball_radius, y - ball_radius, x + ball_radius, y + ball_radius),
            fill="#fff200",
            outline="black",
            width=max(1, round(2 * scale)),
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
            width=max(1, round(2 * scale)),
        )
    return overlay
