from PIL import Image, ImageDraw

from .ruler import normalized_to_pixel
from .schema import Team
from .tracking import TrackedFrame, TrackedPlayer

TEAM_COLORS = {Team.A: "#00b7ff", Team.B: "#ff3b5c", Team.UNKNOWN: "#aaaaaa"}


def draw_player_marker(
    draw: ImageDraw.ImageDraw,
    player: TrackedPlayer,
    size: tuple[int, int],
    show_id: bool,
) -> None:
    x, y = normalized_to_pixel(player.box.foot.x, player.box.foot.y, *size)
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
