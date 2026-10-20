"""把真实三标签逐层概率及场景族区间画成静态曲线，不平滑、不重归一化。"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def plot_patching_curves(summary, destination: Path, *, title, split, setting, eligible_only=False):
    selected = [
        r
        for r in summary["curves"]
        if r["split"] == split
        and r["setting"] == setting
        and r["baseline_correct_subset"] == eligible_only
    ]
    positions = sorted({r["position"] for r in selected})
    if not positions:
        raise ValueError("没有对应曲线，不生成空图")
    width = 600 * len(positions)
    canvas = Image.new("RGB", (width, 480), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    heading = ImageFont.load_default(size=22)
    draw.text((35, 16), title, fill="#17212f", font=heading)
    colors = {
        "clean": ("#c2413b", "#fbe3e0"),
        "address": ("#1769aa", "#dfedfa"),
        "attribute": ("#7751a9", "#eee5f7"),
    }
    for panel, position in enumerate(positions):
        rows = sorted([r for r in selected if r["position"] == position], key=lambda r: r["layer"])
        left, right, top, bottom = panel * 600 + 65, panel * 600 + 560, 100, 390
        last_layer = max(r["layer"] for r in rows)

        def xy(
            layer, value, *, left=left, right=right, top=top, bottom=bottom, last_layer=last_layer
        ):
            return left + (right - left) * layer / max(1, last_layer), bottom - (
                bottom - top
            ) * value

        draw.text((left, 65), f"{position} | n={rows[0]['n']} pairs", fill="#17212f", font=font)
        for n in range(6):
            value = n / 5
            y = xy(0, value)[1]
            draw.line((left, y, right, y), fill="#e5e9ef", width=1)
            draw.text((left - 37, y - 7), f"{value:.1f}", fill="#52606d", font=small)
        for layer in range(0, last_layer + 1, 5):
            x = xy(layer, 0)[0]
            draw.text((x - 7, bottom + 9), str(layer), fill="#52606d", font=small)
        for outcome, (color, shade) in colors.items():
            measures = [r["outcomes"][outcome]["probability"] for r in rows]
            if any(m["estimate"] is None for m in measures):
                continue
            if all(m["ci95"] is not None for m in measures):
                lower = [xy(r["layer"], m["ci95"][0]) for r, m in zip(rows, measures, strict=True)]
                upper = [xy(r["layer"], m["ci95"][1]) for r, m in zip(rows, measures, strict=True)]
                draw.polygon(lower + list(reversed(upper)), fill=shade)
            points = [xy(r["layer"], m["estimate"]) for r, m in zip(rows, measures, strict=True)]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        draw.text(
            (left + 125, bottom + 33), "Zero-based decoder block (post)", fill="#52606d", font=small
        )
    for index, (outcome, (color, _)) in enumerate(colors.items()):
        x = 35 + index * 290
        draw.line((x, 455, x + 24, 455), fill=color, width=4)
        draw.text(
            (x + 32, 446),
            {
                "clean": "Original recipient answer",
                "address": "Recipient at donor target",
                "attribute": "Donor answer",
            }[outcome],
            fill="#17212f",
            font=small,
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
