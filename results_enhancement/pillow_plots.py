from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def pillow_available() -> bool:
    return Image is not None and ImageDraw is not None and ImageFont is not None


def _font(size: int, bold: bool = False) -> Any:
    if ImageFont is None:
        return None
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure(draw: Any, text: str, font: Any) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    return right - left, bottom - top


def _fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.2f}"


def _nice_limits(values: list[float], floor_zero: bool = True) -> tuple[float, float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return (0.0, 1.0)
    lo = min(finite)
    hi = max(finite)
    if floor_zero:
        lo = min(0.0, lo)
    if hi == lo:
        hi = lo + 1.0
    padding = (hi - lo) * 0.08
    if padding <= 0:
        padding = 0.1
    return lo - (0.0 if floor_zero else padding), hi + padding


def _draw_title(draw: Any, canvas: Any, title: str, subtitle: str | None = None) -> int:
    title_font = _font(30, bold=True)
    subtitle_font = _font(16)
    y = 26
    draw.text((34, y), title, fill="#111111", font=title_font)
    y += 42
    if subtitle:
        draw.text((34, y), subtitle, fill="#555555", font=subtitle_font)
        y += 30
    return y


def _draw_horizontal_bar_panel(
    draw: Any,
    box: tuple[int, int, int, int],
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    colors: list[str] | None = None,
    annotate_fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    panel_bg = "#fafafa"
    draw.rounded_rectangle(box, radius=14, fill=panel_bg, outline="#d8d8d8", width=1)

    title_font = _font(20, bold=True)
    text_font = _font(14)
    tick_font = _font(12)

    draw.text((x0 + 18, y0 + 14), title, fill="#111111", font=title_font)
    left_margin = x0 + 210
    right_margin = x1 - 34
    top = y0 + 58
    bottom = y1 - 46
    axis_bottom = bottom
    n = max(1, len(labels))
    bar_step = (bottom - top) / n
    bar_h = min(28, max(18, int(bar_step * 0.55)))
    axis_x0 = left_margin
    axis_x1 = right_margin
    vmin, vmax = _nice_limits(values, floor_zero=True)
    ticks = np.linspace(vmin, vmax, 5)

    draw.line((axis_x0, top - 8, axis_x0, axis_bottom), fill="#333333", width=2)
    draw.line((axis_x0, axis_bottom, axis_x1, axis_bottom), fill="#333333", width=2)
    for tick in ticks:
        frac = 0.0 if vmax == vmin else (float(tick) - vmin) / (vmax - vmin)
        tx = axis_x0 + frac * (axis_x1 - axis_x0)
        draw.line((tx, axis_bottom, tx, top - 8), fill="#e2e2e2", width=1)
        label = _fmt_tick(float(tick))
        tw, th = _measure(draw, label, tick_font)
        draw.text((tx - tw / 2, axis_bottom + 8), label, fill="#444444", font=tick_font)

    for idx, (label, value) in enumerate(zip(labels, values)):
        cy = top + idx * bar_step + bar_step / 2
        ty = cy - bar_h / 2
        frac = 0.0 if vmax == vmin else (float(value) - vmin) / (vmax - vmin)
        bx = axis_x0 + frac * (axis_x1 - axis_x0)
        color = colors[idx] if colors and idx < len(colors) else PALETTE[idx % len(PALETTE)]
        draw.rounded_rectangle((axis_x0, ty, bx, ty + bar_h), radius=7, fill=color)
        label_text = label if len(label) <= 26 else f"{label[:23]}..."
        draw.text((x0 + 14, cy - 8), label_text, fill="#222222", font=text_font)
        value_text = annotate_fmt.format(float(value)) if np.isfinite(value) else "NA"
        vw, _ = _measure(draw, value_text, text_font)
        draw.text((min(bx + 8, axis_x1 - vw), cy - 8), value_text, fill="#111111", font=text_font)

    xw, _ = _measure(draw, xlabel, text_font)
    draw.text((axis_x0 + (axis_x1 - axis_x0 - xw) / 2, y1 - 28), xlabel, fill="#444444", font=text_font)


def save_horizontal_bar_chart(
    path: str | Path,
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    colors: list[str] | None = None,
    subtitle: str | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not pillow_available():
        raise RuntimeError("Pillow is unavailable.")
    height = max(460, 170 + len(labels) * 58)
    canvas = Image.new("RGB", (1400, height), "white")
    draw = ImageDraw.Draw(canvas)
    top = _draw_title(draw, canvas, title, subtitle)
    _draw_horizontal_bar_panel(draw, (28, top, 1372, height - 24), labels, values, "", xlabel, colors=colors)
    canvas.save(target)
    return target


def save_dual_horizontal_bar_chart(
    path: str | Path,
    labels: list[str],
    left_values: list[float],
    right_values: list[float],
    title: str,
    left_title: str,
    right_title: str,
    left_xlabel: str,
    right_xlabel: str,
    colors: list[str] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not pillow_available():
        raise RuntimeError("Pillow is unavailable.")
    height = max(500, 170 + len(labels) * 58)
    canvas = Image.new("RGB", (1700, height), "white")
    draw = ImageDraw.Draw(canvas)
    top = _draw_title(draw, canvas, title)
    _draw_horizontal_bar_panel(draw, (28, top, 840, height - 24), labels, left_values, left_title, left_xlabel, colors=colors)
    _draw_horizontal_bar_panel(draw, (860, top, 1672, height - 24), labels, right_values, right_title, right_xlabel, colors=colors)
    canvas.save(target)
    return target


def save_distribution_boxplot(
    path: str | Path,
    labels: list[str],
    data: list[list[float] | np.ndarray],
    title: str,
    xlabel: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not pillow_available():
        raise RuntimeError("Pillow is unavailable.")
    height = max(420, 180 + len(labels) * 60)
    canvas = Image.new("RGB", (1400, height), "white")
    draw = ImageDraw.Draw(canvas)
    top = _draw_title(draw, canvas, title)
    box = (40, top, 1360, height - 24)
    draw.rounded_rectangle(box, radius=14, fill="#fafafa", outline="#d8d8d8", width=1)
    title_font = _font(20, bold=True)
    text_font = _font(14)
    tick_font = _font(12)
    left = box[0] + 210
    right = box[2] - 34
    inner_top = box[1] + 44
    bottom = box[3] - 44
    all_values = [float(v) for group in data for v in group if np.isfinite(v)]
    vmin, vmax = _nice_limits(all_values, floor_zero=False)
    ticks = np.linspace(vmin, vmax, 5)
    draw.line((left, inner_top - 8, left, bottom), fill="#333333", width=2)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    for tick in ticks:
        frac = 0.0 if vmax == vmin else (float(tick) - vmin) / (vmax - vmin)
        tx = left + frac * (right - left)
        draw.line((tx, bottom, tx, inner_top - 8), fill="#e2e2e2", width=1)
        label = _fmt_tick(float(tick))
        tw, _ = _measure(draw, label, tick_font)
        draw.text((tx - tw / 2, bottom + 8), label, fill="#444444", font=tick_font)
    step = (bottom - inner_top) / max(1, len(labels))
    for idx, (label, group) in enumerate(zip(labels, data)):
        group_arr = np.asarray(group, dtype=float)
        group_arr = group_arr[np.isfinite(group_arr)]
        cy = inner_top + idx * step + step / 2
        draw.text((box[0] + 14, cy - 8), label, fill="#222222", font=text_font)
        if group_arr.size == 0:
            continue
        q1, q2, q3 = np.quantile(group_arr, [0.25, 0.5, 0.75])
        gmin = float(np.min(group_arr))
        gmax = float(np.max(group_arr))

        def _x(val: float) -> float:
            frac = 0.0 if vmax == vmin else (val - vmin) / (vmax - vmin)
            return left + frac * (right - left)

        min_x, q1_x, med_x, q3_x, max_x = map(_x, [gmin, q1, q2, q3, gmax])
        draw.line((min_x, cy, q1_x, cy), fill="#444444", width=2)
        draw.line((q3_x, cy, max_x, cy), fill="#444444", width=2)
        draw.line((min_x, cy - 8, min_x, cy + 8), fill="#444444", width=2)
        draw.line((max_x, cy - 8, max_x, cy + 8), fill="#444444", width=2)
        draw.rounded_rectangle((q1_x, cy - 12, q3_x, cy + 12), radius=6, fill=PALETTE[idx % len(PALETTE)], outline="#444444", width=1)
        draw.line((med_x, cy - 12, med_x, cy + 12), fill="white", width=3)
    xw, _ = _measure(draw, xlabel, text_font)
    draw.text((left + (right - left - xw) / 2, height - 36), xlabel, fill="#444444", font=text_font)
    canvas.save(target)
    return target


def save_panel_bar_chart(
    path: str | Path,
    panels: list[dict[str, Any]],
    title: str,
    cols: int = 2,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not pillow_available():
        raise RuntimeError("Pillow is unavailable.")
    rows = max(1, math.ceil(len(panels) / cols))
    panel_w = 820
    panel_h = 420
    width = cols * panel_w + 40
    height = 110 + rows * panel_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    top = _draw_title(draw, canvas, title)
    for idx, panel in enumerate(panels):
        col = idx % cols
        row = idx // cols
        x0 = 20 + col * panel_w
        y0 = top + row * panel_h
        x1 = x0 + panel_w - 20
        y1 = y0 + panel_h - 24
        _draw_horizontal_bar_panel(
            draw,
            (x0, y0, x1, y1),
            panel.get("labels", []),
            panel.get("values", []),
            panel.get("title", ""),
            panel.get("xlabel", ""),
            colors=panel.get("colors"),
        )
    canvas.save(target)
    return target


def _draw_line_panel(
    draw: Any,
    box: tuple[int, int, int, int],
    series: list[dict[str, Any]],
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    diagonal: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill="#fafafa", outline="#d8d8d8", width=1)
    title_font = _font(20, bold=True)
    text_font = _font(14)
    tick_font = _font(12)
    draw.text((x0 + 18, y0 + 12), title, fill="#111111", font=title_font)
    left = x0 + 74
    right = x1 - 18
    top = y0 + 54
    bottom = y1 - 52
    x_values = [float(v) for row in series for v in row.get("x", []) if np.isfinite(v)]
    y_values = [float(v) for row in series for v in row.get("y", []) if np.isfinite(v)]
    x_min, x_max = xlim if xlim else _nice_limits(x_values, floor_zero=False)
    y_min, y_max = ylim if ylim else _nice_limits(y_values, floor_zero=False)
    if diagonal:
        x_values.extend([0.0, 1.0])
        y_values.extend([0.0, 1.0])
        x_min, x_max = (0.0, 1.0)
        y_min, y_max = (0.0, 1.0)
    if y_min > 0 and y_max <= 1.05:
        y_min = 0.0

    draw.line((left, top, left, bottom), fill="#333333", width=2)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    x_ticks = np.linspace(x_min, x_max, 5)
    y_ticks = np.linspace(y_min, y_max, 5)
    for tick in x_ticks:
        frac = 0.0 if x_max == x_min else (float(tick) - x_min) / (x_max - x_min)
        tx = left + frac * (right - left)
        draw.line((tx, bottom, tx, top), fill="#ececec", width=1)
        label = _fmt_tick(float(tick))
        tw, _ = _measure(draw, label, tick_font)
        draw.text((tx - tw / 2, bottom + 8), label, fill="#444444", font=tick_font)
    for tick in y_ticks:
        frac = 0.0 if y_max == y_min else (float(tick) - y_min) / (y_max - y_min)
        ty = bottom - frac * (bottom - top)
        draw.line((left, ty, right, ty), fill="#ececec", width=1)
        label = _fmt_tick(float(tick))
        tw, th = _measure(draw, label, tick_font)
        draw.text((left - tw - 10, ty - th / 2), label, fill="#444444", font=tick_font)

    if diagonal:
        draw.line((left, bottom, right, top), fill="#666666", width=2)

    for idx, row in enumerate(series):
        xs = np.asarray(row.get("x", []), dtype=float)
        ys = np.asarray(row.get("y", []), dtype=float)
        if xs.size == 0 or ys.size == 0:
            continue
        pts = []
        for xv, yv in zip(xs, ys):
            if not np.isfinite(xv) or not np.isfinite(yv):
                continue
            x_frac = 0.0 if x_max == x_min else (xv - x_min) / (x_max - x_min)
            y_frac = 0.0 if y_max == y_min else (yv - y_min) / (y_max - y_min)
            px = left + x_frac * (right - left)
            py = bottom - y_frac * (bottom - top)
            pts.append((px, py))
        if len(pts) >= 2:
            color = row.get("color") or PALETTE[idx % len(PALETTE)]
            draw.line(pts, fill=color, width=3)

    legend_y = y0 + 18
    legend_x = right - 210
    for idx, row in enumerate(series[:8]):
        color = row.get("color") or PALETTE[idx % len(PALETTE)]
        ly = legend_y + idx * 20
        draw.rectangle((legend_x, ly + 4, legend_x + 14, ly + 14), fill=color)
        draw.text((legend_x + 20, ly), str(row.get("label", ""))[:28], fill="#222222", font=tick_font)

    xw, _ = _measure(draw, xlabel, text_font)
    draw.text((left + (right - left - xw) / 2, y1 - 30), xlabel, fill="#444444", font=text_font)
    draw.text((x0 + 10, top - 6), ylabel, fill="#444444", font=text_font)


def save_multi_line_chart(
    path: str | Path,
    title: str,
    series: list[dict[str, Any]],
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    diagonal: bool = False,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not pillow_available():
        raise RuntimeError("Pillow is unavailable.")
    canvas = Image.new("RGB", (1400, 860), "white")
    draw = ImageDraw.Draw(canvas)
    top = _draw_title(draw, canvas, title)
    _draw_line_panel(draw, (26, top, 1374, 836), series, "", xlabel, ylabel, xlim=xlim, ylim=ylim, diagonal=diagonal)
    canvas.save(target)
    return target


def save_two_panel_line_chart(
    path: str | Path,
    title: str,
    left_title: str,
    right_title: str,
    left_series: list[dict[str, Any]],
    right_series: list[dict[str, Any]],
    xlabel: str,
    left_ylabel: str,
    right_ylabel: str,
    xlim: tuple[float, float] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not pillow_available():
        raise RuntimeError("Pillow is unavailable.")
    canvas = Image.new("RGB", (1700, 880), "white")
    draw = ImageDraw.Draw(canvas)
    top = _draw_title(draw, canvas, title)
    _draw_line_panel(draw, (24, top, 836, 856), left_series, left_title, xlabel, left_ylabel, xlim=xlim)
    _draw_line_panel(draw, (860, top, 1672, 856), right_series, right_title, xlabel, right_ylabel, xlim=xlim)
    canvas.save(target)
    return target


def save_line_grid_chart(
    path: str | Path,
    title: str,
    panels: list[dict[str, Any]],
    cols: int = 2,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not pillow_available():
        raise RuntimeError("Pillow is unavailable.")
    rows = max(1, math.ceil(len(panels) / cols))
    panel_w = 820
    panel_h = 420
    width = cols * panel_w + 40
    height = 110 + rows * panel_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    top = _draw_title(draw, canvas, title)
    for idx, panel in enumerate(panels):
        col = idx % cols
        row = idx // cols
        x0 = 20 + col * panel_w
        y0 = top + row * panel_h
        x1 = x0 + panel_w - 20
        y1 = y0 + panel_h - 24
        _draw_line_panel(
            draw,
            (x0, y0, x1, y1),
            panel.get("series", []),
            panel.get("title", ""),
            panel.get("xlabel", "Epoch"),
            panel.get("ylabel", ""),
            xlim=panel.get("xlim"),
            ylim=panel.get("ylim"),
        )
    canvas.save(target)
    return target


def save_heatmap_table(
    path: str | Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    matrix: np.ndarray,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not pillow_available():
        raise RuntimeError("Pillow is unavailable.")
    cell_w = 110
    cell_h = 32
    row_label_w = 260
    header_h = 120
    width = row_label_w + len(col_labels) * cell_w + 40
    height = header_h + len(row_labels) * cell_h + 40
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    _draw_title(draw, canvas, title)
    header_font = _font(15, bold=True)
    cell_font = _font(14)
    x0 = row_label_w
    y0 = header_h
    for j, label in enumerate(col_labels):
        tw, th = _measure(draw, label, header_font)
        draw.text((x0 + j * cell_w + (cell_w - tw) / 2, y0 - 34), label, fill="#222222", font=header_font)
    for i, label in enumerate(row_labels):
        draw.text((24, y0 + i * cell_h + 8), label, fill="#222222", font=cell_font)
        for j, _ in enumerate(col_labels):
            value = float(matrix[i, j])
            shade = int(255 - (min(max(value, 0.0), 1.0) * 170))
            color = (shade, shade, shade)
            x = x0 + j * cell_w
            y = y0 + i * cell_h
            draw.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=color, outline="#ffffff")
            text = "Y" if value >= 0.5 else ""
            if text:
                tw, th = _measure(draw, text, header_font)
                draw.text((x + (cell_w - tw) / 2, y + (cell_h - th) / 2), text, fill="#111111", font=header_font)
    canvas.save(target)
    return target


def save_calibration_chart(
    path: str | Path,
    title: str,
    curves: list[dict[str, Any]],
) -> Path:
    series = []
    for idx, row in enumerate(curves):
        series.append(
            {
                "label": f"{row['label']} (ECE={row['ece']:.3f})",
                "x": row["x"],
                "y": row["y"],
                "color": PALETTE[idx % len(PALETTE)],
            }
        )
    return save_multi_line_chart(path, title, series, "Mean predicted probability", "Empirical positive rate", xlim=(0.0, 1.0), ylim=(0.0, 1.0), diagonal=True)
