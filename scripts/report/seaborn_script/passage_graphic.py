#!/usr/bin/env python3
from __future__ import annotations

import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# -------------------------
# Content
# -------------------------
query_id = "2000511"
query = "average bahamas temperature at the end of october"

left_title = "Original Document"
right_title = "Bracket-Injection"
left_title_row2 = "Injection Before Passage"
right_title_row2 = "Injection After Passage"
left_title_row3 = "Query Phrase Injection"
right_title_row3 = "Query Word Injection"

passage_a = (
    "The temperature in The Bahamas will typically decline by early evening, to approximately 78.2 degrees. For the month of October as a whole, temperatures in The Bahamas average around 78.8 degrees. October Temperatures in The Bahamas by Time of Day. Early Morning."
)

passage_b = (
    "The temperature in The Bahamas will typically decline by early evening, to approximately 78.2 degrees. For (nhiệt độ ở Bahamas trung bình vào cuối tháng 10)the month of October as a whole, temperatures in The Bahamas average around 78.8 degrees. October Temperatures in The Bahamas by Time of Day. Early Morning."
)

passage_c = (
    "nhiệt độ ở Bahamas trung bình vào cuối tháng 10. The temperature in The Bahamas will typically decline by early evening, to approximately 78.2 degrees. For the month of October as a whole, temperatures in The Bahamas average around 78.8 degrees. October Temperatures in The Bahamas by Time of Day. Early Morning."
)

passage_d = (
    "The temperature in The Bahamas will typically decline by early evening, to approximately 78.2 degrees. For the month of October as a whole, temperatures in The Bahamas average around 78.8 degrees. October Temperatures in The Bahamas by Time of Day. Early Morning. nhiệt độ ở Bahamas trung bình vào cuối tháng 10 "
)

passage_e = (
    "The temperature in The Bahamas will typically nhiệt độ ở Bahamas trung bình vào cuối tháng 10 decline by early evening, to approximately 78.2 degrees. For the month of October as a whole, temperatures in The Bahamas average around 78.8 degrees. October Temperatures in The Bahamas by Time of Day. Early Morning."
)

passage_f = (
    "The temperature in The Bahamas trung vào cuối will typically decline by early evening, to approximately 78.2 bình degrees. For the month of độ 10 October as a whole, Bahamas temperatures tháng in nhiệt The Bahamas average around 78.8 degrees. October Temperatures in The Bahamas by Time ở of Day. Early Morning."
)

# If you still want an arrow label between boxes, set True
DRAW_ARROW = False

# -------------------------
# Helpers
# -------------------------
def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width=width))

# -------------------------
# Styling knobs
# -------------------------
bg = "#ffffff"
panel_bg = "#f7f7f7"
stroke = "#c0c0c0"
text = "#111111"
muted = "#333333"

fig_width_in = 12.5
base_fig_height_in = 2.8
dpi = 200

body_fontsize = 9.2
title_fontsize = 10.0
query_id_fontsize = 10.5
query_fontsize = 10.0
linespacing = 1.25

panel_w_ax = 0.38
inner_pad_px = 12
panel_top_ax = 0.68
panel_min_bottom_ax = 0.20
row_gap_ax = 0.10

char_px = (body_fontsize * dpi / 72.0) * 0.60
panel_width_px = fig_width_in * dpi * panel_w_ax - (2 * inner_pad_px)
wrap_width = max(20, int(panel_width_px / char_px))

wrapped_a = wrap(passage_a, wrap_width)
wrapped_b = wrap(passage_b, wrap_width)
wrapped_c = wrap(passage_c, wrap_width)
wrapped_d = wrap(passage_d, wrap_width)
wrapped_e = wrap(passage_e, wrap_width)
wrapped_f = wrap(passage_f, wrap_width)
lines_row1 = max(wrapped_a.count("\n") + 1, wrapped_b.count("\n") + 1)
lines_row2 = max(wrapped_c.count("\n") + 1, wrapped_d.count("\n") + 1)
lines_row3 = max(wrapped_e.count("\n") + 1, wrapped_f.count("\n") + 1)
max_lines = max(lines_row1, lines_row2, lines_row3)

text_height_in = (max_lines * body_fontsize * linespacing) / 72.0
padding_in = 0.16
required_panel_in = text_height_in + padding_in
max_panel_ax = panel_top_ax - panel_min_bottom_ax
panel_h_ax = required_panel_in / base_fig_height_in
total_panel_ax = (panel_h_ax * 3) + (row_gap_ax * 2)
if total_panel_ax > max_panel_ax:
    fig_height_in = max(
        base_fig_height_in,
        (required_panel_in * 3) / (max_panel_ax - (row_gap_ax * 2)),
    )
    panel_h_ax = required_panel_in / fig_height_in
else:
    fig_height_in = base_fig_height_in

# Figure size: expands in height when text grows
fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

# Outer container (border removed)

# Header: Query ID + Query
ax.text(
    0.06, 0.86,
    f"Query ID: {query_id}",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=query_id_fontsize,
    fontfamily="DejaVu Sans",
    color=text,
    fontweight="bold"
)
ax.text(
    0.06, 0.78,
    f"Query: {query}",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=query_fontsize,
    fontfamily="DejaVu Sans",
    color=text
)

# Panel geometry (six inner boxes)
left = dict(x=0.09, y=0.0, w=panel_w_ax, h=0.0)
right = dict(x=0.52, y=0.0, w=panel_w_ax, h=0.0)
left2 = dict(x=0.09, y=0.0, w=panel_w_ax, h=0.0)
right2 = dict(x=0.52, y=0.0, w=panel_w_ax, h=0.0)
left3 = dict(x=0.09, y=0.0, w=panel_w_ax, h=0.0)
right3 = dict(x=0.52, y=0.0, w=panel_w_ax, h=0.0)

def add_panel(panel, title, body, body_fontsize, y_offset):
    # Title (above panel)
    ax.text(
        panel["x"], panel["y"] + panel["h"] + y_offset,
        title,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=title_fontsize,
        fontfamily="DejaVu Sans",
        color=muted,
        fontweight="bold"
    )

    # Panel box
    box = FancyBboxPatch(
        (panel["x"], panel["y"]), panel["w"], panel["h"],
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0.9,
        edgecolor=stroke,
        facecolor=panel_bg,
        transform=ax.transAxes
    )
    ax.add_patch(box)

    # Panel text
    ax.text(
        panel["x"] + 0.015, panel["y"] + panel["h"] - 0.02,
        body,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=body_fontsize,
        fontfamily="DejaVu Sans",
        color=text,
        linespacing=1.25
    )

title_gap = 0.035
panel_y_row1 = panel_top_ax - panel_h_ax
panel_y_row2 = panel_y_row1 - row_gap_ax - panel_h_ax
panel_y_row3 = panel_y_row2 - row_gap_ax - panel_h_ax

left["h"] = panel_h_ax
right["h"] = panel_h_ax
left["y"] = panel_y_row1
right["y"] = panel_y_row1
left2["h"] = panel_h_ax
right2["h"] = panel_h_ax
left2["y"] = panel_y_row2
right2["y"] = panel_y_row2
left3["h"] = panel_h_ax
right3["h"] = panel_h_ax
left3["y"] = panel_y_row3
right3["y"] = panel_y_row3


add_panel(left, left_title, wrapped_a, body_fontsize, title_gap)
add_panel(right, right_title, wrapped_b, body_fontsize, title_gap)
add_panel(left2, left_title_row2, wrapped_c, body_fontsize, title_gap)
add_panel(right2, right_title_row2, wrapped_d, body_fontsize, title_gap)
add_panel(left3, left_title_row3, wrapped_e, body_fontsize, title_gap)
add_panel(right3, right_title_row3, wrapped_f, body_fontsize, title_gap)

# Optional arrow (if you want)
if DRAW_ARROW:
    arrow = FancyArrowPatch(
        (left["x"] + left["w"], left["y"] + left["h"] * 0.50),
        (right["x"], right["y"] + right["h"] * 0.50),
        transform=ax.transAxes,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.2,
        color=muted
    )
    ax.add_patch(arrow)

# Save
out_pdf = "passage_pair_card.pdf"
out_png = "passage_pair_card.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
print(f"Wrote {out_pdf} and {out_png}")
