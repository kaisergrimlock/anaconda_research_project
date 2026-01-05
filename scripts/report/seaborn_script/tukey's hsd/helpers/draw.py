from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PathCollection


def load_lang_taxonomy(csv_path: Path) -> Dict[str, int]:
    """
    Read a CSV with columns: lang,taxonomy (taxonomy is an int).
    Returns: dict like {"eng": 5, "vi": 4, ...}
    """
    df = pd.read_csv(csv_path)
    if "lang" not in df.columns or "taxonomy" not in df.columns:
        raise ValueError(f"Expected columns lang,taxonomy in {csv_path}. Got {list(df.columns)}")

    out: Dict[str, int] = {}
    for _, r in df.iterrows():
        lang = str(r["lang"]).strip()
        lvl = int(r["taxonomy"])
        out[lang] = lvl
    return out


def _build_level_palette(levels: set[int]):
    """
    Map taxonomy levels -> RGBA colors using a Matplotlib categorical colormap.
    (Avoids hardcoding specific color names.)
    """
    ordered = sorted(levels)
    cmap = plt.get_cmap("tab10")  # categorical palette with distinct colors
    level_to_rgba = {lvl: cmap(i % cmap.N) for i, lvl in enumerate(ordered)}
    return level_to_rgba


def taxonomy_legend(
            ax: Axes,
    *,
    level_to_rgba: Dict[int, Tuple[float, float, float, float]],
    title: str = "Taxonomy level",
    loc: str = "lower right",
) -> None:
    """
    Add a legend to ax mapping taxonomy levels to colors.
    """
    from matplotlib.patches import Patch

    handles = []
    for lvl, rgba in sorted(level_to_rgba.items()):
        if lvl != 0:
            label = f"Level {lvl}"
        else:
            label = "Baseline"
        patch = Patch(color=rgba, label=label)
        handles.append(patch)

    ax.legend(
        handles=handles,
        title=title,
        loc=loc,
        framealpha=0.9,
        edgecolor="black",
        fontsize="small",
        title_fontsize="medium",
    ) 

def color_tukey_by_taxonomy(
    fig: Figure,
    ax: Axes,
    *,
    taxonomy_csv: Path,
    group_sep: str = "|",
    default_level: Optional[int] = None,
    linewidth: float = 2.5,
    eps: float = 1e-6,
) -> dict[int, tuple[float, float, float, float]]:
    """
    After tukey.plot_simultaneous(ax=ax), color:
      - y tick labels
      - CI bars (LineCollection segments)
      - mean dots (PathCollection), if present
    using taxonomy levels loaded from taxonomy_csv.

    Group labels are assumed to end with: "<model>|<lang>"
    """
    lang_to_level = load_lang_taxonomy(taxonomy_csv)

    # Finalize artists & tick label positions
    fig.canvas.draw()

    # Determine which levels appear (including default if used)
    seen_levels: set[int] = set(lang_to_level.values())
    if default_level is not None:
        seen_levels.add(default_level)

    level_to_rgba = _build_level_palette(seen_levels)

    # Build mapping from y-position -> color based on tick labels
    y_to_rgba: Dict[float, Tuple[float, float, float, float]] = {}

    for tick in ax.get_yticklabels():
        text = tick.get_text()
        lang = text.split(group_sep)[-1].strip()

        if lang in lang_to_level:
            lvl = lang_to_level[lang]
        elif default_level is not None:
            lvl = default_level
        else:
            # No taxonomy for this label; skip coloring it
            continue

        rgba = level_to_rgba[lvl]
        tick.set_color(rgba)
        y_to_rgba[float(tick.get_position()[1])] = rgba

    if not y_to_rgba:
        return

    # 1) Color CI bars: they live in ax.collections as LineCollection
    for coll in ax.collections:
        if not isinstance(coll, LineCollection):
            continue

        segs = coll.get_segments()
        if not segs:
            continue

        colors = coll.get_colors()
        if len(colors) == 1:
            colors = [colors[0]] * len(segs)

        for i, seg in enumerate(segs):
            y = float(seg[0][1])
            for ty, rgba in y_to_rgba.items():
                if abs(y - ty) < eps:
                    colors[i] = rgba
                    break

        coll.set_color(colors)
        coll.set_linewidth(linewidth)

    # 2) Color mean dots if present (often a PathCollection)
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue

        offsets = coll.get_offsets()
        if offsets is None or len(offsets) == 0:
            continue

        facecolors = coll.get_facecolors()
        if facecolors is None or len(facecolors) == 0:
            continue

        # Ensure one color per point
        if len(facecolors) == 1:
            facecolors = facecolors.repeat(len(offsets), axis=0)

        for i, (x, y) in enumerate(offsets):
            y = float(y)
            for ty, rgba in y_to_rgba.items():
                if abs(y - ty) < eps:
                    facecolors[i] = rgba
                    break

        coll.set_facecolors(facecolors)

    return level_to_rgba


def center_x_axis_at_zero(ax: Axes) -> None:
    xmin, xmax = ax.get_xlim()
    m = max(abs(xmin), abs(xmax))
    ax.set_xlim(-m, m)
    ax.axvline(0, linewidth=1)
