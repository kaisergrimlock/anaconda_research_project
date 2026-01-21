from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

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


def _build_item_palette(items: Iterable[str]) -> Dict[str, Tuple[float, float, float, float]]:
    ordered = sorted(set(items))
    cmap = plt.get_cmap("tab10")
    return {item: cmap(i % cmap.N) for i, item in enumerate(ordered)}


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


def language_legend(
    ax: Axes,
    *,
    lang_to_rgba: Dict[str, Tuple[float, float, float, float]],
    title: str = "Language",
    loc: str = "lower right",
) -> None:
    from matplotlib.patches import Patch

    if not lang_to_rgba:
        return

    handles = [Patch(color=rgba, label=lang) for lang, rgba in sorted(lang_to_rgba.items())]
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


def _base_language(lang: str) -> str:
    return lang.split("_", 1)[0].strip()


def color_tukey_by_language(
    fig: Figure,
    ax: Axes,
    *,
    group_sep: str = "|",
    linewidth: float = 2.5,
    eps: float = 1e-6,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Color Tukey plot elements by base language. For example, "vi" and "vi_word"
    share the same color.
    """
    fig.canvas.draw()

    base_langs = []
    for tick in ax.get_yticklabels():
        text = tick.get_text()
        lang = text.split(group_sep)[-1].strip()
        base_langs.append(_base_language(lang))

    if not base_langs:
        return {}

    lang_to_rgba = _build_item_palette(base_langs)
    y_to_rgba: Dict[float, Tuple[float, float, float, float]] = {}

    for tick in ax.get_yticklabels():
        text = tick.get_text()
        lang = text.split(group_sep)[-1].strip()
        base_lang = _base_language(lang)
        rgba = lang_to_rgba[base_lang]
        tick.set_color(rgba)
        y_to_rgba[float(tick.get_position()[1])] = rgba

    # 1) Color CI bars
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

    # 2) Color mean dots
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue

        offsets = coll.get_offsets()
        if offsets is None or len(offsets) == 0:
            continue

        facecolors = coll.get_facecolors()
        if facecolors is None or len(facecolors) == 0:
            continue

        if len(facecolors) == 1:
            facecolors = facecolors.repeat(len(offsets), axis=0)

        for i, (x, y) in enumerate(offsets):
            y = float(y)
            for ty, rgba in y_to_rgba.items():
                if abs(y - ty) < eps:
                    facecolors[i] = rgba
                    break

        coll.set_facecolors(facecolors)

    return lang_to_rgba


def center_x_axis_at_zero(ax: Axes) -> None:
    xmin, xmax = ax.get_xlim()
    m = max(abs(xmin), abs(xmax))
    ax.set_xlim(-m, m)
    ax.axvline(0, linewidth=1)

def add_model_separators(
    fig: Figure,
    ax: Axes,
    *,
    group_sep: str = "|",
    linewidth: float = 1.0,
    alpha: float = 0.6,
    linestyle: str = "-",
    eps: float = 1e-6,
) -> None:
    """
    Draw horizontal separator lines between model blocks in a Tukey
    plot_simultaneous chart, assuming y tick labels look like:
        "<model>|<lang>"

    Uses the *actual* y tick positions, so it matches statsmodels' layout.
    """
    # Ensure tick label positions are finalized
    fig.canvas.draw()

    yticks = list(ax.get_yticks())
    ylabels = [t.get_text() for t in ax.get_yticklabels()]

    if not yticks or not ylabels:
        return

    # Pair and sort by y (bottom->top or top->bottom doesn't matter, we handle adjacency)
    pairs = [(float(y), lab) for y, lab in zip(yticks, ylabels) if str(lab).strip() != ""]
    if len(pairs) < 2:
        return
    pairs.sort(key=lambda t: t[0])

    def model_of(label: str) -> str:
        if group_sep in label:
            return label.split(group_sep, 1)[0].strip()
        return label.strip()

    # Find y positions where model changes between adjacent rows,
    # then draw a line halfway between those rows.
    for (y0, lab0), (y1, lab1) in zip(pairs[:-1], pairs[1:]):
        if model_of(lab0) != model_of(lab1):
            y_mid = (y0 + y1) / 2.0
            ax.axhline(y_mid, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
