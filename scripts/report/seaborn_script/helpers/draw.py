# helpers/draw.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Callable, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PathCollection

# Use settings.py from the same folder (helpers/)
import sys
from settings import paper_fmt  # keep only this here

# make sure the script directory is on sys.path so "helpers" is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


RGBA = Tuple[float, float, float, float]


# =========================
# Public helpers
# =========================
def load_lang_taxonomy(csv_path: Path) -> Dict[str, int]:
    """
    Read a CSV with columns: lang,taxonomy (taxonomy is an int).
    Returns: dict like {"eng": 5, "vi": 4, ...}
    """
    df = pd.read_csv(csv_path)
    if "lang" not in df.columns or "taxonomy" not in df.columns:
        raise ValueError(
            f"Expected columns lang,taxonomy in {csv_path}. Got {list(df.columns)}"
        )

    out: Dict[str, int] = {}
    for _, r in df.iterrows():
        lang = str(r["lang"]).strip()
        lvl = int(r["taxonomy"])
        out[lang] = lvl
    return out


def taxonomy_legend(
    ax: Axes,
    *,
    level_to_rgba: Dict[int, RGBA],
    title: str = "Taxonomy level",
    loc: str = "lower right",
) -> None:
    """Add a legend to ax mapping taxonomy levels to colors."""
    from matplotlib.patches import Patch

    if not level_to_rgba:
        return

    handles = []
    for lvl, rgba in sorted(level_to_rgba.items()):
        label = "Baseline" if lvl == 0 else f"Level {lvl}"
        handles.append(Patch(color=rgba, label=label))

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
    lang_to_rgba: Dict[str, RGBA],
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
) -> None:
    """
    Draw horizontal separator lines between model blocks in a Tukey
    plot_simultaneous chart, assuming y tick labels look like:
        "<model>|<lang>"

    Uses the *actual* y tick positions, so it matches statsmodels' layout.
    """
    fig.canvas.draw()

    pairs = _sorted_tick_pairs(ax)
    if len(pairs) < 2:
        return

    for (y0, lab0), (y1, lab1) in zip(pairs[:-1], pairs[1:]):
        if _model_of(lab0, group_sep) != _model_of(lab1, group_sep):
            ax.axhline((y0 + y1) / 2.0, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def add_model_prefix_labels(
    fig: Figure,
    ax: Axes,
    *,
    group_sep: str = "|",
    x: float = -0.08,
    rotation: float = 90,
    fontsize: str = "medium",
    fontweight: str = "bold",
    color: str = "black",
) -> None:
    """
    Add a single model label per contiguous block, positioned parallel to the y axis.
    Assumes y tick labels look like "<model>|<lang>".
    """
    fig.canvas.draw()

    pairs = _sorted_tick_pairs(ax)
    if not pairs:
        return

    i = 0
    while i < len(pairs):
        y_start, lab = pairs[i]
        model = _model_of(lab, group_sep)

        y_end = y_start
        j = i + 1
        while j < len(pairs) and _model_of(pairs[j][1], group_sep) == model:
            y_end = pairs[j][0]
            j += 1

        y_center = (y_start + y_end) / 2.0
        ax.text(
            x,
            y_center,
            model,
            transform=ax.get_yaxis_transform(),
            rotation=rotation,
            va="center",
            ha="right",
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            clip_on=False,
        )
        i = j


# =========================
# Tukey plot coloring
# =========================
def color_tukey_by_taxonomy(
    fig: Figure,
    ax: Axes,
    *,
    taxonomy_csv: Path,
    group_sep: str = "|",
    default_level: Optional[int] = None,
    linewidth: float = 2.5,
    eps: float = 1e-6,
    apply_style: bool = True,
) -> Dict[int, RGBA]:
    """
    After tukey.plot_simultaneous(ax=ax), color:
      - y tick labels
      - CI bars (LineCollection segments)
      - mean dots (PathCollection), if present
    using taxonomy levels loaded from taxonomy_csv.

    Group labels are assumed to end with: "<model>|<lang>"
    """
    if apply_style:
        apply_paper_fmt()

    lang_to_level = load_lang_taxonomy(taxonomy_csv)

    def lang_to_color(label_lang: str) -> Optional[RGBA]:
        if label_lang in lang_to_level:
            lvl = lang_to_level[label_lang]
        elif default_level is not None:
            lvl = default_level
        else:
            return None
        return level_to_rgba[lvl]

    # compute palette AFTER we know which levels can appear
    seen_levels: set[int] = set(lang_to_level.values())
    if default_level is not None:
        seen_levels.add(default_level)

    level_to_rgba = _build_level_palette(seen_levels)

    y_to_rgba = _color_ticks_and_build_y_map(
        fig,
        ax,
        group_sep=group_sep,
        eps=eps,
        color_for_lang=lang_to_color,
    )
    if not y_to_rgba:
        return {}

    _color_tukey_collections(ax, y_to_rgba=y_to_rgba, linewidth=linewidth, eps=eps)
    return level_to_rgba


def color_tukey_by_language(
    fig: Figure,
    ax: Axes,
    *,
    group_sep: str = "|",
    linewidth: float = 2.5,
    eps: float = 1e-6,
    apply_style: bool = True,
) -> Dict[str, RGBA]:
    """
    Color Tukey plot elements by base language. For example, "vi" and "vi_word"
    share the same color.
    """

    # build palette over base languages present in tick labels
    fig.canvas.draw()
    base_langs: List[str] = []
    for tick in ax.get_yticklabels():
        lang = _lang_of(tick.get_text(), group_sep)
        base_langs.append(_base_language(lang))
    if not base_langs:
        return {}

    lang_to_rgba = _build_item_palette(base_langs)

    def lang_to_color(label_lang: str) -> RGBA:
        return lang_to_rgba[_base_language(label_lang)]

    y_to_rgba = _color_ticks_and_build_y_map(
        fig,
        ax,
        group_sep=group_sep,
        eps=eps,
        color_for_lang=lang_to_color,
    )
    if not y_to_rgba:
        return {}

    _color_tukey_collections(ax, y_to_rgba=y_to_rgba, linewidth=linewidth, eps=eps)
    return lang_to_rgba


# =========================
# Internals
# =========================
def _build_level_palette(levels: Iterable[int]) -> Dict[int, RGBA]:
    """Map taxonomy levels -> RGBA using a categorical colormap."""
    ordered = sorted(set(levels))
    cmap = plt.get_cmap("tab10")
    return {lvl: cmap(i % cmap.N) for i, lvl in enumerate(ordered)}


def _build_item_palette(items: Iterable[str]) -> Dict[str, RGBA]:
    ordered = sorted(set(items))
    cmap = plt.get_cmap("tab10")
    return {item: cmap(i % cmap.N) for i, item in enumerate(ordered)}


def _base_language(lang: str) -> str:
    return lang.split("_", 1)[0].strip()


def _lang_of(label: str, group_sep: str) -> str:
    # "<model>|<lang>" -> "<lang>" (or whole label if no sep)
    return label.split(group_sep)[-1].strip()


def _model_of(label: str, group_sep: str) -> str:
    if group_sep in label:
        return label.split(group_sep, 1)[0].strip()
    return label.strip()


def _sorted_tick_pairs(ax: Axes) -> List[Tuple[float, str]]:
    yticks = list(ax.get_yticks())
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    pairs = [(float(y), str(lab)) for y, lab in zip(yticks, ylabels) if str(lab).strip() != ""]
    pairs.sort(key=lambda t: t[0])
    return pairs


def _color_ticks_and_build_y_map(
    fig: Figure,
    ax: Axes,
    *,
    group_sep: str,
    eps: float,
    color_for_lang: Callable[[str], Optional[RGBA] | RGBA],
) -> Dict[float, RGBA]:
    """
    Color y tick labels and return {tick_y: rgba} for matching plot artists.
    Uses the *actual* tick y positions (after draw).
    """
    fig.canvas.draw()

    y_to_rgba: Dict[float, RGBA] = {}
    for tick in ax.get_yticklabels():
        label = tick.get_text()
        lang = _lang_of(label, group_sep)
        rgba = color_for_lang(lang)
        if rgba is None:
            continue

        tick.set_color(rgba)
        y = float(tick.get_position()[1])
        # store exact y; matching uses eps later
        y_to_rgba[y] = rgba

    return y_to_rgba


def _color_tukey_collections(
    ax: Axes,
    *,
    y_to_rgba: Mapping[float, RGBA],
    linewidth: float,
    eps: float,
) -> None:
    """
    Apply y->color mapping to:
      - LineCollection segments (CIs)
      - PathCollection facecolors (means)
    """
    # 1) CI bars: LineCollections
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
            rgba = _lookup_y_color(y, y_to_rgba, eps)
            if rgba is not None:
                colors[i] = rgba

        coll.set_color(colors)
        coll.set_linewidth(linewidth)

    # 2) mean dots: PathCollections
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

        for i, (_x, y) in enumerate(offsets):
            rgba = _lookup_y_color(float(y), y_to_rgba, eps)
            if rgba is not None:
                facecolors[i] = rgba

        coll.set_facecolors(facecolors)


def _lookup_y_color(y: float, y_to_rgba: Mapping[float, RGBA], eps: float) -> Optional[RGBA]:
    for ty, rgba in y_to_rgba.items():
        if abs(y - ty) < eps:
            return rgba
    return None
