# helpers/draw.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from settings import apply_paper_fmt


# make sure the script directory is on sys.path so "helpers" is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


RGBA = Tuple[float, float, float, float]


# =========================
# Public CSV loaders
# =========================
def load_lang_taxonomy(csv_path: Path) -> Dict[str, int]:
    """
    Read a CSV with columns: lang,taxonomy (taxonomy is an int).
    Returns a dict like:
        {"eng": 5, "vi": 4, ...}
    """
    df = pd.read_csv(csv_path)
    _validate_taxonomy_csv(df, csv_path)

    out: Dict[str, int] = {}
    for _, row in df.iterrows():
        lang = str(row["lang"]).strip()
        out[lang] = int(row["taxonomy"])
    return out

def load_categorized_lang_taxonomy(
    csv_path: Path,
    *,
    category_suffixes: Sequence[str] = (),
    extra_strip_suffixes: Sequence[str] = (),
    base_len: int = 2,
) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    _validate_taxonomy_csv(df, csv_path)

    strip_suffixes = tuple(category_suffixes) + tuple(extra_strip_suffixes)

    out: Dict[str, int] = {}
    for _, row in df.iterrows():
        lang_raw = str(row["lang"])
        key = normalize_categorized_language(
            lang_raw,
            category_suffixes=category_suffixes,
            extra_strip_suffixes=extra_strip_suffixes,
            base_len=base_len,
        )
        out[key] = int(row["taxonomy"])
    return out

# =========================
# Legends
# =========================
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
        if lvl == 0:
            continue
        handles.append(Patch(facecolor=rgba, edgecolor="none", label=f"Level {lvl}"))

    if not handles:
        return

    ax.legend(
        handles=handles,
        title=title,
        loc=loc,
        framealpha=0.9,
        edgecolor="black",
        fontsize="small",
        title_fontsize="medium",
    )

def categorized_variant_legend(
    ax: Axes,
    *,
    level_to_rgba: Dict[int, RGBA],
    category_label: str,
    baseline_label: str = "Before Passage",
    baseline_marker: str = "^",
    category_marker: str = "+",
    raw_band_label: str = "Baseline",
    raw_band_color: RGBA | str = "#7ec8e3",
    variant_title: str = "",
    taxonomy_title: str = "Taxonomy level",
    taxonomy_loc: str = "upper left",
    variant_bbox_to_anchor: Tuple[float, float] = (0.5, -0.12),
) -> None:
    """
    Generic legend for categorized content.

    - Taxonomy legend shows only taxonomy levels.
    - Variant legend shows:
        ^ baseline marker
        x/+ categorized marker
        colored square for raw baseline band
    """
    from matplotlib.patches import Patch

    if not level_to_rgba:
        return

    levels = sorted(level_to_rgba.keys())

    taxonomy_handles = []
    for lvl in levels:
        if lvl == 0:
            continue
        taxonomy_handles.append(
            Line2D(
                [0], [0],
                color=level_to_rgba[lvl],
                marker="o",
                linestyle="-",
                linewidth=2.0,
                markersize=7,
                label=f"Level {lvl}",
            )
        )

    if taxonomy_handles:
        legend_tax = ax.legend(
            handles=taxonomy_handles,
            title=taxonomy_title,
            loc=taxonomy_loc,
            framealpha=0.9,
            edgecolor="black",
            fontsize="small",
            title_fontsize="medium",
        )
        ax.add_artist(legend_tax)

    variant_handles = [
        Line2D(
            [0], [0],
            color="black",
            marker=baseline_marker,
            linestyle="None",
            markersize=10,
            label=baseline_label,
        ),
        Line2D(
            [0], [0],
            color="black",
            marker=category_marker,
            linestyle="None",
            markersize=12,
            markeredgewidth=1.8,
            label=category_label,
        ),
        Patch(
            facecolor=raw_band_color,
            edgecolor="none",
            alpha=0.22,
            label=raw_band_label,
        ),
    ]

    ax.legend(
        handles=variant_handles,
        title=variant_title,
        loc="upper center",
        bbox_to_anchor=variant_bbox_to_anchor,
        ncol=3,
        frameon=False,
        fontsize="medium",
        title_fontsize="medium",
        handletextpad=0.8,
        columnspacing=2.0,
        borderaxespad=0.0,
    )

# =========================
# Axis helpers
# =========================
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
    """
    fig.canvas.draw()

    pairs = _sorted_tick_pairs(ax)
    if len(pairs) < 2:
        return

    for (y0, lab0), (y1, lab1) in zip(pairs[:-1], pairs[1:]):
        if _model_of(lab0, group_sep) != _model_of(lab1, group_sep):
            ax.axhline(
                (y0 + y1) / 2.0,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
            )

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
    Add a single model label per contiguous block, positioned parallel to
    the y axis. Assumes y tick labels look like "<model>|<lang>".
    """
    fig.canvas.draw()

    pairs = _sorted_tick_pairs(ax)
    if not pairs:
        return

    i = 0
    while i < len(pairs):
        y_start, label = pairs[i]
        model = _model_of(label, group_sep)

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
    Color a Tukey plot directly from exact language labels in taxonomy_csv.
    """
    if apply_style:
        apply_paper_fmt()

    lang_to_level = load_lang_taxonomy(taxonomy_csv)

    seen_levels: set[int] = set(lang_to_level.values())
    if default_level is not None:
        seen_levels.add(default_level)

    level_to_rgba = _build_level_palette(seen_levels)

    def lang_to_color(label_lang: str) -> Optional[RGBA]:
        if label_lang in lang_to_level:
            lvl = lang_to_level[label_lang]
        elif default_level is not None:
            lvl = default_level
        else:
            return None
        return level_to_rgba[lvl]

    y_to_rgba = _color_ticks_and_build_y_map(
        fig,
        ax,
        group_sep=group_sep,
        color_for_lang=lang_to_color,
    )
    if not y_to_rgba:
        return {}

    _color_tukey_collections(ax, y_to_rgba=y_to_rgba, linewidth=linewidth, eps=eps)
    return level_to_rgba


def color_tukey_by_categorized_taxonomy(
    fig: Figure,
    ax: Axes,
    *,
    taxonomy_csv: Path,
    category_suffixes: Sequence[str],
    extra_strip_suffixes: Sequence[str] = ("_wo",),
    group_sep: str = "|",
    default_level: Optional[int] = None,
    linewidth: float = 2.5,
    eps: float = 1e-6,
    apply_style: bool = True,
    base_len: int = 2,
) -> Dict[int, RGBA]:

    if apply_style:
        apply_paper_fmt()

    lang_to_level = load_categorized_lang_taxonomy(
        taxonomy_csv,
        category_suffixes=category_suffixes,
        extra_strip_suffixes=extra_strip_suffixes,
        base_len=base_len,
    )

    seen_levels: set[int] = set(lang_to_level.values())
    if default_level is not None:
        seen_levels.add(default_level)

    level_to_rgba = _build_level_palette(seen_levels)

    def lang_to_color(label_lang: str) -> Optional[RGBA]:
        key = normalize_categorized_language(
            label_lang,
            group_sep=group_sep,
            category_suffixes=category_suffixes,
            extra_strip_suffixes=extra_strip_suffixes,
            base_len=base_len,
        )

        if key in lang_to_level:
            lvl = lang_to_level[key]
        elif default_level is not None:
            lvl = default_level
        else:
            return None

        return level_to_rgba[lvl]

    y_to_rgba = _color_ticks_and_build_y_map(
        fig,
        ax,
        group_sep=group_sep,
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

    if apply_style:
        apply_paper_fmt()

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
        color_for_lang=lang_to_color,
    )
    if not y_to_rgba:
        return {}

    _color_tukey_collections(ax, y_to_rgba=y_to_rgba, linewidth=linewidth, eps=eps)
    return lang_to_rgba


# =========================
# Internals
# =========================
def _validate_taxonomy_csv(df: pd.DataFrame, csv_path: Path) -> None:
    required = {"lang", "taxonomy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {csv_path}. "
            f"Got {list(df.columns)}"
        )

def _build_level_palette(levels: Iterable[int]) -> Dict[int, RGBA]:
    ordered = sorted(set(levels))
    cmap = plt.get_cmap("tab10")
    return {lvl: cmap(i % cmap.N) for i, lvl in enumerate(ordered)}


def _build_item_palette(items: Iterable[str]) -> Dict[str, RGBA]:
    ordered = sorted(set(items))
    cmap = plt.get_cmap("tab10")
    return {item: cmap(i % cmap.N) for i, item in enumerate(ordered)}

def _base_language(lang: str) -> str:
    return lang.split("_", 1)[0].strip()

def normalize_categorized_language(
        label_lang: str,
        *,
        group_sep: str = "|",
        category_suffixes: Sequence[str] = (),
        extra_strip_suffixes: Sequence[str] = (),
        base_len: int = 2,
    ) -> str:

    s = str(label_lang).strip().lower()

    if group_sep in s:
        s = s.split(group_sep, 1)[1]

    suffixes = tuple(category_suffixes) + tuple(extra_strip_suffixes)

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if suffix and s.endswith(suffix):
                s = s[: -len(suffix)]
                changed = True

    return s[:base_len]

def _lang_of(label: str, group_sep: str) -> str:
    return label.split(group_sep)[-1].strip()

def _model_of(label: str, group_sep: str) -> str:
    if group_sep in label:
        return label.split(group_sep, 1)[0].strip()
    return label.strip()

def _sorted_tick_pairs(ax: Axes) -> List[Tuple[float, str]]:
    yticks = list(ax.get_yticks())
    ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
    pairs = [
        (float(y), str(label))
        for y, label in zip(yticks, ylabels)
        if str(label).strip()
    ]
    pairs.sort(key=lambda pair: pair[0])
    return pairs

def _color_ticks_and_build_y_map(
    fig: Figure,
    ax: Axes,
    *,
    group_sep: str,
    color_for_lang: Callable[[str], Optional[RGBA] | RGBA],
) -> Dict[float, RGBA]:
    """
    Color y tick labels and return {tick_y: rgba} for matching plot artists.
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
        y_to_rgba[float(tick.get_position()[1])] = rgba

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
      - LineCollection segments (confidence intervals)
      - PathCollection facecolors (means)
    """
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

def _lookup_y_color(
    y: float,
    y_to_rgba: Mapping[float, RGBA],
    eps: float,
) -> Optional[RGBA]:
    for ty, rgba in y_to_rgba.items():
        if abs(y - ty) < eps:
            return rgba
    return None