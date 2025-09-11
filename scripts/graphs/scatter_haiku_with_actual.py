#!/usr/bin/env python3
"""
Scatter plot of per-docid relevance across multiple runs,
with the ground-truth (actual_rel) overlaid as big red triangles.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- Config -----------------------------
INPUT = Path("outputs/trec_dl_llm_label/scatter_haiku_with_actual.csv")
OUT   = Path("outputs/trec_dl_llm_graphs/scatter_haiku_by_docid_offset_actual.pdf")

POINT_SIZE   = 4         # size for run points
JITTER       = 0.25      # horizontal jitter within each docid
TRI_SIZE     = 140       # size for actual_rel triangles
TRI_COLOR    = "red"
TRI_EDGE     = "black"
YLIM         = (-0.2, 3.2)
YTICKS       = [0, 1, 2, 3]
BAND_ALPHA   = 0.04      # alternating band opacity
GRID_COLOR   = "0.85"    # vertical separator lines
# -----------------------------------------------------------------

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    # Ensure docid is categorical in CSV order
    df["docid"] = df["docid"].astype(str)
    order = pd.unique(df["docid"])
    df["docid"] = pd.Categorical(df["docid"], categories=order, ordered=True)

    # Long-form data: one row per (docid, run)
    rel_cols = [c for c in df.columns if c.startswith("rel_")]
    if not rel_cols:
        raise ValueError("No columns starting with 'rel_' found.")

    # Sort rel columns by trailing integer if present (rel_1, rel_2, ...)
    def run_key(c):
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return c
    rel_cols = sorted(rel_cols, key=run_key)

    long = (
        df.melt(id_vars=["docid"], value_vars=rel_cols,
                var_name="run", value_name="relevance")
          .dropna(subset=["relevance"])
    )

    # Style & figure
    sns.set_theme(style="whitegrid", context="paper")

    # Width heuristic for many docids
    fig_width = max(12, 0.25 * len(order))
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Categorical scatter with dodge+jitter for runs
    sns.stripplot(
        data=long, x="docid", y="relevance", hue="run",
        dodge=True, jitter=JITTER, alpha=0.9, size=POINT_SIZE, ax=ax
    )

    # Axes labels/limits
    ax.set_xlabel("docid")
    ax.set_ylabel("Relevance")
    ymin, ymax = YLIM
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(YTICKS)

    # Vertical separators between docids
    n = len(order)
    for x in np.arange(-0.5, n - 0.5 + 1, 1.0):
        ax.axvline(x, color=GRID_COLOR, linewidth=0.8, zorder=0)

    # Alternating light bands per docid
    for i in range(n):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, alpha=BAND_ALPHA, zorder=0)

    # --- Overlay ground-truth as big red triangles ---
    if "actual_rel" in df.columns:
        truth = df[["docid", "actual_rel"]].dropna()
        if not truth.empty:
            ax.scatter(
                truth["docid"], truth["actual_rel"],
                marker="^", s=TRI_SIZE, color=TRI_COLOR,
                edgecolor=TRI_EDGE, linewidth=0.7, zorder=5, label="Actual"
            )
    else:
        print("Warning: 'actual_rel' column not found; no truth overlay drawn.")

    # Keep points above grid/bands
    ax.set_axisbelow(True)

    # Readability for many docids
    plt.xticks(rotation=90)

    # Legend (after adding triangles) — keep 'Actual' last if present
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Remove duplicate legend entries that stripplot sometimes adds
        seen = {}
        uniq_handles, uniq_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                uniq_handles.append(h)
                uniq_labels.append(l)

        # Move 'Actual' to end if present
        if "Actual" in uniq_labels:
            idx = uniq_labels.index("Actual")
            h_actual = uniq_handles.pop(idx)
            l_actual = uniq_labels.pop(idx)
            uniq_handles.append(h_actual)
            uniq_labels.append(l_actual)

        ax.legend(uniq_handles, uniq_labels, title="Run / Actual",
                  bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax.legend().remove()

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
