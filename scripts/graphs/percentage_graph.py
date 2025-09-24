#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# ---- Data (percentages must sum to ~100) ----
parts = pd.Series(
    [21.11, 19.99, 26.81, 32.09], 
    index=[3, 2, 1, 0], name="percent"
)

# Order top-to-bottom in the stack (3 → 0)
order = [3, 2, 1, 0]
parts = parts.loc[order]

# ---- Plot ----
sns.set_theme()
fig, ax = plt.subplots(figsize=(6, 4))

bottom = 0
xpos = 0  # single bar position
for lbl, pct in parts.items():
    ax.bar(xpos, pct, bottom=bottom, label=f"Label {lbl}")
    # annotate the slice (skip very tiny ones)
    if pct >= 2:
        ax.text(xpos, bottom + pct/2, f"{pct:.2f}%", ha="center", va="center", fontsize=9)
    bottom += pct

ax.set_xlim(-0.6, 0.6)
ax.set_xticks([xpos])
ax.set_xticklabels(["TREC-DL 2019"])
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel("Percentage of all documents")
ax.set_title("Relevance distribution — TREC-DL 2019")

ax.legend(title="Relevance label", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
# plt.savefig("trecdl2019_relevance_stacked_bar.png", dpi=200)
