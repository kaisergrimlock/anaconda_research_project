# scatter_by_docid_offset.py
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

INPUT = Path("outputs/trec_dl_llm_label/scatter_mixtral.csv")
OUT   = Path("outputs/trec_dl_llm_graphs/scatter_mixtral_by_docid_offset.pdf")

df = pd.read_csv(INPUT)

# Make docid a discrete/categorical label (preserve CSV order)
df["docid"] = df["docid"].astype(str)
order = pd.unique(df["docid"])
df["docid"] = pd.Categorical(df["docid"], categories=order, ordered=True)

# Long form: one row per (docid, run)
rel_cols = [c for c in df.columns if c.startswith("rel_")]
long = df.melt(id_vars=["docid"], value_vars=rel_cols,
               var_name="run", value_name="relevance").dropna()

sns.set_theme(style="whitegrid", context="paper")
fig, ax = plt.subplots(figsize=(12, 5))

# stripplot: categorical x; dodge separates hues; jitter spreads within docid
sns.stripplot(
    data=long, x="docid", y="relevance", hue="run",
    dodge=True, jitter=0.25, alpha=0.9, size=4, ax=ax
)

ax.set_xlabel("docid")
ax.set_ylabel("Relevance")
ax.set_ylim(-0.2, 3.2)
ax.set_yticks([0, 1, 2, 3])
ax.legend(title="Run", bbox_to_anchor=(1.01, 1), loc="upper left")

# draw vertical lines between docid categories
n = len(order)                     # 'order' is the categorical order you built
ymin, ymax = -0.2, 3.2             # same y-limits as your plot
for x in np.arange(-0.5, n-0.5+1, 1.0):
    ax.axvline(x, color="0.85", linewidth=0.8, zorder=0)

# (optional) alternate light background bands per docid
for i in range(n):
    if i % 2 == 0:
        ax.axvspan(i-0.5, i+0.5, alpha=0.04, zorder=0)

ax.set_ylim(ymin, ymax)
ax.set_axisbelow(True)  # keep points above the separators

# readability for many docids
plt.xticks(rotation=90)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}")

