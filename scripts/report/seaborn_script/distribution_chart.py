#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# ---- Data (from your CSV content) ----
df = pd.DataFrame({
    "label":   [0, 1, 2, 3],
    "NIST":    [5158, 1601, 1804, 697],
    "llm":     [3618, 2498, 1689, 1250],
    "%_NIST":  [55.7, 17.29, 19.48, 7.53],
    "%_llm":   [39.96, 27.59, 18.65, 13.8],
})

# We only need the percentages for this plot
pct = df[["label", "%_NIST", "%_llm"]].rename(columns={"%_NIST": "NIST", "%_llm": "LLM"})

# Long format for seaborn
pct_long = pct.melt(id_vars="label", var_name="Source", value_name="Percent")

# ---- Plot ----
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4.5))

ax = sns.barplot(data=pct_long, x="label", y="Percent", hue="Source")

# Make y-axis show percentages (0–100)
ax.yaxis.set_major_formatter(PercentFormatter(100))

# Titles & labels
ax.set_title("Label Distribution (%): NIST vs LLM", pad=12)
ax.set_xlabel("Label")
ax.set_ylabel("Percentage")

# Add value labels on bars
for p in ax.patches:
    height = p.get_height()
    if pd.notnull(height):
        ax.annotate(f"{height:.2f}%",
                    (p.get_x() + p.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

# Legend and layout
ax.legend(title="Source")
plt.tight_layout()

# Save (optional)
# plt.savefig("nist_llm_label_percent.png", dpi=200)

plt.show()
