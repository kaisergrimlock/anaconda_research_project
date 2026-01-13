#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Load the CSV ======
LANG = "ga"  # Change as needed: "raw", "vi", "fr", ...
YEAR = "2023"
df = pd.read_csv (f"outputs/baseline/{YEAR}/{LANG}/confusion_matrix_llm_vs_nist_pct.csv", index_col=0)

# ====== Plot setup ======
plt.figure(figsize=(6, 5))
ax = sns.heatmap(
    df,
    annot=True, fmt=".2f", cmap="YlGnBu",
    vmin=0, vmax=100,        # Fixed 0–100% color range
    cbar_kws={'label': 'Percentage (%)'},
    linewidths=0.5, linecolor='gray'
)


# ====== Axis labels ======
ax.set_xlabel("LLM / Predicted Label")
ax.set_ylabel("NIST / True Label")
ax.set_title("Confusion Matrix (%)")

plt.tight_layout()
plt.show()
