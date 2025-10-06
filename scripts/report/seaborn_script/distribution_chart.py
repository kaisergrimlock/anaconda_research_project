#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from pathlib import Path

# ========= Configure this =========
# Option A: long format produced by your scripts: label,no. of docs,judge (judge in {"NIST","llm"})
INPUT_CSV = Path("outputs/baseline/label_counts.csv")

# Option B (alternative): if you already have a wide CSV with columns:
# label,NIST,llm,%_NIST,%_llm then point INPUT_CSV to that file instead.
# ==================================

def load_percent_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Wide format present? (already has percentage columns)
    wide_cols = {"label", "%_NIST", "%_llm"}
    if wide_cols.issubset(df.columns):
        pct = df[["label", "%_NIST", "%_llm"]].rename(columns={"%_NIST": "NIST", "%_llm": "LLM"})
        return pct

    # Otherwise assume long format: label,no. of docs,judge
    required = {"label", "no. of docs", "judge"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must have either columns {sorted(wide_cols)} or {sorted(required)}. "
            f"Found: {list(df.columns)}"
        )

    # normalize judge names (NIST/llm) and labels to int
    df = df.copy()
    df["judge"] = df["judge"].str.strip().str.upper().map({"NIST": "NIST", "LLM": "LLM"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["judge", "label"])

    # keep only labels 0..3; anything else can be mapped to 0 if you like.
    df = df[df["label"].isin([0, 1, 2, 3])]

    # pivot to counts per judge x label
    pivot = df.pivot_table(index="label", columns="judge", values="no. of docs", aggfunc="sum", fill_value=0)

    # compute percentages per judge
    for col in ["NIST", "LLM"]:
        if col in pivot.columns:
            total = pivot[col].sum()
            pivot[f"%_{col}"] = pivot[col] / total * 100 if total > 0 else 0.0
        else:
            pivot[f"%_{col}"] = 0.0

    pct = pivot.reset_index()[["label", "%_NIST", "%_LLM"]].rename(columns={"%_LLM": "LLM", "%_NIST": "NIST"})
    return pct

# ---- Load & reshape ----
pct = load_percent_data(INPUT_CSV)
pct_long = pct.melt(id_vars="label", var_name="Source", value_name="Percent")

# ---- Plot ----
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4.5))

ax = sns.barplot(data=pct_long, x="label", y="Percent", hue="Source")
ax.yaxis.set_major_formatter(PercentFormatter(100))
ax.set_title("Label Distribution (%): NIST vs LLM", pad=12)
ax.set_xlabel("Label")
ax.set_ylabel("Percentage")

for p in ax.patches:
    h = p.get_height()
    if pd.notnull(h):
        ax.annotate(f"{h:.2f}%",
                    (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

ax.legend(title="Source")
plt.tight_layout()

# Save (optional)
# Path("outputs/baseline/plots").mkdir(parents=True, exist_ok=True)
# plt.savefig("outputs/baseline/plots/label_distribution_percent.png", dpi=200)

plt.show()
