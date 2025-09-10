import pandas as pd
from pathlib import Path

# --- Paths based on your structure ---
base_path = Path("outputs")
haiku_path = base_path / "trec_dl_llm_label" / "scatter_mixtral.csv"
top2_path  = base_path / "trec_dl" / "combined_results_top2.csv"
output_path = base_path / "trec_dl_llm_label" / "scatter_mixtral_with_actual.csv"

# --- Load CSVs ---
haiku = pd.read_csv(haiku_path)
top2 = pd.read_csv(top2_path)

# --- Ensure docid types match ---
haiku["docid"] = pd.to_numeric(haiku["docid"], errors="coerce")
top2["docid"] = pd.to_numeric(top2["docid"], errors="coerce")

# --- Remove duplicates: keep max relevance if multiple entries per docid ---
top2_clean = (top2
              .dropna(subset=["docid"])
              .groupby("docid", as_index=False)["relevance"]
              .max())

# --- Merge relevance into haiku data ---
merged = haiku.merge(top2_clean, on="docid", how="left")
merged = merged.rename(columns={"relevance": "actual_rel"})

# --- Convert to integer type with support for NA values ---
merged["actual_rel"] = pd.to_numeric(merged["actual_rel"], errors="coerce").astype("Int64")

# --- Save output ---
merged.to_csv(output_path, index=False)

print(f"Done! File saved at: {output_path}")
