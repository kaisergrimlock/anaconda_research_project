import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict

# --- Config ---
ROOT = Path("outputs/llm_label")
PATTERN = "llm_labels_q524332_*.tsv"

# --- Helpers ---
def is_original(p: Path) -> bool:
    """Original has no language/translation tag in name."""
    name = p.name.lower()
    return not any(tag in name for tag in ["viet", "eng_2", "eng", "thai", "translated"])

def variant_of(p: Path) -> str:
    """Return a readable variant label based on filename."""
    name = p.name.lower()
    if "eng_2" in name:     return "eng_2"
    if "eng" in name:       return "eng"
    if "viet" in name:      return "viet"
    if "thai" in name:      return "thai"
    if "translated" in name:return "translated"
    return "original"

def base_key(p: Path) -> str:
    """
    Group files by base model.
    Example filename:
      llm_labels_q524332_anthropic.claude-3-haiku-20240307-v1_0_modified_eng.tsv
    We keep the first 4 underscore-delimited parts: 
      llm_labels + q524332 + <model_name> + 0
    """
    parts = p.stem.split("_")
    return "_".join(parts[:4])

# --- Load files and group ---
files = sorted(ROOT.glob(PATTERN))
groups = defaultdict(list)
for f in files:
    groups[base_key(f)].append(f)

# Collect all results for a CSV
summary_rows = []

for base, fs in groups.items():
    originals = [f for f in fs if is_original(f)]
    if not originals:
        print(f"[SKIP] No original found for group: {base}")
        continue

    original_file = originals[0]
    print(f"\n=== Comparing files for base model: {base} ===")

    # Load original labels
    usecols = ["docid", "relevance"]
    df_orig = pd.read_csv(original_file, sep="\t", comment="#", usecols=usecols)

    for f in fs:
        if f == original_file:
            continue

        df_other = pd.read_csv(f, sep="\t", comment="#", usecols=usecols)

        merged = pd.merge(
            df_orig.rename(columns={"relevance": "relevance_orig"}),
            df_other.rename(columns={"relevance": "relevance_other"}),
            on="docid",
            how="inner"
        ).dropna(subset=["relevance_orig", "relevance_other"])

        if merged.empty:
            print(f"[SKIP] No overlapping labeled docids for {f.name}.")
            continue

        y1 = merged["relevance_orig"].astype(int)
        y2 = merged["relevance_other"].astype(int)

        mae = (y1 - y2).abs().mean()
        kappa = cohen_kappa_score(y1, y2)
        total_relevant_orig = (y1 == 1).sum()
        total_relevant_other = (y2 == 1).sum()

        vlabel = variant_of(f)
        print(f"[{vlabel.upper():8}] MAE for {f.name} vs {original_file.name}: {mae:.3f}")
        print(f"[{vlabel.upper():8}] Cohen's kappa: {kappa:.3f}")
        print(f"[{vlabel.upper():8}] Relevant passages (original): {total_relevant_orig}")
        print(f"[{vlabel.upper():8}] Relevant passages (variant):  {total_relevant_other}\n")

        summary_rows.append({
            "base": base,
            "variant_file": f.name,
            "variant": vlabel,
            "original_file": original_file.name,
            "n_overlap": len(merged),
            "MAE": round(mae, 3),
            "Kappa": round(kappa, 3),
            "Relevant_Orig": int(total_relevant_orig),
            "Relevant_Variant": int(total_relevant_other)
        })

# --- Write consolidated CSV ---
if summary_rows:
    out_csv = ROOT / "comparison_summary_with_eng2.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nSaved summary: {out_csv}")
else:
    print("\nNo comparisons produced a result.")
