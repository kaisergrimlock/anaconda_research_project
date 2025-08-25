import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# Find all label files for this query
label_files = sorted(Path("outputs").glob("llm_labels_q524332_*.tsv"))

# Identify the original file (no viet, eng, thai, translated)
def is_original(f):
    name = f.name
    return not any(x in name for x in ["viet", "eng", "thai", "translated"])

original_files = [f for f in label_files if is_original(f)]
if not original_files:
    raise ValueError("No original label file found.")
original_file = original_files[0]

# Read the original file
df_orig = pd.read_csv(original_file, sep="\t", comment="#")

# Compare each other file to the original
for f in label_files:
    if f == original_file:
        continue
    df = pd.read_csv(f, sep="\t", comment="#")
    merged = pd.merge(df_orig, df, on="docid", suffixes=("_orig", "_other"))
    mae = (merged["relevance_orig"] - merged["relevance_other"]).abs().mean()
    kappa = cohen_kappa_score(merged["relevance_orig"], merged["relevance_other"])
    print(f"MAE for {f.name} vs {original_file.name}: {mae:.3f}")
    print(f"Cohen's kappa for {f.name} vs {original_file.name}: {kappa:.3f}\n")