import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

label_files = sorted(Path("outputs/llm_label").glob("llm_labels_q524332_*.tsv"))

def is_original(f):
    name = f.name
    return not any(x in name for x in ["viet", "eng", "thai", "translated"])

original_files = [f for f in label_files if is_original(f)]
if not original_files:
    raise ValueError("No original label file found.")
original_file = original_files[0]

# Read with explicit columns and dtypes; keep bad/missing as NaN for filtering
df_orig = pd.read_csv(original_file, sep="\t", comment="#", usecols=["docid","relevance"])

for f in label_files:
    if f == original_file:
        continue

    df = pd.read_csv(f, sep="\t", comment="#", usecols=["docid","relevance"])

    merged = pd.merge(
        df_orig.rename(columns={"relevance":"relevance_orig"}),
        df.rename(columns={"relevance":"relevance_other"}),
        on="docid",
        how="inner"
    )

    # Drop rows with missing labels and cast to int
    merged = merged.dropna(subset=["relevance_orig", "relevance_other"])
    if merged.empty:
        print(f"[SKIP] No overlapping labeled docids after cleaning for {f.name}.")
        continue

    y1 = merged["relevance_orig"].astype(int)
    y2 = merged["relevance_other"].astype(int)

    mae = (y1 - y2).abs().mean()
    kappa = cohen_kappa_score(y1, y2)

    print(f"MAE for {f.name} vs {original_file.name}: {mae:.3f}")
    print(f"Cohen's kappa for {f.name} vs {original_file.name}: {kappa:.3f}\n")
