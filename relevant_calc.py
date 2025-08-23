import pandas as pd
from pathlib import Path

# Find all relevant label files
label_files = sorted(Path("outputs").glob("llm_labels_q524332_*.tsv"))

for f in label_files:
    df = pd.read_csv(f, sep="\t", comment="#")
    relevant_count = (df["relevance"] == 1).sum()
    print(f"{f.name}: {relevant_count} relevant passages")