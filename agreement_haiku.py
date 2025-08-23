import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Paths to your label files
files = [
    "outputs/llm_labels_q524332_anthropic.claude-3-haiku-20240307-v1_0.tsv",
    "outputs/llm_labels_q524332_anthropic.claude-3-haiku-20240307-v1_0_viet.tsv",
    "outputs/llm_labels_q524332_anthropic.claude-3-haiku-20240307-v1_0_eng.tsv"
]

# Read all files into DataFrames
dfs = [pd.read_csv(f, sep="\t", comment="#") for f in files]

# Merge on docid
merged = dfs[0][["docid"]].copy()
for i, df in enumerate(dfs):
    merged[f"label_{i+1}"] = df["relevance"]

# Calculate pairwise agreement and Cohen's kappa
pairs = [(0, 1), (0, 2), (1, 2)]
for i, j in pairs:
    labels1 = merged[f"label_{i+1}"]
    labels2 = merged[f"label_{j+1}"]
    agreement = (labels1 == labels2).mean()
    kappa = cohen_kappa_score(labels1, labels2)
    print(f"Agreement between label_{i+1} and label_{j+1}: {agreement:.3f}")
    print(f"Cohen's kappa between label_{i+1} and label_{j+1}: {kappa:.3f}\n")