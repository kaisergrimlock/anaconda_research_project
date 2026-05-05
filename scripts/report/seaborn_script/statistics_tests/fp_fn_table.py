import os
import pandas as pd

# -----------------------------
# CONFIG (adjust to your setup)
# -----------------------------
BASE_DIR = "outputs/llm_label/trec_dl_2021"   # root folder
MODELS = ["gpt-oss-20b", "qwen3-32b"]        # update if needed
LANGS = ["eng", "vi", "ru", "th", "sw", "ga", "he", "zh", "fr", "hi", "ar"]

THRESHOLD = 2   # relevance threshold

OUTPUT_FILE = "fp_fn_table.csv"

# -----------------------------
# CORE FUNCTION
# -----------------------------
def compute_fp_fn(df, llm_col="llm_label", gold_col="relevance"):
    llm_bin = (df[llm_col] >= THRESHOLD).astype(int)
    gold_bin = (df[gold_col] >= THRESHOLD).astype(int)

    FP = ((llm_bin == 1) & (gold_bin == 0)).sum()
    FN = ((llm_bin == 0) & (gold_bin == 1)).sum()

    return FP, FN

# -----------------------------
# LOAD + PROCESS
# -----------------------------
results = []

for model in MODELS:
    model_path = os.path.join(BASE_DIR, model)

    for lang in LANGS:
        # Expected filename pattern
        file_name = f"{model}_trecdl_2021_{lang}.csv"
        file_path = os.path.join(model_path, file_name)

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Basic safety check
        if "llm_label" not in df.columns or "relevance" not in df.columns:
            print(f"Skipping (missing columns): {file_name}")
            continue

        FP, FN = compute_fp_fn(df)

        results.append({
            "language": lang,
            "model": model,
            "FP": FP,
            "FN": FN
        })

# -----------------------------
# SAVE TABLE
# -----------------------------
results_df = pd.DataFrame(results)

# Sort nicely
results_df = results_df.sort_values(by=["language", "model"])

results_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved to {OUTPUT_FILE}")