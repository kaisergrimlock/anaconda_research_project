import pandas as pd
from pathlib import Path

# Paths (adjust if needed)
all_topics_path = Path(r"d:\Work\Research_Project\anaconda_research_project\retrieved\trec_dl_2022\judged\all_topics_trecdl_2022_part1.csv")
gpt_raw_path   = Path(r"d:\Work\Research_Project\anaconda_research_project\outputs\llm_label\gpt-oss-20b\gpt-oss-20b_trec_dl_2022_raw.csv")
out_path       = gpt_raw_path.with_name(gpt_raw_path.stem + "_with_doc_meta.csv")

# Read files
df_topics = pd.read_csv(all_topics_path, dtype=str).fillna("")
df_gpt    = pd.read_csv(gpt_raw_path, dtype=str).fillna("")

# Normalize topic pid: prefer pid_qrels, else pid_resolved
def choose_pid(row):
    return row["pid_qrels"] if row["pid_qrels"].strip() else row["pid_resolved"]

df_topics["pid"] = df_topics.apply(choose_pid, axis=1)

# Keep relevant topic columns for merge
meta = df_topics[["pid", "qid", "query", "relevance"]].copy()
# If multiple labels per pid/qid exist keep first (or you can aggregate)
meta = meta.drop_duplicates(subset=["pid", "qid"], keep="first")

# Ensure gpt has 'pid' column; if not, try 'docid' fallback
if "pid" not in df_gpt.columns:
    if "docid" in df_gpt.columns:
        df_gpt["pid"] = df_gpt["docid"]
    else:
        raise SystemExit("gpt raw CSV missing both 'pid' and 'docid' columns")

# Merge on pid (left join to preserve all gpt rows)
merged = df_gpt.merge(meta, on="pid", how="left", suffixes=("", "_topic"))

# If docid empty in gpt but pid present, set docid := pid
if "docid" not in merged.columns:
    merged["docid"] = merged["pid"]
else:
    merged["docid"] = merged["docid"].fillna(merged["pid"])

# Report summary
matched = merged["qid"].notna().sum()
total = len(merged)
print(f"Total rows in GPT raw: {total}")
print(f"Rows matched to all_topics metadata: {matched}")
print(f"Unmatched rows: {total - matched}")

# Write merged CSV
merged.to_csv(out_path, index=False, encoding="utf-8")
print(f"Wrote merged CSV to: {out_path}")