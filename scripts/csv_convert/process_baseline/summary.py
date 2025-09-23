#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# ---- CONFIG ----
# Point this to either a single file OR a directory containing many CSVs
INPUT_PATH = Path("outputs/trec_dl/retrieved/all_topics")  # e.g. a folder
# INPUT_PATH = Path("outputs/trec_dl/retrieved/all_topics/all_topics_trecdl_2019_part1.csv")
WRITE_SUMMARY_CSV = True
SUMMARY_OUT = Path("outputs/csv/relevance_summary.csv")
# ---------------

def read_all_csvs(path: Path) -> pd.DataFrame:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs found under: {path}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
    if not dfs:
        raise SystemExit("No readable CSVs.")
    return pd.concat(dfs, ignore_index=True)

def main():
    df = read_all_csvs(INPUT_PATH)

    if "relevance" not in df.columns:
        raise SystemExit("Input must have a 'relevance' column.")

    # Coerce to numeric, drop non-numeric/nulls
    rel = pd.to_numeric(df["relevance"], errors="coerce").dropna().astype(int)

    # Count per label; ensure all 3/2/1/0 appear even if zero
    order = [3, 2, 1, 0]
    counts = rel.value_counts().reindex(order, fill_value=0)
    total = int(counts.sum())

    # Percentages
    pct = (counts / total * 100).round(2) if total > 0 else counts.astype(float)

    # Pretty summary table
    summary = (
        pd.DataFrame({"label": order, "count": counts.values, "percent": pct.values})
        .assign(percent=lambda d: d["percent"].map(lambda x: f"{x:.2f}%"))
    )

    print("\nRelevance distribution (across all rows):")
    print(summary.to_string(index=False))
    print(f"\nTotal judged rows: {total}")

    if WRITE_SUMMARY_CSV:
        SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
        # Save numeric percentages (0–100) rather than formatted strings
        summary_csv = pd.DataFrame({
            "label": order,
            "count": counts.values,
            "percent": (counts / total * 100) if total > 0 else counts.astype(float)
        })
        summary_csv.to_csv(SUMMARY_OUT, index=False)
        print(f"\n[Saved] {SUMMARY_OUT.resolve()}")

if __name__ == "__main__":
    main()
