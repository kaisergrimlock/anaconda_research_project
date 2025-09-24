#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd

# ---- CONFIG ----
DEFAULT_CANDIDATES = [
    Path("outputs/trec_dl_llm_label/relevant"),
    Path("outputs/trec_dl/relevant"),
]
WRITE_SUMMARY_CSV = True
SUMMARY_OUT = Path("outputs/csv/relevance_summary.csv")
# ---------------

def pick_input_path() -> Path:
    # 1) CLI arg wins
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p
        raise SystemExit(f"Path not found: {p}")

    # 2) First candidate that contains any CSV
    for p in DEFAULT_CANDIDATES:
        csvs = list(p.rglob("*.csv")) if p.is_dir() else ([p] if p.is_file() else [])
        if csvs:
            return p

    raise SystemExit(f"No CSVs found under any of: {', '.join(str(p) for p in DEFAULT_CANDIDATES)}")

def read_all_csvs(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted(path.rglob("*.csv"))
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
    input_path = pick_input_path()
    df = read_all_csvs(input_path)

    if "relevance" not in df.columns:
        raise SystemExit("Input must have a 'relevance' column.")

    rel = pd.to_numeric(df["relevance"], errors="coerce").dropna().astype(int)
    order = [3, 2, 1, 0]
    counts = rel.value_counts().reindex(order, fill_value=0)
    total = int(counts.sum())
    pct = (counts / total * 100).round(2) if total > 0 else counts.astype(float)

    summary = (
        pd.DataFrame({"label": order, "count": counts.values, "percent": pct.values})
          .assign(percent=lambda d: d["percent"].map(lambda x: f"{x:.2f}%"))
    )

    print("\nRelevance distribution (across all rows):")
    print(summary.to_string(index=False))
    print(f"\nTotal judged rows: {total}")

    if WRITE_SUMMARY_CSV:
        SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "label": order,
            "count": counts.values,
            "percent": (counts / total * 100) if total > 0 else counts.astype(float)
        }).to_csv(SUMMARY_OUT, index=False)
        print(f"\n[Saved] {SUMMARY_OUT.resolve()}")

if __name__ == "__main__":
    main()
