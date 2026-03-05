#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# =========================
# Config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
RETRIEVED_ROOT = PROJECT_ROOT / "retrieved"
OUT_DIR = THIS_FILE.parent / "judged_query_lengths"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QID_CANDIDATES = ["qid", "query_id", "q_id"]
QUERY_CANDIDATES = ["query", "query_text", "question"]


def detect_columns(path: Path) -> tuple[str | None, str | None]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    qid_col = next((c for c in QID_CANDIDATES if c in header), None)
    query_col = next((c for c in QUERY_CANDIDATES if c in header), None)
    return qid_col, query_col


def year_from_path(path: Path) -> str | None:
    # retrieved/trec_dl_2022/judged/...
    parts = path.parts
    for p in parts:
        if p.startswith("trec_dl_"):
            return p.replace("trec_dl_", "")
    return None


def main() -> None:
    if not RETRIEVED_ROOT.exists():
        print(f"[FATAL] retrieved root not found: {RETRIEVED_ROOT}")
        sys.exit(1)

    judged_dirs = list(RETRIEVED_ROOT.glob("trec_dl_*/*/judged"))
    if not judged_dirs:
        judged_dirs = list(RETRIEVED_ROOT.glob("trec_dl_*/judged"))

    if not judged_dirs:
        print(f"[INFO] No judged directories found under: {RETRIEVED_ROOT}")
        return

    year_to_frames: dict[str, list[pd.DataFrame]] = {}

    for judged_dir in judged_dirs:
        if not judged_dir.is_dir():
            continue
        year = year_from_path(judged_dir)
        if not year:
            continue
        csv_files = list(judged_dir.glob("*.csv"))
        if not csv_files:
            continue

        for path in csv_files:
            qid_col, query_col = detect_columns(path)
            if not qid_col or not query_col:
                print(f"[WARN] Missing qid/query cols in {path}")
                continue
            df = pd.read_csv(path, usecols=[qid_col, query_col])
            df = df.rename(columns={qid_col: "qid", query_col: "query"})
            df = df.dropna(subset=["qid", "query"])
            year_to_frames.setdefault(year, []).append(df)

    if not year_to_frames:
        print("[INFO] No judged CSVs with qid/query columns found.")
        return

    for year, frames in sorted(year_to_frames.items()):
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["qid", "query"])
        merged["query"] = merged["query"].astype(str)
        merged["word_count"] = merged["query"].str.strip().str.split().map(len)
        out_path = OUT_DIR / f"judged_query_lengths_{year}.csv"
        merged[["qid", "query", "word_count"]].to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] Wrote {len(merged)} rows: {out_path}")


if __name__ == "__main__":
    main()
