#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd

# ============= Config =============
TREC_DL_YEAR = "2023"

RAW_IN  = Path("outputs/llm_label/gpt-oss-20b/gpt-oss-20b_trec_dl_2023_raw.csv")
RAW_OUT = RAW_IN.with_name("gpt-oss-20b_trec_dl_2023_raw_with_ids.csv")

JUDGED_DIR            = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged")
JUDGED_DIR_FALLBACK   = Path(f"retrieved/tred_dl_{TREC_DL_YEAR}/judged")  # typo fallback

PROGRESS_EVERY_ROWS   = 1000  # print rewrite progress every N rows
# ==================================


# ---------- CSV field size fix ----------
def _bump_field_limit() -> None:
    """
    Raise Python csv module's max field size limit to allow very long cells.
    Tries progressively smaller limits if sys.maxsize overflows on this platform.
    """
    limit = getattr(sys, "maxsize", 2_000_000_000)
    while limit >= 131_072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2
    # Fallback if somehow we never set it above default
    csv.field_size_limit(1_000_000)

_bump_field_limit()
# ---------------------------------------


def norm(s: str) -> str:
    """Normalize for matching: lower, strip, collapse whitespace/newlines."""
    if s is None:
        return ""
    return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip().lower()


def pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols:
            return cols[key]
    return None


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Robust CSV loader: after bumping field size, use python engine,
    keep everything as strings, and skip malformed lines instead of dying.
    """
    return pd.read_csv(
        path,
        engine="python",
        dtype=str,
        on_bad_lines="skip",
    )


def build_index(judged_dir: Path) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """
    Build {(norm_query, norm_passage): (qid, pid)} from all judged CSVs.
    Shows per-file progress.
    """
    if not judged_dir.exists():
        if JUDGED_DIR_FALLBACK.exists():
            print(f"[WARN] {judged_dir} not found. Using fallback: {JUDGED_DIR_FALLBACK}")
            judged_dir = JUDGED_DIR_FALLBACK
        else:
            raise FileNotFoundError(f"No judged directory found: {judged_dir}")

    files = sorted(judged_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {judged_dir}")

    index: Dict[Tuple[str, str], Tuple[str, str]] = {}
    collisions = 0
    added = 0

    t0 = time.time()
    for fi, fp in enumerate(files, start=1):
        df = read_csv_robust(fp)

        qid_col = pick(df, ["qid", "topic"])  # numeric topic id
        qry_col = pick(df, ["query", "question", "topic_text"])
        psg_col = pick(df, ["passage", "text", "context", "document"])
        pid_col = pick(df, ["pid_resolved", "pid_qrels", "pid", "docid", "doc_id"])

        if not qry_col or not psg_col:
            print(f"[WARN] Skipping {fp.name} (missing query/passage).")
            continue

        # iterate efficiently
        local_added = 0
        for row in df.itertuples(index=False):
            rowd = row._asdict() if hasattr(row, "_asdict") else row._asdict()
            q_text = norm(rowd.get(qry_col, ""))
            p_text = norm(rowd.get(psg_col, ""))
            if not q_text or not p_text:
                continue
            qid = (rowd.get(qid_col, "") or "").strip() if qid_col else ""
            pid = (rowd.get(pid_col, "") or "").strip() if pid_col else ""
            key = (q_text, p_text)
            if key in index:
                if index[key] != (qid, pid):
                    collisions += 1
            else:
                index[key] = (qid, pid)
                added += 1
                local_added += 1

        elapsed = time.time() - t0
        print(f"[INDEX] ({fi}/{len(files)}) {fp.name:40}  +{local_added:6}  total={added:8}  collisions={collisions:5}  t={elapsed:6.1f}s")

    print(f"[INDEX] Done. Entries={added:,}  Collisions={collisions}  Files={len(files)}  Time={time.time()-t0:.1f}s")
    return index


def main():
    if not RAW_IN.exists():
        print(f"[ERROR] Input raw file not found: {RAW_IN}")
        sys.exit(1)

    idx = build_index(JUDGED_DIR)

    raw_df = read_csv_robust(RAW_IN)

    qry_col = pick(raw_df, ["query", "question", "topic_text"])
    psg_col = pick(raw_df, ["passage", "text", "context", "document"])
    rel_col = pick(raw_df, ["relevance", "label", "llm", "o"])

    if not qry_col or not psg_col:
        raise KeyError(f"Raw file must contain query & passage columns. Found: {list(raw_df.columns)}")

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    out_cols = ["qid", "query", "pid", "passage"]
    if rel_col:
        out_cols.append("relevance")

    total = len(raw_df)
    matched = 0
    missing = 0
    t0 = time.time()

    with RAW_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(out_cols)

        for i, row in enumerate(raw_df.itertuples(index=False), start=1):
            rowd = row._asdict()
            q_raw = rowd.get(qry_col, "") or ""
            p_raw = rowd.get(psg_col, "") or ""

            key = (norm(q_raw), norm(p_raw))
            qid, pid = idx.get(key, ("", ""))

            if qid or pid:
                matched += 1
            else:
                missing += 1

            out_row = [qid, str(q_raw).strip(), pid, str(p_raw).strip()]
            if rel_col:
                out_row.append(rowd.get(rel_col, ""))

            w.writerow(out_row)

            if i % PROGRESS_EVERY_ROWS == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"[REWRITE] {i:7}/{total:7}  matched={matched:7}  missing={missing:7}  rate={rate:6.1f} r/s  t={elapsed:6.1f}s", end="\r")

    # final newline after carriage-return progress
    print()
    print(f"[DONE] Wrote: {RAW_OUT}")
    hit_rate = (matched / total * 100.0) if total else 0.0
    print(f"[STATS] total={total:,}  matched={matched:,}  missing={missing:,}  hit-rate={hit_rate:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
