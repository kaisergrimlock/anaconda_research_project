#!/usr/bin/env python3
from __future__ import annotations

import csv, sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# CSV merging utilities for LLM labeling outputs.
#
# Modes:
#   - append  : blindly append rows from per-file label CSVs to the combined CSV.
#   - replace : for each key, replace up to N existing occurrences with the next
#               N newly labeled rows for that same key (FIFO), preserving order
#               and preserving duplicates that are not replaced. New keys that
#               do not already exist in the combined CSV are ignored.
#
# Row identity (replace mode) is defined by KEY_COLS, which now includes:
#   * pid_qrels, pid_resolved, query, and
#     - passage_injected for injected languages
#     - passage for 'raw'
# The exact set is chosen dynamically in write_combined() based on lang.
# The combined file’s header must match header_out exactly.
# -----------------------------------------------------------------------------
RAW_HEADER: List[str] = [
    "qid", "query", "pid_qrels", "pid_resolved", "passage", "relevance", "llm_relevance"
]

def _row_key_from_list(
    row: List[str],
    header: List[str],
    key_cols: Tuple[str, ...],
) -> Tuple[str, ...]:
    """
    Compute the tuple key for a CSV row given the header and the columns used
    as identity (key_cols). Raises if any key column is missing.
    """
    idx = [header.index(c) for c in key_cols]
    return tuple(row[i] for i in idx)

def _ensure_csv_with_header(path: Path, header: List[str]) -> None:
    """
    Ensure `path` exists with the exact `header`.
    - If the file does not exist, create it and write the header.
    - If it exists, validate that its header matches `header` exactly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            h = next(r, None)
            if h != header:
                print(f"[FATAL] Header mismatch for {path}.\n  got: {h}\n  exp: {header}")
                sys.exit(4)

def _merge_append(per_file_labels: List[Path], combined_out: Path, header_out: List[str]) -> None:
    """
    Append mode:
      - Ensure combined file exists (with the correct header).
      - For each per-file CSV:
          * Validate header equality.
          * Append every non-empty row, blindly (duplicates allowed).
    """
    _ensure_csv_with_header(combined_out, header_out)
    appended = 0

    with combined_out.open("a", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)

        for p in per_file_labels:
            if not p.exists():
                print(f"[WARN] Missing per-file labels: {p}")
                continue

            with p.open("r", encoding="utf-8", newline="") as fin:
                r = csv.reader(fin)
                h = next(r, None)

                if h != header_out:
                    print(f"[FATAL] Inconsistent header in {p.name}.\n  got: {h}\n  exp: {header_out}")
                    sys.exit(4)

                for row in r:
                    if row:
                        w.writerow(row)
                        appended += 1

    print(f"[MERGE] append | +{appended} -> {combined_out}")

def _merge_replace(
    per_file_labels: List[Path],
    combined_out: Path,
    header_out: List[str],
    key_cols: Tuple[str, ...],
) -> None:
    """
    Replace mode with dynamic identity (key_cols).
    - Does not add new keys that do not already exist in combined_out.
    - Replaces at most N occurrences per key (FIFO), preserves order/duplicates.
    """
    if not combined_out.exists():
        _ensure_csv_with_header(combined_out, header_out)
        print("[MERGE] replace | combined missing; created header only (no keys to overwrite).")
        return

    with combined_out.open("r", encoding="utf-8", newline="") as fin:
        r_combined = csv.reader(fin)
        h_combined = next(r_combined, None)
        if h_combined != header_out:
            print(f"[FATAL] Inconsistent header in combined.\n  got: {h_combined}\n  exp: {header_out}")
            sys.exit(4)
        combined_rows_iter = list(r_combined)

    if not combined_rows_iter:
        print("[MERGE] replace | combined empty; nothing to overwrite.")
        return

    # Build FIFO replacement map per key.
    repl_map: Dict[Tuple[str, ...], List[List[str]]] = {}
    staged = 0

    for p in per_file_labels:
        if not p.exists():
            print(f"[WARN] Missing per-file labels for merge: {p}")
            continue

        with p.open("r", encoding="utf-8", newline="") as fin:
            r = csv.reader(fin)
            h = next(r, None)
            if h != header_out:
                print(f"[FATAL] Inconsistent header in {p.name}.\n  got: {h}\n  exp: {header_out}")
                sys.exit(4)

            for row in r:
                if not row:
                    continue
                k = _row_key_from_list(row, header_out, key_cols)
                repl_map.setdefault(k, []).append(row)
                staged += 1

    # Optional bookkeeping (not required, but useful for reasoning/logging).
    # occurrence_counts: Dict[Tuple[str, ...], int] = {}
    # for row in combined_rows_iter:
    #     k = _row_key_from_list(row, header_out, key_cols)
    #     occurrence_counts[k] = occurrence_counts.get(k, 0) + 1

    replaced = 0
    tmp = combined_out.with_suffix(combined_out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header_out)
        for row in combined_rows_iter:
            k = _row_key_from_list(row, header_out, key_cols)
            queue = repl_map.get(k)
            if queue:
                new_row = queue.pop(0)
                w.writerow(new_row)
                replaced += 1
            else:
                w.writerow(row)

    tmp.replace(combined_out)
    print(f"[MERGE] replace | replaced={replaced} ignored_new={staged - replaced} -> {combined_out}")

def _merge_concat(
    per_file_labels: List[Path],
    combined_out: Path,
    header_out: List[str],
) -> None:
    """
    Concat mode: write every row from every per-file CSV into the combined file in order.
    Repeating qid rows are preserved. Validates headers match.
    """
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    with combined_out.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header_out)
        total = 0
        for p in per_file_labels:
            if not p.exists():
                print(f"[WARN] Missing per-file labels: {p}")
                continue
            with p.open("r", encoding="utf-8", newline="") as fin:
                r = csv.reader(fin)
                h = next(r, None)
                if h != header_out:
                    print(f"[FATAL] Inconsistent header in {p.name}.\n  got: {h}\n  exp: {header_out}")
                    sys.exit(4)
                for row in r:
                    if row:
                        w.writerow(row)
                        total += 1
    print(f"[MERGE] concat | +{total} -> {combined_out}")

def write_combined(
    *,
    per_file_labels: List[str],
    header_out: List[str],
    model_short: str,
    lang: str,
    year: str,
    mode: str = "replace",
) -> Path:
    """
    Public entry point.
    mode:
      - "concat": preserve all rows and order (keeps repeating qid)
      - "append": append rows (also preserves repeats)
      - "replace": existing replace semantics (FIFO replacement by key_cols)
    """
    out_dir = Path("outputs/llm_label") / model_short

    # --- Pick output path and identity columns
    if lang == "raw":
        combined_out = out_dir / f"{model_short}_trec_dl_{year}_raw.csv"
        # include qid in the identity so replace mode matches on qid + other fields
        key_cols: Tuple[str, ...] = ("qid", "pid_qrels", "pid_resolved", "query", "passage")
        expected_header = RAW_HEADER  # force exact raw schema
    else:
        combined_out = out_dir / f"{model_short}_trec_dl_{year}_{lang}.csv"
        # include qid for injected languages as well
        key_cols = ("qid", "pid_qrels", "pid_resolved", "query", "passage_injected")
        if "llm_relevance" in header_out:
            expected_header = [c for c in header_out if c != "llm_relevance"] + ["llm_relevance"]
        else:
            expected_header = header_out + ["llm_relevance"]

    paths = [Path(p) for p in per_file_labels]

    if mode == "concat":
        _merge_concat(paths, combined_out, expected_header)
    elif mode == "append":
        _merge_append(paths, combined_out, expected_header)
    elif mode == "replace":
        _merge_replace(paths, combined_out, expected_header, key_cols)
    else:
        print(f"[FATAL] Unknown mode: {mode}")
        sys.exit(3)

    return combined_out
