#!/usr/bin/env python3
from __future__ import annotations

import csv, sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# CSV merging utilities for LLM labeling outputs.
#
# Supports two modes:
#   - append  : blindly append rows from per-file label CSVs to the combined CSV.
#   - replace : for each key, replace up to N existing occurrences with the next
#               N newly labeled rows for that same key (FIFO), preserving order
#               and preserving duplicates that are not replaced. New keys that
#               do not already exist in the combined CSV are ignored.
#
# The row identity is defined by KEY_COLS. The combined file’s header must
# match header_out exactly.
# -----------------------------------------------------------------------------

# ===== identity for replace mode =====
# Keys that uniquely (for our purposes) identify a row. We intentionally allow
# duplicates in the combined file; "replace" only swaps as many occurrences as
# provided by the new rows (per key) and never deduplicates.
KEY_COLS: Tuple[str, str, str] = ("pid_qrels", "pid_resolved", "passage")


def _row_key_from_list(
    row: List[str],
    header: List[str],
    key_cols: Tuple[str, str, str],
) -> Tuple[str, str, str]:
    """
    Compute the tuple key for a CSV row given the header and the columns used
    as identity (key_cols).

    This relies on the header order to map column names to indices, then packs
    the row values for those indices into a tuple.

    Example:
        header = ["qid", "pid_qrels", "pid_resolved", "passage", "llm_relevance"]
        key_cols = ("pid_qrels", "pid_resolved", "passage")
        -> returns ("msmarco_...", "msmarco_...", "<passage text>")

    NOTE: If any key column is missing in header, header.index(...) will raise.
    """
    idx = [header.index(c) for c in key_cols]
    # type: ignore[return-value] is used because mypy cannot infer precise tuple size here.
    return tuple(row[i] for i in idx)  # type: ignore[return-value]


def _ensure_csv_with_header(path: Path, header: List[str]) -> None:
    """
    Ensure `path` exists with the exact `header`.
    - If the file does not exist, create it and write the header.
    - If it exists, validate that its header matches `header` exactly.

    Exits with code 4 on header mismatch to avoid corrupting downstream merges.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Create the file with the expected header.
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)
    else:
        # Validate header matches exactly.
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

    This mode never removes or overwrites existing rows.
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

                # Safety: do not mix files with different headers.
                if h != header_out:
                    print(f"[FATAL] Inconsistent header in {p.name}.\n  got: {h}\n  exp: {header_out}")
                    sys.exit(4)

                # Append all rows as-is.
                for row in r:
                    if row:
                        w.writerow(row)
                        appended += 1

    print(f"[MERGE] append | +{appended} -> {combined_out}")


def _merge_replace(per_file_labels: List[Path], combined_out: Path, header_out: List[str]) -> None:
    """
    Replace mode:
      - If combined file is missing, create it with the header only and exit.
      - Load the combined rows (after validating header). We keep the rows
        as a list to preserve order and multiplicity (duplicates).
      - Build a FIFO queue of replacement rows for each key from the per-file
        label CSVs (validating headers along the way).
      - Stream through the combined rows again and for each row:
          * If its key has a queued replacement, pop(0) and write the new row.
          * Otherwise write the original row unchanged.
      - Write to a temporary file and atomically replace the original.

    Guarantees:
      * Does not add new keys that do not already exist in combined_out.
      * Replaces at most N occurrences per key, where N is the number of new
        rows for that key across all per-file labels (FIFO).
      * Preserves ordering and leaves unmatched duplicates untouched.
    """
    # If combined file doesn't exist, we cannot "replace" anything. Create header only.
    if not combined_out.exists():
        _ensure_csv_with_header(combined_out, header_out)
        print("[MERGE] replace | combined missing; created header only (no keys to overwrite).")
        return

    # --- Load combined header and rows (preserve order + duplicates) ---
    with combined_out.open("r", encoding="utf-8", newline="") as fin:
        r_combined = csv.reader(fin)
        h_combined = next(r_combined, None)

        # Header consistency check.
        if h_combined != header_out:
            print(f"[FATAL] Inconsistent header in combined.\n  got: {h_combined}\n  exp: {header_out}")
            sys.exit(4)

        # Materialize rows into a list so we can single-pass rewrite with index order preserved.
        combined_rows_iter = list(r_combined)

    if not combined_rows_iter:
        # Nothing to overwrite.
        print("[MERGE] replace | combined empty; nothing to overwrite.")
        return

    # --- Build a FIFO replacement queue per key (preserve multiplicity) ---
    # repl_map[key] = [row1, row2, ...] in the order they appear across per-file CSVs.
    repl_map: Dict[Tuple[str, str, str], List[List[str]]] = {}
    staged = 0  # number of new rows examined (for logging)

    for p in per_file_labels:
        if not p.exists():
            print(f"[WARN] Missing per-file labels for merge: {p}")
            continue

        with p.open("r", encoding="utf-8", newline="") as fin:
            r = csv.reader(fin)
            h = next(r, None)

            # Header consistency check per file.
            if h != header_out:
                print(f"[FATAL] Inconsistent header in {p.name}.\n  got: {h}\n  exp: {header_out}")
                sys.exit(4)

            # Collect new rows into per-key queues.
            for row in r:
                if not row:
                    continue
                k = _row_key_from_list(row, header_out, KEY_COLS)
                # Append to the queue for this key (FIFO behavior).
                repl_map.setdefault(k, []).append(row)
                staged += 1

    # (Optional bookkeeping) Count occurrences per key in the combined file.
    # Not strictly needed for correctness, but useful for reasoning/logging.
    occurrence_counts: Dict[Tuple[str, str, str], int] = {}
    for row in combined_rows_iter:
        k = _row_key_from_list(row, header_out, KEY_COLS)
        occurrence_counts[k] = occurrence_counts.get(k, 0) + 1

    # --- Rewrite: replace up to len(repl_map[key]) occurrences for each key, FIFO ---
    replaced = 0
    tmp = combined_out.with_suffix(combined_out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        # Write header first.
        w.writerow(header_out)

        # Walk original rows in order; replace when a queued row is available.
        for row in combined_rows_iter:
            k = _row_key_from_list(row, header_out, KEY_COLS)
            queue = repl_map.get(k)

            if queue and len(queue) > 0:
                # Replacement available: pop the next one for this key (FIFO).
                new_row = queue.pop(0)
                w.writerow(new_row)
                replaced += 1
            else:
                # No replacement staged (or queue exhausted): keep original row.
                w.writerow(row)

    # Atomic replace to avoid partial writes on crash.
    tmp.replace(combined_out)
    print(f"[MERGE] replace | replaced={replaced} ignored_new={staged - replaced} -> {combined_out}")


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
    Public entry point used by the Bedrock runner.

    Args:
        per_file_labels: paths (as strings) to per-file label CSVs produced by the labeler.
        header_out     : exact header expected in both per-file and combined CSVs.
        model_short    : short model name (used for output directory naming).
        lang           : language tag, controls the combined filename.
        year           : TREC-DL year, used in the combined filename.
        mode           : "append" or "replace" (see top-of-file docs).

    Returns:
        Path to the combined CSV.

    Behavior:
        - Computes the combined CSV path to match your historical naming scheme.
        - Dispatches to append or replace merge strategy as requested.
        - Ensures header correctness and fails fast on mismatches.
    """
    # Compute combined output path in the same style as the original pipeline.
    out_dir = Path("outputs/llm_label") / model_short
    if lang == "raw":
        combined_out = out_dir / f"{model_short}_trec_dl_{year}_raw.csv"
    else:
        combined_out = out_dir / f"{model_short}_trec_dl_{year}_{lang}.csv"

    # Normalize input file list to Paths.
    paths = [Path(p) for p in per_file_labels]

    # Dispatch to merge strategy.
    if mode == "append":
        _merge_append(paths, combined_out, header_out)
    elif mode == "replace":
        _merge_replace(paths, combined_out, header_out)
    else:
        print(f"[FATAL] Unknown mode: {mode}")
        sys.exit(3)

    return combined_out
