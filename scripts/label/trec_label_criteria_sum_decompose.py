#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import (
    bump_field_limit,
    ensure_csv_with_header,
    model_short_name,
    _inspect_header,
)

# ===== Config =====
LANG = "raw"          # language variant used in the criterion cache
START_PART = 1
END_PART = 6
TREC_DL_YEAR = "2022"
MODE = "append"       # "append" or "replace"

# Models are still used only to locate per-model criterion_cache files
MODELS = [
    "openai.gpt-oss-20b-1:0",
    # add more if you have criterion caches for them
]

# Where the criterion-composed part files and outputs live
OUTPUT_ROOT_BASE = PROJECT_ROOT / "outputs" / "llm_label" / f"trec_dl_{TREC_DL_YEAR}"

# Criterion columns we expect and use for summation
CRITERIA_COLS = ["exactness", "topicality", "coverage", "contextuality"]

# ===== helpers =====
bump_field_limit()  # allow huge csv fields

RowDict = Dict[str, Any]


def iter_part_files(
    start: int,
    end: int,
    part_dir: Path,
    pattern_template: str,
):
    """Yield each existing criterion-cache part file."""
    for n in range(start, end + 1):
        p = part_dir / pattern_template.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")


def read_rows_stream(path: Path):
    f = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(f, skipinitialspace=True)
    try:
        for row in reader:
            yield row
    finally:
        f.close()


def count_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        # total lines minus header
        return max(0, sum(1 for _ in f) - 1)


def parse_int_safe(v: Any) -> int:
    """Parse an integer grade, treating missing/invalid as 0."""
    if v is None:
        return 0
    s = str(v).strip()
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


def criterion_sum_to_relevance(total: int) -> int:
    """
    Map the sum of criterion grades to a relevance label:

        Sum 10–12 -> 3
        Sum  7–9  -> 2
        Sum  5–6  -> 1
        Sum  0–4  -> 0
    """
    if 10 <= total <= 12:
        return 3
    if 7 <= total <= 9:
        return 2
    if 5 <= total <= 6:
        return 1
    return 0


def write_combined_dynamic(
    per_file_labels: List[str],
    header_out: List[str],
    model_short: str,
    lang: str,
    year: str,
    mode: str,
    out_dir: Path,
) -> Path:
    """
    Combine per-part label CSVs into a single CSV, without enforcing specific
    qid/pid schema. Just header + rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / f"{model_short}_trecdl_{year}_{lang}_labels.csv"

    if mode == "replace" or not combined_path.exists():
        # Fresh file: write header and all rows
        with combined_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header_out)
            for p in per_file_labels:
                with Path(p).open("r", encoding="utf-8", newline="") as f_in:
                    reader = csv.reader(f_in)
                    in_header = next(reader, None)
                    if in_header is None:
                        continue
                    if in_header != header_out:
                        print(
                            f"[FATAL] Inconsistent header in {p}.\n"
                            f"  got: {in_header}\n"
                            f"  exp: {header_out}"
                        )
                        sys.exit(4)
                    for row in reader:
                        writer.writerow(row)
    else:
        # Append mode: check header once, then append rows
        with combined_path.open("r", encoding="utf-8", newline="") as f_ex:
            existing_header = next(csv.reader(f_ex), None)
        if existing_header != header_out:
            print(
                f"[FATAL] Combined file header mismatch.\n"
                f"  got: {existing_header}\n"
                f"  exp: {header_out}"
            )
            sys.exit(4)

        with combined_path.open("a", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out)
            for p in per_file_labels:
                with Path(p).open("r", encoding="utf-8", newline="") as f_in:
                    reader = csv.reader(f_in)
                    in_header = next(reader, None)
                    if in_header != header_out:
                        print(
                            f"[FATAL] Inconsistent header in {p}.\n"
                            f"  got: {in_header}\n"
                            f"  exp: {header_out}"
                        )
                        sys.exit(4)
                    for row in reader:
                        writer.writerow(row)

    return combined_path


def label_single_part_file(
    part_csv: Path,
    model_id: str,
    per_file_out_dir: Path,
) -> dict:
    """
    For a single criterion-cache part file, compute llm_relevance by summing
    the four criterion grades and mapping via criterion_sum_to_relevance.
    """
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    header_in = _inspect_header(part_csv)

    # We need at least the four criterion columns
    missing = [c for c in CRITERIA_COLS if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv.name}: missing criterion columns {missing}.")
        sys.exit(2)

    # Output header = input header + llm_relevance
    if "llm_relevance" in header_in:
        print(f"[WARN] {part_csv.name}: 'llm_relevance' already in header; will overwrite values.")
        header_out = header_in
    else:
        header_out = header_in + ["llm_relevance"]

    labels_path = per_file_out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    total_rows = count_data_rows(part_csv)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] LANG='{LANG}' | output columns = {header_out}")

    def append_row_csv(path: Path, header: List[str], new_row: List[str]) -> None:
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(header)
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(new_row)

    for idx, row in enumerate(read_rows_stream(part_csv), start=1):
        # Map all input columns
        row_out_map: Dict[str, str] = {k: (row.get(k, "") or "") for k in header_in}

        # best-effort pid_resolved (cache rows might not have it)
        pr = (row_out_map.get("pid_resolved", "") or "").strip()
        if not pr:
            pr = (
                row.get("docid", "")
                or row.get("pid", "")
                or row.get("pid_qrels", "")
                or row.get("passage_id", "")
                or ""
            ).strip()
            if pr and "pid_resolved" in header_in:
                row_out_map["pid_resolved"] = pr

        # Sum the four criterion grades
        total_crit = 0
        for col in CRITERIA_COLS:
            total_crit += parse_int_safe(row_out_map.get(col, ""))

        relevance_label = criterion_sum_to_relevance(total_crit)
        row_out_map["llm_relevance"] = str(relevance_label)

        row_values = [row_out_map.get(col, "") for col in header_out]
        append_row_csv(labels_path, header_out, row_values)

        if idx % 500 == 0 or idx == total_rows:
            print(
                f"[{part_csv.name}] [{idx}/{total_rows}] "
                f"criterion_sum={total_crit} -> rel={relevance_label}",
                end="\r",
                flush=True,
            )

    print()
    print(f"[{part_csv.name}] Wrote labels: {labels_path.name}")

    return {
        "part": part_csv.name,
        "rows": total_rows,
        "labels_csv": str(labels_path),
        "header_out": header_out,
    }


def run_for_model(model_id: str, mode: str):
    short = model_short_name(model_id)

    # Input criterion-cache chunk files for THIS model
    part_dir = (
        OUTPUT_ROOT_BASE
        / short
        / "criteria_composed"
        / LANG
    )
    part_pattern = f"{short}_trecdl_{TREC_DL_YEAR}_{LANG}_criterion_cache_part{{n:03d}}.csv"

    # Output dir for label CSVs
    model_out_dir = OUTPUT_ROOT_BASE / short / "temp"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    part_files = list(iter_part_files(START_PART, END_PART, part_dir, part_pattern))
    if not part_files:
        print(f"[INFO] No part files found in range in {part_dir}.")
        return

    print(
        f"\n--- Computing relevance from criteria for model: {model_id} "
        f"(short={short}, LANG={LANG}, mode={mode}) ---"
    )

    per_file_out_dir = model_out_dir / f"_tmp_{short}_{LANG}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for p in part_files:
        r = label_single_part_file(p, model_id, per_file_out_dir)
        results.append(r)

    if not results:
        print("[DONE] No outputs to merge.")
        return

    # verify consistent headers & collect per-file CSVs
    header_out_set = {tuple(r["header_out"]) for r in results}
    if len(header_out_set) != 1:
        print(f"[FATAL] Inconsistent output headers across parts: {header_out_set}")
        sys.exit(4)
    header_out = list(next(iter(header_out_set)))
    per_file_labels = [r["labels_csv"] for r in results]

    # write combined CSV
    combined_path = write_combined_dynamic(
        per_file_labels=per_file_labels,
        header_out=header_out,
        model_short=short,
        lang=LANG,
        year=TREC_DL_YEAR,
        mode=mode,
        out_dir=model_out_dir,
    )

    num_rows = sum(r["rows"] for r in results)

    print(f"[DONE] Model: {model_id} | Rows: {num_rows} | Combined: {combined_path}")

    # optional: clean up temp per-file outputs for this run
    try:
        import shutil
        shutil.rmtree(per_file_out_dir, ignore_errors=False)
        print(f"[CLEANUP] Removed temp folder: {per_file_out_dir}")
    except Exception as e:
        print(f"[WARN] Failed to remove temp folder {per_file_out_dir}: {e}")


def main():
    for model_id in MODELS:
        run_for_model(model_id, MODE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Top-level stop.")
