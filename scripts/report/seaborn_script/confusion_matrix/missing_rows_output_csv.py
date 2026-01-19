#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit
from helpers.output_writer import write_df

# -------- Config --------
TREC_DL_YEAR = "2021"
MODEL = "gpt-oss-20b"
LANG = "hi_last"  # "raw","eng","vi","fr", etc.

# Expected input rows (baseline)
EXPECTED_FILE = (
    Path("retrieved")
    / f"trec_dl_{TREC_DL_YEAR}"
    / LANG
    / f"all_topics_trecdl_{TREC_DL_YEAR}_part0.csv"
)

# Output rows (LLM labels)
LLM_FILE = (
    Path("outputs/llm_label")
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{LANG}_labels.csv"
)

OUT_DIR = Path("figures") / TREC_DL_YEAR / MODEL / "confusion_matrix" / LANG
OUT_MISSING_ROWS = OUT_DIR / "missing_rows_from_output.csv"
OUT_EXTRA_ROWS = OUT_DIR / "extra_rows_in_output.csv"

QID_CANDIDATES = ["qid", "query_id"]
PID_CANDIDATES = ["pid", "passage_id", "docid", "pid_resolved", "pid_qrels"]


def _pick_col(df: pd.DataFrame, candidates: list[str], label: str, path: Path) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Expected a {label} column from {candidates} in {path}, "
        f"but got: {list(df.columns)}"
    )


def _key_frame(
    df: pd.DataFrame, qid_col: str, pid_col: str
) -> tuple[pd.DataFrame, int]:
    keys = df[[qid_col, pid_col]].copy()
    keys[qid_col] = keys[qid_col].astype("string").str.strip()
    keys[pid_col] = keys[pid_col].astype("string").str.strip()
    keys = keys.rename(columns={qid_col: "_qid", pid_col: "_pid"})

    valid_mask = (
        keys["_qid"].notna()
        & keys["_pid"].notna()
        & (keys["_qid"] != "")
        & (keys["_pid"] != "")
    )
    invalid_count = int((~valid_mask).sum())
    return keys[valid_mask], invalid_count


def _attach_keys(df: pd.DataFrame, qid_col: str, pid_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_qid"] = out[qid_col].astype("string").str.strip()
    out["_pid"] = out[pid_col].astype("string").str.strip()
    return out


def main() -> None:
    bump_field_limit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_df = pd.read_csv(EXPECTED_FILE)
    output_df = pd.read_csv(LLM_FILE)

    exp_qid = _pick_col(expected_df, QID_CANDIDATES, "qid", EXPECTED_FILE)
    exp_pid = _pick_col(expected_df, PID_CANDIDATES, "pid", EXPECTED_FILE)
    out_qid = _pick_col(output_df, QID_CANDIDATES, "qid", LLM_FILE)
    out_pid = _pick_col(output_df, PID_CANDIDATES, "pid", LLM_FILE)

    exp_keys, exp_invalid = _key_frame(expected_df, exp_qid, exp_pid)
    out_keys, out_invalid = _key_frame(output_df, out_qid, out_pid)

    print(f"[EXPECTED] rows={len(expected_df):,} valid_keys={len(exp_keys):,} invalid_keys={exp_invalid:,}")
    print(f"[OUTPUT]   rows={len(output_df):,} valid_keys={len(out_keys):,} invalid_keys={out_invalid:,}")

    missing_keys = exp_keys.merge(out_keys, on=["_qid", "_pid"], how="left", indicator=True)
    missing_keys = missing_keys[missing_keys["_merge"] == "left_only"][["_qid", "_pid"]].drop_duplicates()

    extra_keys = out_keys.merge(exp_keys, on=["_qid", "_pid"], how="left", indicator=True)
    extra_keys = extra_keys[extra_keys["_merge"] == "left_only"][["_qid", "_pid"]].drop_duplicates()

    expected_with_keys = _attach_keys(expected_df, exp_qid, exp_pid)
    output_with_keys = _attach_keys(output_df, out_qid, out_pid)

    missing_rows = expected_with_keys.merge(missing_keys, on=["_qid", "_pid"], how="inner")
    extra_rows = output_with_keys.merge(extra_keys, on=["_qid", "_pid"], how="inner")

    missing_rows = missing_rows.drop(columns=["_qid", "_pid"], errors="ignore")
    extra_rows = extra_rows.drop(columns=["_qid", "_pid"], errors="ignore")

    write_df(missing_rows, OUT_MISSING_ROWS)
    write_df(extra_rows, OUT_EXTRA_ROWS)

    print(f"[MISSING] Expected rows missing in output: {len(missing_rows):,}")
    print(f"[EXTRA] Output rows not in expected: {len(extra_rows):,}")
    print(f"[WRITE] {OUT_MISSING_ROWS}")
    print(f"[WRITE] {OUT_EXTRA_ROWS}")


if __name__ == "__main__":
    main()
