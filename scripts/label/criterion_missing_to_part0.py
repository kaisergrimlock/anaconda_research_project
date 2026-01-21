#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
MODEL = "gpt-oss-20b"

# Set to a single criterion (e.g. "contextuality"). Leave empty to scan all criteria.
CRITERION = "contextuality"

# Language filter (required). Only these languages will be processed.
#LANGS: list[str] = ["sw"]
LANGS: list[str] = ["ar", "he", "th", "vi", "ru", "raw", "hi", "fr", "sw", "zh", "ga"]

# Output mode for part0 files
MODE = "replace"  # "replace" or "append"

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

CRITERION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / "criterion"
)


def parse_lang_criterion(path: Path) -> tuple[str, str] | None:
    """
    Parse:
      {MODEL}_trecdl_{YEAR}_{lang}_{criterion}_labels.csv
    """
    prefix = f"{MODEL}_trecdl_{TREC_DL_YEAR}_"
    if not path.name.startswith(prefix) or not path.name.endswith("_labels.csv"):
        return None

    stem = path.name[: -len("_labels.csv")]
    rest = stem[len(prefix) :]
    if "_" not in rest:
        return None

    lang, criterion = rest.rsplit("_", 1)
    return lang, criterion


def part0_path_for_lang(lang: str) -> Path:
    if lang == "raw":
        return (
            PROJECT_ROOT
            / "retrieved"
            / f"trec_dl_{TREC_DL_YEAR}"
            / "judged"
            / f"all_topics_trecdl_{TREC_DL_YEAR}_part0.csv"
        )
    return (
        PROJECT_ROOT
        / "retrieved"
        / f"trec_dl_{TREC_DL_YEAR}"
        / lang
        / f"all_topics_trecdl_{TREC_DL_YEAR}_part0.csv"
    )


def header_matches(path: Path, cols: list[str]) -> bool:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        existing = next(reader, None)
    return existing == cols


def expected_columns_for_lang(lang: str) -> list[str]:
    if lang == "raw":
        return ["qid", "query", "pid", "passage", "relevance"]
    return [
        "qid",
        "query",
        "pid",
        "passage",
        "relevance",
        f"query_{lang}",
        "passage_injected",
    ]


def ensure_expected_columns(df: pd.DataFrame, lang: str, expected_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    if "passage" in expected_cols and "passage" not in df.columns:
        judged_path = part0_path_for_lang("raw")
        if judged_path.exists():
            passage_df = pd.read_csv(
                judged_path,
                usecols=["qid", "pid", "passage"],
                dtype={"qid": "string", "pid": "string"},
            )
            df = df.merge(passage_df, on=["qid", "pid"], how="left")
        else:
            df["passage"] = ""
        if "passage" in df.columns:
            df["passage"] = df["passage"].fillna("")
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    return df[expected_cols]


def main() -> None:
    if not CRITERION_DIR.exists():
        print(f"[FATAL] Criterion dir not found: {CRITERION_DIR}")
        sys.exit(1)

    pattern = (
        f"{MODEL}_trecdl_{TREC_DL_YEAR}_*_{CRITERION}_labels.csv"
        if CRITERION
        else f"{MODEL}_trecdl_{TREC_DL_YEAR}_*_labels.csv"
    )

    if not LANGS:
        print("[FATAL] LANGS is empty. Set LANGS to the languages you want to process.")
        sys.exit(1)

    matched = list(CRITERION_DIR.glob(pattern))
    if not matched:
        print(f"[INFO] No files matched pattern: {pattern}")
        return

    total_written = 0
    for path in matched:
        parsed = parse_lang_criterion(path)
        if parsed is None:
            print(f"[WARN] Skipping unrecognized filename: {path.name}")
            continue
        lang, criterion = parsed
        if LANGS and lang not in LANGS:
            continue
        if CRITERION and criterion != CRITERION:
            continue

        df = pd.read_csv(path, dtype={"qid": "string", "pid": "string"})
        if criterion not in df.columns:
            print(f"[WARN] Column {criterion!r} missing in {path.name}, skipping.")
            continue

        scores = pd.to_numeric(df[criterion], errors="coerce")
        total_rows = int(len(df))
        valid_rows = int(scores.notna().sum())
        missing_df = df[scores.isna()].copy()
        if missing_df.empty:
            print(f"[OK] No missing values in {path.name} (total={total_rows}, valid={valid_rows})")
            continue

        # Drop criterion column so the part0 file matches input format for re-runs.
        missing_df = missing_df.drop(columns=[criterion], errors="ignore")
        expected_cols = expected_columns_for_lang(lang)
        missing_df = ensure_expected_columns(missing_df, lang, expected_cols)

        out_path = part0_path_for_lang(lang)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = True
        if MODE == "append" and out_path.exists():
            if not header_matches(out_path, missing_df.columns.tolist()):
                print(
                    f"[FATAL] Header mismatch when appending to {out_path}.\n"
                    f"  existing: {out_path}\n"
                    f"  new:      {missing_df.columns.tolist()}"
                )
                sys.exit(2)
            write_header = False

        if MODE == "replace":
            missing_df.to_csv(out_path, index=False, encoding="utf-8")
        else:
            missing_df.to_csv(out_path, index=False, encoding="utf-8", mode="a", header=write_header)

        print(
            f"[WRITE] {path.name} -> {out_path} "
            f"(missing={len(missing_df)}, total={total_rows}, valid={valid_rows})"
        )
        total_written += len(missing_df)

    print(f"[DONE] Wrote {total_written} missing rows to part0 files.")


if __name__ == "__main__":
    main()
