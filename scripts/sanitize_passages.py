#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
MODEL = "gpt-oss-20b"
YEAR = "2021"
LANGUAGE = None  # e.g. "arcwb_instruct", "vi", etc. Use None to process all.

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

RETRIEVED_DIR = PROJECT_ROOT / "retrieved" / f"trec_dl_{YEAR}"
SANITATION_DIR = PROJECT_ROOT / "outputs" / "sanitation_checker" / f"trec_dl_{YEAR}" / MODEL

SUPPORTED_TEXT_SUFFIXES = {".csv", ".tsv", ".jsonl"}
PASSAGE_COLUMN_CANDIDATES = ["passage", "contents", "content", "text", "passage_injected"]


# =============================================================================
# Helpers
# =============================================================================
def extract_lang_from_sanitation_filename(filename: str) -> str | None:
    """
    Example:
        gpt-oss-20b_trecdl_2022_arcwb_instruct_labels.csv -> arcwb_instruct
    """
    match = re.search(r"trecdl_202\d_(.*?)_labels\.csv$", filename)
    return match.group(1) if match else None


def remove_detected_injection(passage: str, detected_injection: str) -> str:
    """
    Remove the detected injection text from the passage.

    This uses literal removal, not regex pattern matching, so the detected
    injection is escaped first.
    """
    if pd.isna(passage):
        return ""

    passage = str(passage)

    if pd.isna(detected_injection):
        return passage

    detected_injection = str(detected_injection).strip()
    if not detected_injection:
        return passage

    cleaned = re.sub(re.escape(detected_injection), "", passage)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def detect_passage_column(df: pd.DataFrame) -> str | None:
    for col in PASSAGE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def normalize_key(qid: object, pid: object) -> tuple[str, str]:
    return str(qid).strip(), str(pid).strip()


# =============================================================================
# Load sanitation lookup
# =============================================================================
def load_sanitation_lookup(
    sanitation_dir: Path,
    language: str | None = None,
) -> dict[str, dict[tuple[str, str], str]]:
    """
    Build lookup:
        {
            "arcwb_instruct": {
                ("qid", "pid"): "detected injection text",
                ...
            },
            ...
        }

    Only rows with has_prompt_injection == Yes are included.
    """
    if not sanitation_dir.exists():
        raise FileNotFoundError(f"Sanitation folder not found: {sanitation_dir}")

    csv_files = sorted(sanitation_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No sanitation CSVs found in: {sanitation_dir}")

    lookup: dict[str, dict[tuple[str, str], str]] = {}

    for csv_path in csv_files:
        lang = extract_lang_from_sanitation_filename(csv_path.name)
        if lang is None:
            continue

        if language is not None and lang != language:
            continue

        df = pd.read_csv(csv_path)

        required_columns = {"qid", "pid", "has_prompt_injection", "detected_injection"}
        missing = required_columns - set(df.columns)
        if missing:
            print(f"Skipping {csv_path.name}: missing columns {sorted(missing)}")
            continue

        mask = (
            df["has_prompt_injection"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        )

        df_yes = df.loc[mask, ["qid", "pid", "detected_injection"]].copy()

        lang_lookup: dict[tuple[str, str], str] = {}
        for _, row in df_yes.iterrows():
            key = normalize_key(row["qid"], row["pid"])
            detected = "" if pd.isna(row["detected_injection"]) else str(row["detected_injection"])
            lang_lookup[key] = detected

        lookup[lang] = lang_lookup
        print(f"Loaded sanitation map for {lang}: {len(lang_lookup)} injected rows")

    return lookup


# =============================================================================
# Sanitization logic
# =============================================================================
def sanitize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Add a sanitized column by matching (qid, pid) against the sanitation lookup.

    This function expects the dataframe to already contain:
        - qid
        - pid
        - one recognized passage column
        - __detected_injection
    """
    if "qid" not in df.columns or "pid" not in df.columns:
        raise KeyError("Input file must contain 'qid' and 'pid' columns.")

    if "__detected_injection" not in df.columns:
        raise KeyError("Internal column '__detected_injection' is missing.")

    passage_col = detect_passage_column(df)
    if passage_col is None:
        raise KeyError(
            f"No recognized passage column found. Tried: {PASSAGE_COLUMN_CANDIDATES}"
        )

    df = df.copy()
    df["sanitized"] = df[passage_col]

    mask = df["__detected_injection"].notna() & (
        df["__detected_injection"].astype(str).str.strip() != ""
    )

    changed = 0
    for idx in df.index[mask]:
        original = df.at[idx, passage_col]
        detected = df.at[idx, "__detected_injection"]
        cleaned = remove_detected_injection(original, detected)

        if str(cleaned) != str(original):
            changed += 1

        df.at[idx, "sanitized"] = cleaned

    df = df.drop(columns=["__detected_injection"])
    return df, changed


def attach_detected_injection_lookup(
    df: pd.DataFrame,
    lang_lookup: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """
    Add internal column __detected_injection by matching (qid, pid).
    """
    if "qid" not in df.columns or "pid" not in df.columns:
        raise KeyError("Input file must contain 'qid' and 'pid' columns.")

    df = df.copy()
    df["__key"] = [normalize_key(qid, pid) for qid, pid in zip(df["qid"], df["pid"])]
    df["__detected_injection"] = df["__key"].map(lang_lookup)
    df = df.drop(columns=["__key"])
    return df


# =============================================================================
# File processors
# =============================================================================
def sanitize_csv_file(file_path: Path, lang_lookup: dict[tuple[str, str], str], output_path: Path) -> int:
    df = pd.read_csv(file_path)
    df = attach_detected_injection_lookup(df, lang_lookup)
    cleaned_df, changed = sanitize_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    return changed


def sanitize_tsv_file(file_path: Path, lang_lookup: dict[tuple[str, str], str], output_path: Path) -> int:
    df = pd.read_csv(file_path, sep="\t")
    df = attach_detected_injection_lookup(df, lang_lookup)
    cleaned_df, changed = sanitize_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, sep="\t", index=False)
    return changed


def sanitize_jsonl_file(file_path: Path, lang_lookup: dict[tuple[str, str], str], output_path: Path) -> int:
    rows: list[dict] = []
    changed = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if "qid" not in obj or "pid" not in obj:
                rows.append(obj)
                continue

            passage_col = next((c for c in PASSAGE_COLUMN_CANDIDATES if c in obj), None)
            if passage_col is None:
                rows.append(obj)
                continue

            key = normalize_key(obj["qid"], obj["pid"])
            detected = lang_lookup.get(key)

            obj["sanitized"] = obj[passage_col]

            if detected is not None and str(detected).strip():
                cleaned = remove_detected_injection(obj[passage_col], detected)

                if str(cleaned) != str(obj[passage_col]):
                    changed += 1

                obj["sanitized"] = cleaned

            rows.append(obj)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return changed


def copy_file_unchanged(file_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, output_path)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    if not RETRIEVED_DIR.exists():
        print(f"Retrieved folder not found: {RETRIEVED_DIR}")
        return

    if not SANITATION_DIR.exists():
        print(f"Sanitation folder not found: {SANITATION_DIR}")
        return

    try:
        sanitation_lookup = load_sanitation_lookup(SANITATION_DIR, LANGUAGE)
    except FileNotFoundError as e:
        print(e)
        return

    if not sanitation_lookup:
        print("No sanitation mappings loaded.")
        return

    total_files_written = 0
    total_rows_changed = 0
    total_files_copied = 0

    for lang_dir in sorted(RETRIEVED_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue

        lang = lang_dir.name

        if LANGUAGE is not None and lang != LANGUAGE:
            continue

        if lang not in sanitation_lookup:
            print(f"Skipping {lang_dir.name}: no sanitation CSV found for this language")
            continue

        lang_lookup = sanitation_lookup[lang]
        output_lang_dir = lang_dir.parent / f"{lang_dir.name}_sanitized"

        print(f"\nProcessing language folder: {lang_dir.name}")
        print(f"Output folder: {output_lang_dir}")

        for file_path in sorted(lang_dir.rglob("*")):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(lang_dir)
            output_path = output_lang_dir / relative_path
            suffix = file_path.suffix.lower()

            try:
                if suffix == ".csv":
                    changed = sanitize_csv_file(file_path, lang_lookup, output_path)
                    total_rows_changed += changed
                    total_files_written += 1
                    print(f"  Saved {output_path} | cleaned rows: {changed}")

                elif suffix == ".tsv":
                    changed = sanitize_tsv_file(file_path, lang_lookup, output_path)
                    total_rows_changed += changed
                    total_files_written += 1
                    print(f"  Saved {output_path} | cleaned rows: {changed}")

                elif suffix == ".jsonl":
                    changed = sanitize_jsonl_file(file_path, lang_lookup, output_path)
                    total_rows_changed += changed
                    total_files_written += 1
                    print(f"  Saved {output_path} | cleaned rows: {changed}")

                else:
                    copy_file_unchanged(file_path, output_path)
                    total_files_copied += 1
                    print(f"  Copied {output_path}")

            except Exception as e:
                print(f"  Failed on {file_path}: {e}")

    print("\nDone.")
    print(f"Sanitized text files written: {total_files_written}")
    print(f"Non-text files copied unchanged: {total_files_copied}")
    print(f"Rows actually changed: {total_rows_changed}")


if __name__ == "__main__":
    main()