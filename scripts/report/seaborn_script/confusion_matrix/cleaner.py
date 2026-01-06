#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# =========================
# Config (edit these)
# =========================
IN_CSV: Path = Path(
    r"outputs/llm_label/trec_dl_2021/gpt-oss-20b/gpt-oss-20b_trecdl_2021_raw_labels.csv"
)

OUT_CSV: Path = Path(
    r"outputs/llm_label/trec_dl_2021/gpt-oss-20b/gpt-oss-20b_trecdl_2021_raw_labels_clean.csv"
)

# Set to None to clean ALL columns
COLUMNS_TO_CLEAN: Optional[list[str]] = ["relevance", "llm_relevance"]

# If True: converts "2.0" -> "2" after cleaning (handy for label columns)
INTIFY_NUMERIC_LABELS: bool = True

# Treat these tokens (case-insensitive) as missing -> ""
NA_LIKE = {
    "nan", "none", "null", "na", "n/a", "missing", "error",
    "nil", "undef", "undefined",
}

# =========================
# Cleaning helpers
# =========================
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d]")
BOM_RE = re.compile(r"^\ufeff")
NBSP_RE = re.compile(r"\xa0")


def _clean_text_cell(x: object) -> str:
    """
    Normalize whitespace + remove invisible chars.
    Returns a string (possibly empty).
    """
    if x is None:
        return ""

    s = str(x)

    # Remove BOM if present
    s = BOM_RE.sub("", s)

    # Replace NBSP with normal space
    s = NBSP_RE.sub(" ", s)

    # Remove zero-width characters
    s = ZERO_WIDTH_RE.sub("", s)

    # Strip surrounding whitespace
    s = s.strip()

    # Standardize NA-like tokens -> empty
    if s.lower() in NA_LIKE:
        return ""

    return s


def _clean_series(s: pd.Series) -> pd.Series:
    # keep_default_na=False will preserve blanks as "", but we still guard for NaN
    return s.fillna("").astype(str).map(_clean_text_cell)


def _intify_label_series(s: pd.Series) -> pd.Series:
    """
    If a cell parses as a float like "2.0", convert to "2".
    If it parses as "2", keep "2".
    Otherwise leave as-is.
    """
    def fix_one(x: str) -> str:
        if x == "":
            return ""
        try:
            v = float(x)
        except Exception:
            return x
        if v.is_integer():
            return str(int(v))
        return str(v)

    return s.map(fix_one)


def clean_csv(
    in_path: Path,
    out_path: Path,
    *,
    columns: Optional[Iterable[str]] = None,
    intify: bool = True,
) -> None:
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)

    if columns is None:
        target_cols = list(df.columns)
    else:
        target_cols = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"[WARN] These columns were not found and will be skipped: {missing}")

    for c in target_cols:
        df[c] = _clean_series(df[c])
        if intify:
            df[c] = _intify_label_series(df[c])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[DONE] Cleaned CSV written to: {out_path}")
    print(f"[INFO] Cleaned columns: {target_cols}")
    print(f"[INFO] INTIFY_NUMERIC_LABELS={intify}")


def main() -> None:
    clean_csv(
        IN_CSV,
        OUT_CSV,
        columns=COLUMNS_TO_CLEAN,
        intify=INTIFY_NUMERIC_LABELS,
    )


if __name__ == "__main__":
    main()
