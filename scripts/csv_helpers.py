# scripts/helpers.py
from __future__ import annotations

import sys
import csv
import re
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd


# ======================
# CSV / IO
# ======================
def bump_field_limit() -> None:
    """Allow very large CSV cells."""
    limit = getattr(sys, "maxsize", 2_000_000_000)
    while limit >= 131_072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2
    csv.field_size_limit(1_000_000)


def read_csv_smart(path: Path) -> pd.DataFrame:
    """Robust CSV reader, BOM-safe, skip bad lines."""
    return pd.read_csv(
        path,
        engine="python",
        dtype=str,
        on_bad_lines="skip",
        encoding="utf-8-sig",
    )


def _clean_key(k: Optional[str]) -> str:
    return (k or "").lstrip("\ufeff").strip()

def _inspect_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return [_clean_key(k) for k in (reader.fieldnames or [])]

def write_chunked_csv(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    chunk_size: int = 500,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    if n == 0:
        return []
    paths = []
    num_parts = (n + chunk_size - 1) // chunk_size
    pad = max(4, len(str(num_parts)))
    for i in range(num_parts):
        start, end = i * chunk_size, min((i + 1) * chunk_size, n)
        part = df.iloc[start:end]
        fp = out_dir / f"{base_name}_part_{(i + 1):0{pad}d}.csv"
        part.to_csv(fp, index=False, encoding="utf-8")
        paths.append(fp)
    return paths


def ensure_csv_with_header(path: Path, header: List[str]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)


# ======================
# Column pickers
# ======================
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.strip().lower(): c for c in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in cols:
            return cols[key]
    return None


def pick_qid_col(df: pd.DataFrame) -> Optional[str]:
    return pick_col(df, ["qid", "topic", "topic_id"])


def pick_pid_col(df: pd.DataFrame) -> str:
    c = pick_col(df, ["pid", "pid_resolved", "pid_qrels", "docid", "doc_id"])
    if not c:
        raise KeyError(f"No pid-like column in {list(df.columns)}")
    return c


def pick_label_col_generic(df: pd.DataFrame, candidates: List[str], who: str) -> str:
    c = pick_col(df, candidates)
    if not c:
        raise KeyError(f"{who}: none of {candidates} found in columns {list(df.columns)}")
    return c


# ======================
# Text / label utils
# ======================
def norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


_DIGIT_0_3 = re.compile(r"\b([0-3])\b")


def parse_label(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s in {"0", "1", "2", "3"}:
        return int(s)
    m = _DIGIT_0_3.search(s)
    return int(m.group(1)) if m else None


# ======================
# LANG-aware schema
# ======================
def base_trec_cols() -> List[str]:
    return ["qid", "query", "pid_qrels", "pid_resolved", "passage", "relevance"]


def extra_trec_cols_for_lang(lang: str) -> List[str]:
    if lang == "raw":
        return []
    return [f"query_{lang}", "passage_injected"]


def output_header_from_input(cols: List[str]) -> List[str]:
    cols = list(cols)
    if "llm_relevance" not in cols:
        cols.append("llm_relevance")
    return cols

def model_short_name(model_id: str) -> str:
    """
    'anthropic.claude-3-5-haiku-20241022-v1:0' -> 'claude-3-5'
    Rule: drop provider (before first '.'), strip version (after ':'), keep first 3 '-' parts.
    """
    s = model_id
    if "." in s:
        s = s.split(".", 1)[1]
    if ":" in s:
        s = s.split(":", 1)[0]
    parts = s.split("-")
    s = "-".join(parts[:3])
    return "".join(ch if (ch.isalnum() or ch == "-") else "-" for ch in s).strip("-")

def pick_query_for_lang(row: Dict[str, str], lang: str) -> str:
    if lang != "raw":
        q_lang = f"query_{lang}"
        if row.get(q_lang):
            return row[q_lang].strip()
    return (row.get("query") or "").strip()


def pick_passage_for_lang(row: Dict[str, str], lang: str) -> str:
    if lang != "raw" and row.get("passage_injected"):
        return row["passage_injected"].strip()
    return (row.get("passage") or "").strip()
