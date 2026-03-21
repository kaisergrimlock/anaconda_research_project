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

def write_combined_dynamic(
    *,
    per_file_labels: list[str],
    header_out: list[str],
    model_short: str,
    lang: str,
    year: str,
    mode: str,
    out_dir: Path,
) -> Path:
    """
    Merge per-file label CSVs into one combined CSV.

    - mode="append": always append incoming rows.
    - mode="replace": replace existing rows by key (qid + pid-like), append unseen keys.

    FIX:
      In replace mode, do NOT overwrite an existing non-blank llm_relevance with a blank/NaN-like value.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / f"{model_short}_trecdl_{year}_{lang}_labels.csv"

    def same_columns(a: list[str], b: list[str]) -> bool:
        return len(a) == len(b) and set(a) == set(b)

    header_used = list(header_out)
    reorder_incoming: Optional[list[int]] = None
    incoming_to_used: Optional[list[tuple[int, int]]] = None
    missing_from_incoming_idx: list[int] = []

    if combined_path.exists():
        existing_header = _inspect_header(combined_path)
        if existing_header != header_out:
            if same_columns(existing_header, header_out):
                header_used = existing_header
                reorder_incoming = [header_out.index(c) for c in header_used]
                print(
                    "[WARN] Combined header order differs; will align incoming rows to existing header."
                )
            elif mode == "replace" and set(existing_header).issuperset(header_out):
                header_used = existing_header
                incoming_idx = {c: i for i, c in enumerate(header_out)}
                incoming_to_used = [
                    (header_used.index(c), incoming_idx[c]) for c in header_out if c in incoming_idx
                ]
                missing_from_incoming_idx = [
                    i for i, c in enumerate(header_used) if c not in incoming_idx
                ]
                print(
                    "[WARN] Combined header has extra columns; will align incoming rows and preserve existing values."
                )
            else:
                raise ValueError(
                    f"Combined file header mismatch.\n"
                    f"  got: {existing_header}\n"
                    f"  exp: {header_out}"
                )

    # -----------------------------
    # Key columns: qid + pid-like
    # -----------------------------
    def pick_pid_col(header: list[str]) -> str:
        candidates = ["pid", "pid_qrels", "pid_resolved", "docid", "passage_id", "doc_id"]
        for c in candidates:
            if c in header:
                return c
        raise ValueError(
            f"Cannot do replace by qid+pid: no pid column found in header. "
            f"Need one of {candidates}. Header={header}"
        )

    if "qid" not in header_out:
        raise ValueError(f"Cannot do replace by qid+pid: missing 'qid' in header_out={header_out}")

    pid_col = pick_pid_col(header_used)
    qid_i = header_used.index("qid")
    pid_i = header_used.index(pid_col)

    llm_idx = header_used.index("llm_relevance") if "llm_relevance" in header_used else None

    def norm_row_len_to(r: list[str], n: int) -> list[str]:
        if len(r) < n:
            return r + [""] * (n - len(r))
        if len(r) > n:
            return r[:n]
        return r

    def norm_row_len(r: list[str]) -> list[str]:
        return norm_row_len_to(r, len(header_used))

    def make_key(r: list[str]) -> str:
        r = norm_row_len(r)
        return f"{(r[qid_i] or '').strip()}|{(r[pid_i] or '').strip()}"

    def llm_val(r: list[str]) -> str:
        if llm_idx is None:
            return ""
        r = norm_row_len(r)
        return (r[llm_idx] or "").strip()

    def is_blank_or_nan_like(v: str) -> bool:
        s = (v or "").strip()
        if s == "":
            return True
        return s.lower() in {"nan", "none", "null"}

    def should_preserve_old_llm(old: str, new: str) -> bool:
        old_s = (old or "").strip()
        new_s = (new or "").strip()
        return (old_s != "") and is_blank_or_nan_like(new_s)

    # -----------------------------------
    # Load incoming rows as key -> row
    # -----------------------------------
    incoming: dict[str, list[str]] = {}

    for p in per_file_labels:
        pth = Path(p)
        with pth.open("r", encoding="utf-8", newline="") as f_in:
            reader = csv.reader(f_in)
            in_header = next(reader, None)
            if in_header is None:
                continue

            if list(in_header) != list(header_out):
                raise ValueError(
                    f"Inconsistent header in {pth}.\n"
                    f"  got: {in_header}\n"
                    f"  exp: {header_out}"
                )

            for r in reader:
                r = norm_row_len_to(r, len(header_out))
                if reorder_incoming:
                    r = [r[i] for i in reorder_incoming]
                if incoming_to_used:
                    aligned = [""] * len(header_used)
                    for used_i, in_i in incoming_to_used:
                        if in_i < len(r):
                            aligned[used_i] = r[in_i]
                    r = aligned
                r = norm_row_len(r)
                incoming[make_key(r)] = r

    # -----------------------------
    # If file doesn't exist, write fresh
    # -----------------------------
    if not combined_path.exists():
        with combined_path.open("w", encoding="utf-8", newline="") as f_out:
            w = csv.writer(f_out)
            w.writerow(header_used)
            for r in incoming.values():
                w.writerow(r)
        print(f"[WRITE] Created new combined file with {len(incoming)} rows: {combined_path}")
        return combined_path

    # -----------------------------
    # Append mode
    # -----------------------------
    if mode == "append":
        with combined_path.open("a", encoding="utf-8", newline="") as f_out:
            w = csv.writer(f_out)
            for r in incoming.values():
                w.writerow(r)

        print(f"[APPEND] Appended {len(incoming)} rows to: {combined_path}")
        return combined_path

    # -----------------------------
    # Replace mode
    # -----------------------------
    if mode != "replace":
        raise ValueError(f"Unknown mode: {mode}")

    tmp_path = combined_path.with_suffix(".tmp.csv")

    replaced = 0
    kept = 0
    appended_new = 0
    preserved_old = 0
    used_keys: set[str] = set()

    with combined_path.open("r", encoding="utf-8", newline="") as f_in, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        _ = next(reader, None)  # skip header
        writer.writerow(header_used)

        line_no = 1  # header is line 1

        for old_row in reader:
            line_no += 1
            old_row = norm_row_len(old_row)
            k = make_key(old_row)

            if k in incoming:
                new_row = norm_row_len(incoming[k])
                if missing_from_incoming_idx:
                    for i in missing_from_incoming_idx:
                        if i < len(new_row) and (new_row[i] or "") == "":
                            new_row[i] = old_row[i]
                used_keys.add(k)

                if llm_idx is not None:
                    old_llm = llm_val(old_row)
                    new_llm = llm_val(new_row)

                    if should_preserve_old_llm(old_llm, new_llm):
                        new_row[llm_idx] = old_llm
                        preserved_old += 1
                        print(
                            f"[REPLACE-PRESERVE] line={line_no} key={k} "
                            f"llm_relevance: {old_llm!r} -> {new_llm!r} (kept {old_llm!r})"
                        )
                    else:
                        print(
                            f"[REPLACE] line={line_no} key={k} "
                            f"llm_relevance: {old_llm!r} -> {new_llm!r}"
                        )
                    # print(
                    #     f"[REPLACE] line={line_no} key={k} "
                    #     f"llm_relevance: {old_llm!r} -> {new_llm!r}"
                    # )
                else:
                    print(f"[REPLACE] line={line_no} key={k}")

                replaced += 1
                writer.writerow(new_row)
            else:
                kept += 1
                writer.writerow(old_row)

        for k, r in incoming.items():
            if k not in used_keys:
                r = norm_row_len(r)
                appended_new += 1
                writer.writerow(r)
                if llm_idx is not None:
                    print(f"[ADD] key={k} llm_relevance={llm_val(r)!r} (not previously in file)")
                else:
                    print(f"[ADD] key={k} (not previously in file)")

    tmp_path.replace(combined_path)

    print(
        f"[DONE replace] replaced={replaced} kept={kept} appended_new={appended_new} "
        f"preserved_old_llm={preserved_old} "
        f"key_cols=('qid','{pid_col}') file={combined_path}"
    )
    return combined_path
