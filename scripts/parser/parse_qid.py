#!/usr/bin/env python3
from __future__ import annotations

import csv, sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# =========================
# CONFIG — edit these
# =========================
TREC_DL_YEAR = "2023"
MODEL        = "gpt-oss-20b"
LANG         = "eng"   # e.g. "eng", "vi", "fr", or "raw"

# Where your judged/NIST files live (unused for the join, but kept for parity)
NIST_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"

# Input file you want to add qid to (typical LLM labels export)
if LANG != "raw":
    LLM_FILE = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_raw.csv"
    OUT_DIR  = Path("outputs/baseline") / TREC_DL_YEAR / LANG
else:
    LLM_FILE = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_raw.csv"
    OUT_DIR  = Path("outputs/baseline") / TREC_DL_YEAR / "raw"

# Folder that contains the per-topic CSVs like: all_topics_trecdl_2023_part*.csv
# Adjust these two lines if your layout differs (e.g., change "en" or use another LANG)
TOPICS_DIR  = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / LANG
TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"
# =========================


def _bump_field_limit():
    # allow very large passage cells
    limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
    while limit >= 131072:
        try:
            csv.field_size_limit(limit); return
        except OverflowError:
            limit //= 10
_bump_field_limit()


def _pick(colnames, *candidates):
    cols = {c.lower() for c in colnames}
    for cset in candidates:
        for c in cset:
            if c.lower() in cols:
                return c
    return None


def build_lookup(topics_dir: Path, pattern: str) -> Tuple[Dict[Tuple[str, str], str], Dict[str, str]]:
    """
    Returns:
      mapping_pair[(pid, docid)] = qid
      mapping_pid[pid]           = qid (first-seen wins)
    """
    mapping_pair: Dict[Tuple[str, str], str] = {}
    mapping_pid: Dict[str, str] = {}

    files = sorted(topics_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No topic files matching {pattern!r} in {topics_dir}")

    for fp in files:
        with fp.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = [h.strip() for h in (reader.fieldnames or [])]
            if not header:  # empty file
                continue

            qid_col = _pick(header, ("qid",), ("topic_id",), ("topic", "qid"))
            pid_col = _pick(header, ("pid",), ("passage_id",), ("docid",), ("doc_id",), ("docno",))
            doc_col = _pick(header, ("docid",), ("doc_id",), ("docno",))

            if not qid_col or not pid_col:
                # skip files that don't have the required cols
                continue

            for row in reader:
                qid = (row.get(qid_col) or "").strip()
                pid = (row.get(pid_col) or "").strip()
                doc = (row.get(doc_col) or "").strip() if doc_col else ""
                if not qid or not pid:
                    continue
                if doc:
                    mapping_pair.setdefault((pid, doc), qid)
                mapping_pid.setdefault(pid, qid)

    if not mapping_pair and not mapping_pid:
        raise RuntimeError(f"Could not build any (pid/docid)->qid mapping from {topics_dir} / {pattern}")
    return mapping_pair, mapping_pid


def add_qid(input_csv: Path, output_csv: Path, topics_dir: Path, pattern: str):
    mapping_pair, mapping_pid = build_lookup(topics_dir, pattern)

    with input_csv.open("r", newline="", encoding="utf-8") as fin, \
         output_csv.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        in_headers = [h.strip() for h in (reader.fieldnames or [])]
        if not in_headers:
            raise RuntimeError("Input file has no header row.")

        pid_col = _pick(in_headers, ("pid",), ("passage_id",), ("docid",), ("doc_id",), ("docno",))
        doc_col = _pick(in_headers, ("docid",), ("doc_id",), ("docno",))
        if not pid_col:
            raise RuntimeError("Input CSV needs a pid-like column (pid/passage_id/docid/doc_id/docno).")

        # Insert qid after pid if not present
        out_headers = in_headers.copy()
        if "qid" not in [h.lower() for h in out_headers]:
            insert_pos = out_headers.index(pid_col) + 1 if pid_col in out_headers else len(out_headers)
            out_headers.insert(insert_pos, "qid")

        writer = csv.DictWriter(fout, fieldnames=out_headers)
        writer.writeheader()

        total = found = 0
        for row in reader:
            total += 1
            pid = (row.get(pid_col) or "").strip()
            doc = (row.get(doc_col) or "").strip() if doc_col else ""

            qid: Optional[str] = None
            if pid and doc and (pid, doc) in mapping_pair:
                qid = mapping_pair[(pid, doc)]
            elif pid in mapping_pid:
                qid = mapping_pid[pid]

            if qid:
                found += 1

            row_out = dict(row)
            row_out["qid"] = qid or ""
            writer.writerow(row_out)

    print(f"[done] wrote: {output_csv}")
    print(f"Rows: {total}, matched qid for: {found} ({(found/total*100):.1f}%)")


def main():
    # Resolve output path based on config
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / (LLM_FILE.stem + "_with_qid.csv")

    if not LLM_FILE.exists():
        raise FileNotFoundError(f"Input CSV not found: {LLM_FILE}")
    if not TOPICS_DIR.exists():
        raise FileNotFoundError(f"Topics dir not found: {TOPICS_DIR} (edit TOPICS_DIR in CONFIG)")

    add_qid(LLM_FILE, out_file, TOPICS_DIR, TOPICS_GLOB)


if __name__ == "__main__":
    main()
