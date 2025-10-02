#!/usr/bin/env python3
"""
Fill missing 'relevance' in all_llm_labels.csv using relevance_missing_rows.csv,
with robust normalization + fallback matching.

Priority of matching:
  1) exact (docid, query, passage) after normalization
  2) (docid, query) if uniquely mapped in lookup
  3) (docid) if uniquely mapped in lookup

Outputs:
  - all_llm_labels_filled.csv
  - prints a summary + a few examples that still don't match
"""

from pathlib import Path
import csv
import argparse
import re
import unicodedata
from collections import defaultdict, Counter

REPO_ROOT = Path(".").resolve()
MISSING_PATH = REPO_ROOT / "outputs" / "trec_dl_llm_label" / "processed" / "relevance_missing_rows.csv"
TARGET_PATH  = REPO_ROOT / "outputs" / "trec_dl_llm_label" / "processed" / "all_llm_labels.csv"
OUTPUT_PATH  = REPO_ROOT / "outputs" / "trec_dl_llm_label" / "processed" / "all_llm_labels_filled.csv"

WS_RE = re.compile(r"\s+", flags=re.UNICODE)

def norm(s: str | None) -> str:
    """Unicode-normalize, strip control chars, collapse whitespace, strip quotes."""
    if s is None:
        return ""
    # Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    # Remove control chars (except \n which we unify to space anyway)
    s = "".join(ch for ch in s if (ch >= " " or ch == "\n"))
    # Replace newlines/tabs with spaces, collapse to single spaces
    s = WS_RE.sub(" ", s.replace("\t", " ").replace("\r", " ").replace("\n", " "))
    # Strip outer quotes and surrounding spaces
    s = s.strip().strip('"').strip("'").strip()
    return s

def is_missing(val: str | None) -> bool:
    if val is None:
        return True
    v = str(val).strip()
    return v == "" or v.upper() in {"NA", "N/A", "NULL", "NONE"}

def build_unique_map(pairs: list[tuple[tuple[str, ...], str]]) -> dict[tuple[str, ...], str]:
    """Keep only keys that map to a single unique value."""
    bag = defaultdict(set)
    for k, v in pairs:
        bag[k].add(v)
    return {k: next(iter(vs)) for k, vs in bag.items() if len(vs) == 1}

def main(missing_csv: Path, target_csv: Path, out_csv: Path) -> None:
    if not missing_csv.exists():
        raise FileNotFoundError(missing_csv)
    if not target_csv.exists():
        raise FileNotFoundError(target_csv)

    # ---- Load lookup rows (these *have* the correct relevance) ----
    with missing_csv.open("r", encoding="utf-8-sig", newline="") as f:
        lr = csv.DictReader(f)
        need = {"docid", "query", "passage", "relevance"}
        if not need.issubset(set(lr.fieldnames or [])):
            raise ValueError(f"{missing_csv} must have columns {sorted(need)}; found {lr.fieldnames}")

        exact_pairs = []
        dq_pairs = []
        d_pairs = []
        for row in lr:
            rel = norm(row["relevance"])
            if is_missing(rel):
                continue
            docid = norm(row["docid"])
            query = norm(row["query"])
            passage = norm(row["passage"])
            exact_pairs.append(((docid, query, passage), rel))
            dq_pairs.append(((docid, query), rel))
            d_pairs.append(((docid,), rel))

    lut_exact = dict(exact_pairs)  # if duplicate exact keys exist, last wins (fine)
    lut_dq    = build_unique_map(dq_pairs)
    lut_d     = build_unique_map(d_pairs)

    # ---- Process target and fill ----
    fills = Counter()  # counts by strategy
    still_unmatched_samples = []

    with target_csv.open("r", encoding="utf-8-sig", newline="") as fin, \
         out_csv.open("w", encoding="utf-8", newline="") as fout:

        r = csv.DictReader(fin)
        fieldnames = r.fieldnames or []
        for k in ("docid", "query", "passage", "relevance"):
            if k not in fieldnames:
                raise ValueError(f"Column '{k}' missing in {target_csv}. Columns: {fieldnames}")

        w = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()

        total_missing = 0
        filled = 0

        for row in r:
            if is_missing(row.get("relevance")):
                total_missing += 1
                docid = norm(row["docid"])
                query = norm(row["query"])
                passage = norm(row["passage"])

                key3 = (docid, query, passage)
                rel = None
                if key3 in lut_exact:
                    rel = lut_exact[key3]
                    fills["exact(docid,query,passage)"] += 1
                elif (docid, query) in lut_dq:
                    rel = lut_dq[(docid, query)]
                    fills["fallback(docid,query)"] += 1
                elif (docid,) in lut_d:
                    rel = lut_d[(docid,)]
                    fills["fallback(docid)"] += 1

                if rel is not None:
                    row["relevance"] = rel
                    filled += 1
                else:
                    if len(still_unmatched_samples) < 8:
                        still_unmatched_samples.append({
                            "docid": row["docid"],
                            "query": row["query"][:120],
                            "passage(head)": row["passage"][:120]
                        })

            w.writerow(row)

    # ---- Summary ----
    print("=== Fill Summary ===")
    print(f"Target file:        {target_csv}")
    print(f"Lookup file:        {missing_csv}")
    print(f"Output written to:  {out_csv}")
    print(f"Missing in target:  {total_missing}")
    print(f"Filled total:       {filled}")
    for k, v in fills.items():
        print(f"  - {k:28s}: {v}")
    if still_unmatched_samples:
        print("\nExamples still unmatched (showing first 8, truncated):")
        for s in still_unmatched_samples:
            print(f"  docid={s['docid']} | query='{s['query']}' | passage='{s['passage(head)']}...'")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fill missing relevance values with normalization and fallbacks.")
    ap.add_argument("--missing", type=Path, default=MISSING_PATH, help="relevance_missing_rows.csv")
    ap.add_argument("--target",  type=Path, default=TARGET_PATH,  help="all_llm_labels.csv")
    ap.add_argument("--out",     type=Path, default=OUTPUT_PATH,  help="output CSV path")
    args = ap.parse_args()
    main(args.missing, args.target, args.out)
