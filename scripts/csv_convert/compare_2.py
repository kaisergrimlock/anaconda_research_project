#!/usr/bin/env python3
import csv
from pathlib import Path
import re

BASE_DIR = Path("outputs/trec_dl_llm_label")
OUT_CSV  = BASE_DIR / "doc_rel_summary.csv"

# Match files like:
#   doc_rel_compare_anthropic.claude-3-haiku-20240307-v1.csv
#   doc_rel_compare_mistral.mixtral-8x7b-instruct-v0.csv
#   doc_rel_compare_openai.gpt-oss-20b-1.csv
# (optionally skip plain 'doc_rel_compare.csv' if present)
FILE_GLOB = "doc_rel_compare*.csv"
MODEL_RE  = re.compile(r"^doc_rel_compare_(.+)\.csv$", re.IGNORECASE)

def as_int(s):
    try:
        return int(str(s).strip())
    except Exception:
        return None

def summarize_file(path: Path):
    """Return (model, more_count, less_count) for one doc_rel_compare_*.csv."""
    m = MODEL_RE.match(path.name)
    if not m:
        return None  # skip files that don't encode a model name
    model = m.group(1)

    more = less = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return (model, more, less)

        # Find the right columns case-insensitively
        cols = {c.lower(): c for c in reader.fieldnames}
        doc_col = cols.get("docid")
        rel_col = cols.get("rel") or cols.get("relevance")
        tr_col  = cols.get("rel_translated") or cols.get("translated") or cols.get("relevance_translated")

        if not (rel_col and tr_col):
            # Not a comparison file; skip
            return (model, more, less)

        for row in reader:
            r  = as_int(row.get(rel_col))
            rt = as_int(row.get(tr_col))
            if r is None or rt is None:
                continue
            if rt > r:
                more += 1
            elif rt < r:
                less += 1

    return (model, more, less)

def main():
    rows = []
    for path in sorted(BASE_DIR.glob(FILE_GLOB)):
        # Optionally skip a combined file named exactly 'doc_rel_compare.csv'
        if path.name.lower() == "doc_rel_compare.csv":
            continue
        summary = summarize_file(path)
        if summary:
            rows.append(summary)

    # Write summary CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "more_relevant", "less_relevant"])
        for model, more, less in rows:
            writer.writerow([model, more, less])

    print(f"Wrote {len(rows)} model summaries to {OUT_CSV}")

if __name__ == "__main__":
    main()
