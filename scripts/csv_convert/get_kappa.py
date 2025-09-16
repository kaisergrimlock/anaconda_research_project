# pip install scikit-learn
import csv
from pathlib import Path
import re
from typing import List, Tuple, Optional
from sklearn.metrics import cohen_kappa_score

# --- Config ---
BASE_DIR = Path("outputs/trec_dl_llm_label")
OUT_CSV  = BASE_DIR / "doc_rel_kappa_summary.csv"

# Files like:
#   doc_rel_compare_anthropic.claude-3-haiku-20240307-v1.csv
#   doc_rel_compare_mistral.mixtral-8x7b-instruct-v0.csv
#   doc_rel_compare_openai.gpt-oss-20b-1.csv
FILE_GLOB = "doc_rel_compare*.csv"
MODEL_RE  = re.compile(r"^doc_rel_compare_(.+)\.csv$", re.IGNORECASE)

def checkInt(string: str) -> Optional[int]:
    try:
        return int(str(string).strip())
    except ValueError:
        return None
    
def load_rel_columns(path: Path) -> Tuple[List[int], List[int]]:
    """
    Read CSV and return (rel, rel_translated) as integer lists.
    Case-insensitive column matching; skips rows with missing/non-int values.
    """
    y1, y2 = [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        # Read a sample to auto-detect delimiter for robustness
        sample = f.read(4096)
        f.seek(0)
        try:
            # Try to detect the CSV dialect (delimiter, etc.)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            # Fallback to default CSV dialect if detection fails
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            # Return empty lists if no columns found
            return y1, y2
        # Build a mapping of lowercase column names to actual names for case-insensitive matching
        cols = {c.lower(): c for c in reader.fieldnames}
        # Try to find the relevant columns for rel and rel_translated
        rel_col = cols.get("rel") or cols.get("relevance")
        tr_col  = cols.get("rel_translated") or cols.get("translated") or cols.get("relevance_translated")
        if not (rel_col and tr_col):
            # Return empty lists if required columns are missing
            return y1, y2
        for row in reader:
            # Convert values to int, skip if conversion fails
            r  = checkInt(row.get(rel_col))
            rt = checkInt(row.get(tr_col))
            if r is None or rt is None:
                continue
            # Append valid integer values to result lists
            y1.append(r)
            y2.append(rt)
    return y1, y2


def summarize_file(path: Path):
    """
    Return (model, n, kappa_nominal, kappa_linear, kappa_quadratic, more, less, unchanged)
    for one comparison file. If the filename doesn't encode a model name, return None.
    """
    m = MODEL_RE.match(path.name)
    if not m:
        return None
    model = m.group(1)

    y1, y2 = load_rel_columns(path)
    n = min(len(y1), len(y2))
    if n == 0:
        return (model, 0, float("nan"), float("nan"), float("nan"), 0, 0, 0)

    # scikit-learn κ
    k_nom  = cohen_kappa_score(y1, y2, weights=None)         # nominal
    k_lin  = cohen_kappa_score(y1, y2, weights="linear")     # linear-weighted
    k_quad = cohen_kappa_score(y1, y2, weights="quadratic")  # quadratic-weighted

    # Simple change counts
    more = less = unchanged = 0
    for a, b in zip(y1, y2):
        if b > a:   more += 1
        elif b < a: less += 1
        else:       unchanged += 1

    return (model, n, k_nom, k_lin, k_quad, more, less, unchanged)

def main():
    rows = []
    for path in sorted(BASE_DIR.glob(FILE_GLOB)):
        # Optionally skip combined file named exactly 'doc_rel_compare.csv'
        if path.name.lower() == "doc_rel_compare.csv":
            continue
        summary = summarize_file(path)
        if summary:
            rows.append(summary)

    # Write summary CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "n",
            "kappa_nominal", "kappa_linear", "kappa_quadratic",
            "more_relevant", "less_relevant", "unchanged"
        ])
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} model summaries to {OUT_CSV}")

if __name__ == "__main__":
    main()