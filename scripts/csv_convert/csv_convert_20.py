import csv
import json
import re
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
INPUT_DIR = Path("outputs/trec_dl")          # folder containing the txt files
OUTPUT_FILE = Path("outputs/trec_dl/combined_results_top2.csv")
MAX_JUDGED_PER_QUERY = 2                     # keep only 2 judged docs per query

# Regex for document header lines like: Doc 1: 8305152 (rel=3, score=12.887)
DOC_HEADER_RE = re.compile(r'^Doc\s+\d+:\s+(\S+)\s+\(rel=(\d+),\s+score=.*\)$')

def extract_contents(passage_text: str) -> str:
    """Extract 'contents' from the JSON block; fall back to raw text if JSON parse fails."""
    try:
        passage_json = json.loads(passage_text)
        return passage_json.get("contents", "").strip()
    except json.JSONDecodeError:
        return passage_text.strip()

def parse_txt_file_top2(file_path: Path):
    """
    Parse a single retrieval text file and return up to 2 rows:
    (query, docid, contents, relevance) with numeric relevance only.
    """
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    query = None
    current_doc = None
    current_rel = None
    current_passage = None
    in_passage = False

    for line in lines:
        if line.startswith("Query:"):
            query = line.replace("Query:", "").strip()

        elif line.startswith("Doc "):
            # finalize previous doc if eligible
            if (current_doc is not None and
                current_passage is not None and
                str(current_rel).isdigit() and
                len(rows) < MAX_JUDGED_PER_QUERY):
                rows.append([query, current_doc, extract_contents(current_passage), int(current_rel)])

            # start new doc
            m = DOC_HEADER_RE.match(line)
            if m:
                current_doc, current_rel = m.groups()
                current_passage = ""
                in_passage = False

        elif line.startswith("Passage:") or line.startswith("Document:"):
            in_passage = True
            current_passage = ""

        elif line.startswith("-" * 5):  # separator
            if (current_doc is not None and
                current_passage is not None and
                str(current_rel).isdigit() and
                len(rows) < MAX_JUDGED_PER_QUERY):
                rows.append([query, current_doc, extract_contents(current_passage), int(current_rel)])
            # reset for next doc
            current_doc = current_rel = current_passage = None
            in_passage = False

        elif in_passage:
            current_passage += line + "\n"

        # stop early if we already collected 2 judged docs
        if len(rows) >= MAX_JUDGED_PER_QUERY:
            # no need to parse further docs in this file
            break

    # flush last doc if file didn’t end with separator and we still need rows
    if (len(rows) < MAX_JUDGED_PER_QUERY and
        current_doc is not None and
        current_passage is not None and
        str(current_rel).isdigit()):
        rows.append([query, current_doc, extract_contents(current_passage), int(current_rel)])

    # ensure at most 2
    return rows[:MAX_JUDGED_PER_QUERY]

def combine_txt_to_csv(input_dir: Path, output_file: Path):
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    all_rows = []
    for txt in txt_files:
        print(f"Processing: {txt.name}")
        all_rows.extend(parse_txt_file_top2(txt))

    with output_file.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query", "docid", "passage", "relevance"])
        for q, did, contents, rel in all_rows:
            writer.writerow([q, did, contents, rel])

    print(f"Wrote combined CSV (top {MAX_JUDGED_PER_QUERY} judged per query): {output_file}")

if __name__ == "__main__":
    combine_txt_to_csv(INPUT_DIR, OUTPUT_FILE)
