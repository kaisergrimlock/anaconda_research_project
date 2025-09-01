import csv
import json
import re
from pathlib import Path
from collections import defaultdict

# ----------------------------
# Config
# ----------------------------
INPUT_DIR = Path("outputs/trec_dl")  # folder containing the .txt files
OUTPUT_FILE = Path("outputs/trec_dl/combined_irrelevant_results_20.csv")
MAX_DOCS_PER_QUERY = 2  # keep only 2 irrelevant docs per query

# Regex for document header lines like: Doc 1: 8305152 (rel=3, score=12.887)
DOC_HEADER_RE = re.compile(r'^Doc\s+\d+:\s+(\S+)\s+\(rel=(\d+),\s+score=.*\)$')


def extract_contents(passage_text: str) -> str:
    """Extract 'contents' from the JSON block; fall back to raw text if JSON parse fails."""
    try:
        passage_json = json.loads(passage_text)
        return passage_json.get("contents", "").strip()
    except json.JSONDecodeError:
        return passage_text.strip()


def parse_txt_file_irrelevant(file_path: Path):
    """
    Parse a single retrieval text file and return ALL rows with relevance == 0:
    (query, docid, contents, relevance)
    """
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    query = None
    current_doc = None
    current_rel = None
    current_passage = None
    in_passage = False

    def flush_current():
        # append only if judged irrelevant (rel == 0)
        if current_doc is not None and current_passage is not None and str(current_rel).isdigit():
            if int(current_rel) == 0:
                rows.append([query, current_doc, extract_contents(current_passage), 0])

    for line in lines:
        if line.startswith("Query:"):
            query = line.replace("Query:", "").strip()

        elif line.startswith("Doc "):
            flush_current()
            m = DOC_HEADER_RE.match(line)
            if m:
                current_doc, current_rel = m.groups()
                current_passage = ""
                in_passage = False

        elif line.startswith("Passage:") or line.startswith("Document:"):
            in_passage = True
            current_passage = ""

        elif line.startswith("-" * 5):  # separator
            flush_current()
            current_doc = current_rel = current_passage = None
            in_passage = False

        elif in_passage:
            current_passage += line + "\n"

    flush_current()
    return rows


def combine_txt_to_csv_irrelevant(input_dir: Path, output_file: Path):
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    query_doc_map = defaultdict(list)

    # Parse all files and group by query
    for txt in txt_files:
        print(f"Processing: {txt.name}")
        for query, docid, contents, rel in parse_txt_file_irrelevant(txt):
            if len(query_doc_map[query]) < MAX_DOCS_PER_QUERY:
                query_doc_map[query].append([query, docid, contents, rel])

    # Write out combined CSV
    with output_file.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query", "docid", "passage", "relevance"])
        for query, rows in query_doc_map.items():
            for row in rows:
                writer.writerow(row)

    print(f"Wrote combined CSV with up to {MAX_DOCS_PER_QUERY} irrelevant docs per query: {output_file}")


if __name__ == "__main__":
    combine_txt_to_csv_irrelevant(INPUT_DIR, OUTPUT_FILE)
