import csv
import json
import re
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
INPUT_DIR = Path("outputs/trec_dl")   # folder containing the txt files
OUTPUT_FILE = Path("outputs/trec_dl/combined_results.csv")

# Regex to parse document header line
DOC_HEADER_RE = re.compile(r'^Doc\s+\d+:\s+(\S+)\s+\(rel=(\d+),\s+score=.*\)$')

def extract_contents(passage_text):
    """
    Extract the "contents" value from the JSON-like text.
    If parsing fails, return the original passage text.
    """
    try:
        passage_json = json.loads(passage_text)
        return passage_json.get("contents", "").strip()
    except json.JSONDecodeError:
        return passage_text.strip()

def parse_txt_file(file_path):
    """
    Parse a single retrieval text file and return rows as (query, docid, contents, relevance)
    Only rows with numeric relevance are returned.
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    query = None
    current_doc, current_rel, current_passage = None, None, None
    in_passage = False

    for line in lines:
        if line.startswith("Query:"):
            query = line.replace("Query:", "").strip()
        elif line.startswith("Doc "):
            # Save the previous document if it exists and has a numeric relevance
            if current_doc and current_passage is not None and str(current_rel).isdigit():
                rows.append([query, current_doc, extract_contents(current_passage), current_rel])

            match = DOC_HEADER_RE.match(line)
            if match:
                current_doc, current_rel = match.groups()
                current_passage = ""
                in_passage = False
        elif line.startswith("Passage:") or line.startswith("Document:"):
            in_passage = True
            current_passage = ""
        elif line.startswith("-" * 5):
            # Save current document if relevance is numeric
            if current_doc and current_passage is not None and str(current_rel).isdigit():
                rows.append([query, current_doc, extract_contents(current_passage), current_rel])
                current_doc, current_rel, current_passage = None, None, None
                in_passage = False
        elif in_passage:
            current_passage += line + "\n"

    # Handle the last document if relevance is numeric
    if current_doc and current_passage is not None and str(current_rel).isdigit():
        rows.append([query, current_doc, extract_contents(current_passage), current_rel])

    return rows


def combine_txt_to_csv(input_dir, output_file):
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    all_rows = []
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")
        rows = parse_txt_file(txt_file)
        all_rows.extend(rows)

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query", "docid", "passage", "relevance"])
        for row in all_rows:
            writer.writerow(row)

    print(f"Combined CSV written to: {output_file}")


if __name__ == "__main__":
    combine_txt_to_csv(INPUT_DIR, OUTPUT_FILE)
