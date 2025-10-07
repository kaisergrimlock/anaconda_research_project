import csv
import random
from pathlib import Path
import boto3
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.report.process_baseline.helper import allow_huge_csv_fields


# ==============================
# Config (edit as needed)
# ==============================
REGION = "ap-southeast-2"   # AWS region
TARGET_LANG = "vi"          # e.g., 'vi' for Vietnamese
SEED = 42                   # set None for non-deterministic
INJECT_COUNT = 1            # how many times to inject the translated query
INJECT_PROB = 1.0           # probability per injection attempt (0..1)
TRECDL_YEAR = "2023"    # for folder naming only
# Process ALL CSV files in this folder:
INPUT_DIR = Path("retrieved/trec_dl_" + TRECDL_YEAR + "/judged")

# Output folder will mirror input filenames:
OUTPUT_DIR = Path("retrieved/trec_dl_" + TRECDL_YEAR + "/" + TARGET_LANG + "/")  
# ==============================

allow_huge_csv_fields() # Raise CSV field size limit for giant cells

# AWS Translate client
translate = boto3.client("translate", region_name=REGION)

rng = random.Random(SEED)

def find_between_word_positions(text: str):
    """
    Return insertion indices such that inserting at that index places content
    BETWEEN words (never slicing a token). Works on whitespace runs.
    """
    positions = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            # whitespace run is [i, j); insert before next non-space if surrounded by non-spaces
            if i > 0 and j < n and not text[i-1].isspace() and not text[j].isspace():
                positions.append(j)
            i = j
        else:
            i += 1
    return positions

def inject_once(text: str, snippet: str) -> str:
    """Inject `snippet` at a random valid boundary; falls back to original text if none."""
    spots = find_between_word_positions(text)
    if not spots:
        return text
    idx = rng.choice(spots)
    # add a trailing space so the next token isn't glued to snippet
    return text[:idx] + snippet + " " + text[idx:]

def inject_n(text: str, snippet: str, n: int, prob: float) -> str:
    """Inject up to n times with probability prob per attempt, rescanning boundaries each time."""
    out = text
    for _ in range(max(0, n)):
        if rng.random() <= prob:
            out = inject_once(out, snippet)
    return out

def process_file(in_path: Path, out_path: Path):
    col_query_lang = "query_" + TARGET_LANG
    col_injected = "passage_injected"

    with open(in_path, newline="", encoding="utf-8") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])

        # Ensure our new columns exist
        if col_query_lang not in fieldnames:
            fieldnames.append(col_query_lang)
        if col_injected not in fieldnames:
            fieldnames.append(col_injected)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # 1) Translate the QUERY (not the passage)
            resp = translate.translate_text(
                Text=row.get("query", ""),
                SourceLanguageCode="auto",
                TargetLanguageCode=TARGET_LANG
            )
            query_translated = resp["TranslatedText"]

            # 2) Inject translated query into random positions in the passage
            passage = row.get("passage", "")
            passage_injected = inject_n(passage, query_translated, INJECT_COUNT, INJECT_PROB)

            # 3) Write results
            row[col_query_lang] = query_translated
            row[col_injected] = passage_injected
            writer.writerow(row)

def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in: {INPUT_DIR}")

    print(f"Processing {len(files)} file(s) from {INPUT_DIR}")
    print(f"Writing outputs to {OUTPUT_DIR}\n")

    for i, in_path in enumerate(files, 1):
        out_path = OUTPUT_DIR / in_path.name
        print(f"[{i}/{len(files)}] {in_path.name} -> {out_path.name}")
        process_file(in_path, out_path)

    print("\nDone.")

if __name__ == "__main__":
    main()
