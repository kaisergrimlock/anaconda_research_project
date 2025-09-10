import csv
import random
from pathlib import Path
import boto3

# ==============================
# Config (edit as needed)
# ==============================
REGION = "ap-southeast-2"   # AWS region
TARGET_LANG = "vi"          # e.g., 'vi' for Vietnamese
SEED = 42                   # set None for non-deterministic
INJECT_COUNT = 1            # how many times to inject the translated query
INJECT_PROB = 1.0           # probability per injection attempt (0..1)
INPUT_FILE = Path("outputs/trec_dl/combined_irrelevant_results_20.csv")
OUTPUT_FILE = Path("outputs/trec_dl/combined_result_translated_"+ TARGET_LANG +"_20.csv")
# ==============================

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

with open(INPUT_FILE, newline="", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    fieldnames = list(reader.fieldnames or [])

    # Add our new columns
    colName = "query_" + TARGET_LANG
    if "query_translated" not in fieldnames:
        fieldnames.append(colName)
    if "passage_injected" not in fieldnames:
        fieldnames.append("passage_injected")

    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # 1) Translate the QUERY (not the passage)
        resp = translate.translate_text(
            Text=row["query"],
            SourceLanguageCode="auto",
            TargetLanguageCode=TARGET_LANG
        )
        query_translated = resp["TranslatedText"]

        # 2) Inject translated query into random positions in the passage
        passage = row.get("passage", "")
        passage_injected = inject_n(passage, query_translated, INJECT_COUNT, INJECT_PROB)

        # 3) Write results
        row[colName] = query_translated
        row["passage_injected"] = passage_injected
        writer.writerow(row)

print(f"Done. Output saved to: {OUTPUT_FILE}")
