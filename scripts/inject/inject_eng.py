#!/usr/bin/env python3
import csv
import random
from pathlib import Path

# ==============================
# Config (edit as needed)
# ==============================
SEED = 42                   # set None for non-deterministic
INJECT_COUNT = 1            # how many times to inject the query
INJECT_PROB = 1.0           # probability per injection attempt (0..1)
INPUT_FILE = Path("outputs/trec_dl/combined_irrelevant_results_20.csv")
OUTPUT_FILE = Path("outputs/trec_dl/combined_result_injected_eng_20.csv")  # keeping same name for compatibility
# ==============================

rng = random.Random(SEED)

def find_between_word_positions(text: str):
    """Return insertion indices that place content BETWEEN words (never slicing a token)."""
    positions = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if i > 0 and j < n and not text[i-1].isspace() and not text[j].isspace():
                positions.append(j)
            i = j
        else:
            i += 1
    return positions

def inject_once(text: str, snippet: str) -> str:
    spots = find_between_word_positions(text)
    if not spots:
        return text
    idx = rng.choice(spots)
    return text[:idx] + snippet + " " + text[idx:]

def inject_n(text: str, snippet: str, n: int, prob: float) -> str:
    out = text
    for _ in range(max(0, n)):
        if rng.random() <= prob:
            out = inject_once(out, snippet)
    return out

with open(INPUT_FILE, newline="", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    fieldnames = list(reader.fieldnames or [])

    # Ensure we add the injected column
    if "passage_injected" not in fieldnames:
        fieldnames.append("passage_injected")

    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        query = row.get("query", "")
        passage = row.get("passage", "")

        # Inject the ORIGINAL query (no translation) into the passage
        row["passage_injected"] = inject_n(passage, query, INJECT_COUNT, INJECT_PROB)

        writer.writerow(row)

print(f"Done. Output saved to: {OUTPUT_FILE}")
