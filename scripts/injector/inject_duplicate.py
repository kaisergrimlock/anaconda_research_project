import csv
import random
from pathlib import Path
import boto3

# ==============================
# Config (edit as needed)
# ==============================
SEED = 42                   # set None for non-deterministic
INJECT_COUNT = 1            # how many times to inject the translated query
INJECT_PROB = 1.0           # probability per injection attempt (0..1)
INPUT_FILE = Path("outputs/trec_dl/combined_irrelevant_results_20.csv")
OUTPUT_FILE = Path("outputs/trec_dl/combined_result_translated_duplicate_20.csv")
# ==============================

with open(INPUT_FILE, newline="", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    fieldnames = list(reader.fieldnames or [])

    # Only add this one extra column
    if "passage_injected" not in fieldnames:
        fieldnames.append("passage_injected")

    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        passage = row.get("passage", "")
        # Duplicate the passage with a single space in between (if non-empty)
        row["passage_injected"] = passage + ((" " + passage) if passage else "")
        writer.writerow(row)


print(f"Done. Output saved to: {OUTPUT_FILE}")
