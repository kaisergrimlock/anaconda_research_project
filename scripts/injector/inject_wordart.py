from pyfiglet import Figlet
#!/usr/bin/env python3
import csv
import random
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from helper import allow_huge_csv_fields

# ==============================
# Config (edit as needed)
# ==============================
SEED = 42                   # set None for non-deterministic injection
INJECT_COUNT = 1            # how many times to inject the leetspeak query
INJECT_PROB = 1.0           # probability per injection attempt (0..1)
TRECDL_YEAR = "2023"        # for folder naming only

LANG = "art"

INPUT_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/nr")       # read these CSVs
OUTPUT_DIR = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/{LANG}/")    # write mirrored CSVs

# Filenames pattern to process
GLOB_PATTERN = "*part0.csv"

# ==============================
allow_huge_csv_fields()  # Raise CSV field size limit for giant cells
rng = random.Random(SEED)

# ---------- Leetspeak translation ----------

def to_ascii(text: str) -> str:
    """Convert a string to ASCII-art (Figlet) and return multiline art (preserve line breaks)."""
    f = Figlet(font="standard")  # or "slant", "big", "doom", etc.
    ascii_art = f.renderText(text or "")
    return ascii_art.rstrip("\n")

# ---------- Injection helpers ----------
def find_between_word_positions(text: str) -> List[int]:
    """Return insertion indices such that inserting at that index places content BETWEEN words."""
    positions: List[int] = []
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
    """
    Insert the snippet as its own paragraph (separate lines).
    If no suitable between-word spot is found, append snippet as a new paragraph.
    """
    spots = find_between_word_positions(text)
    if not spots:
        if text.strip():
            return text.rstrip() + "\n\n" + snippet + "\n"
        return snippet + "\n"

    idx = rng.choice(spots)
    # place the snippet on its own lines, trimming surrounding whitespace to avoid double spaces
    left = text[:idx].rstrip()
    right = text[idx:].lstrip()
    return left + "\n\n" + snippet + "\n\n" + right

def inject_n(text: str, snippet: str, n: int, prob: float) -> str:
    out = text
    for _ in range(max(0, n)):
        if rng.random() <= prob:
            out = inject_once(out, snippet)
    return out

# ---------- Per-file processing ----------
def process_file(in_path: Path, out_path: Path) -> None:
    col_query_art = "query_art"   # renamed from query_leet to query_art (holds original query)
    col_injected  = "passage_injected"
    col_query_nr  = "query_nr"

    with in_path.open("r", newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])

        # --- Replace query_nr column with query_art in the header ---
        if col_query_nr in fieldnames:
            idx = fieldnames.index(col_query_nr)
            fieldnames[idx] = col_query_art
        elif col_query_art not in fieldnames:
            # if there was no query_nr, just append query_art
            fieldnames.append(col_query_art)

        # ensure passage_injected exists
        if col_injected not in fieldnames:
            fieldnames.append(col_injected)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            q = (row.get("query", "") or "").strip()
            p = (row.get("passage", "") or "")

            # generate word-art for injection (kept only for passage injection)
            q_wordart = to_ascii(q)
            p_inj      = inject_n(p, q_wordart, INJECT_COUNT, INJECT_PROB)

            # populate query_art with the ORIGINAL query (do NOT copy the wordart here)
            row[col_query_art] = q
            row[col_injected]  = p_inj

            # optional: drop old query_nr key if present in the row dict
            if col_query_nr in row:
                row.pop(col_query_nr, None)

            # Ensure all keys in row are valid and not None
            valid_row = {k: v for k, v in row.items() if k in fieldnames and v is not None}
            writer.writerow(valid_row)

# ---------- Main ----------
def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit(f"No CSV files found in: {INPUT_DIR} (pattern: {GLOB_PATTERN})")

    print(f"Processing {len(files)} file(s) from {INPUT_DIR}")
    print(f"Writing outputs to {OUTPUT_DIR}\n")

    for i, in_path in enumerate(files, 1):
        out_path = OUTPUT_DIR / in_path.name
        print(f"[{i}/{len(files)}] {in_path.name} -> {out_path.name}")
        process_file(in_path, out_path)

    print("\nDone.")

if __name__ == "__main__":
    main()
