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

LANG = "leet"           # 'leet' or 'enclosed'

INPUT_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/nr")       # read these CSVs
OUTPUT_DIR = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/{LANG}/")    # write mirrored CSVs

# Filenames pattern to process
GLOB_PATTERN = "*part0.csv"

# ==============================
allow_huge_csv_fields()  # Raise CSV field size limit for giant cells
rng = random.Random(SEED)

# ---------- Leetspeak translation ----------
# Basic leetspeak mapping; extend as desired
_LEET_MAP: Dict[str, List[str]] = {
    "a": ["4", "@"],
    "b": ["8"],
    "e": ["3"],
    "g": ["9"],
    "i": ["1", "!"],
    "l": ["1"],
    "o": ["0"],
    "s": ["5", "$"],
    "t": ["7"],
    "z": ["2"],
    " ": ["_"],
}

_ENCLOSED_MAP: Dict[str, List[str]] = {
    "a": ["ⓐ", "Ⓐ"],
    "b": ["ⓑ", "Ⓑ"],
    "c": ["ⓒ", "Ⓒ"],
    "d": ["ⓓ", "Ⓓ"],
    "e": ["ⓔ", "Ⓔ"],
    "f": ["ⓕ", "Ⓕ"],
    "g": ["ⓖ", "Ⓖ"],
    "h": ["ⓗ", "Ⓗ"],
    "i": ["ⓘ", "Ⓘ"],
    "j": ["ⓙ", "Ⓙ"],
    "k": ["ⓚ", "Ⓚ"],
    "l": ["ⓛ", "Ⓛ"],
    "m": ["ⓜ", "Ⓜ"],
    "n": ["ⓝ", "ⓝ"],
    "o": ["ⓞ", "Ⓞ"],
    "p": ["ⓟ", "Ⓟ"],
    "q": ["ⓠ", "Ⓠ"],
    "r": ["ⓡ", "Ⓡ"],
    "s": ["ⓢ", "Ⓢ"],
    "t": ["ⓣ", "Ⓣ"],
    "u": ["ⓤ", "Ⓤ"],
    "v": ["ⓥ", "Ⓥ"],
    "w": ["ⓦ", "Ⓦ"],
    "x": ["ⓧ", "Ⓧ"],
    "y": ["ⓨ", "ⓨ"],
    "z": ["ⓩ", "Ⓩ"],
}

def to_ascii(text: str) -> str:
    """Convert a string to leetspeak using _LEET_MAP."""
    out_chars: List[str] = []
    translate_map: Dict[str, List[str]] = {}
    match LANG:
        case "leet":
             translate_map = _LEET_MAP
        case "enclosed":
             translate_map = _ENCLOSED_MAP

    for ch in text:
        lower = ch.lower()
        if lower in translate_map:
            # choose a (deterministic) replacement variant
            options = translate_map[lower]
            repl = rng.choice(options)
            out_chars.append(repl)
        else:
            out_chars.append(ch)
    return "".join(out_chars)

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

# ---------- Per-file processing ----------
def process_file(in_path: Path, out_path: Path) -> None:
    col_query_leet = "query_leet"
    col_injected   = "passage_injected"
    col_query_nr   = "query_nr"

    with in_path.open("r", newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])

        # --- Replace query_nr column with query_leet in the header ---
        if col_query_nr in fieldnames:
            idx = fieldnames.index(col_query_nr)
            fieldnames[idx] = col_query_leet
        elif col_query_leet not in fieldnames:
            # if there was no query_nr, just append query_leet
            fieldnames.append(col_query_leet)

        # ensure passage_injected exists
        if col_injected not in fieldnames:
            fieldnames.append(col_injected)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            q = (row.get("query", "") or "").strip()
            p = (row.get("passage", "") or "")

            q_leet = to_ascii(q)
            p_inj  = inject_n(p, q_leet, INJECT_COUNT, INJECT_PROB)

            # populate query_leet (which has replaced query_nr if it existed)
            row[col_query_leet] = q_leet
            row[col_injected]   = p_inj

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
