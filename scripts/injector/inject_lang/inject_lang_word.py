#!/usr/bin/env python3
import csv
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Set, List
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from helper import allow_huge_csv_fields

# ==============================
# Config (edit as needed)
# ==============================
REGION = "ap-southeast-2"   # AWS region
TARGET_LANG = "zh"          # e.g., 'vi' for Vietnamese; 'eng'/'en' => no translation
SEED = 42                   # set None for non-deterministic injection
INJECT_COUNT = 1            # how many times to inject EACH translated word
INJECT_PROB = 1.0           # probability per injection attempt (0..1)
TRECDL_YEAR = "2021"        # for folder naming only

INPUT_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/judged")                 # read these CSVs
OUTPUT_DIR = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/{TARGET_LANG}_word/")    # write mirrored CSVs

CACHE_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/translate_cache")

# Existing full-query cache (kept, but no longer required for word-level injection)
MAP_FILE   = CACHE_DIR / f"query_map_{TARGET_LANG}.csv"  # cols: query, translated

# NEW: word-level cache
WORD_MAP_FILE = CACHE_DIR / f"word_map_{TARGET_LANG}.csv"  # cols: word, translated

GLOB_PATTERN = "*.csv"

# ==============================
allow_huge_csv_fields()
rng = random.Random(SEED)

IDENTITY_LANG = TARGET_LANG.lower() in {"eng", "en"}
_translate = None
if not IDENTITY_LANG:
    import boto3
    _translate = boto3.client("translate", region_name=REGION)

# ---------- Tokenization ----------
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize_query_to_words(q: str) -> List[str]:
    """
    Tokenize the *English* query into individual word-like tokens.
    Adjust regex if you want to keep hyphens, etc.
    """
    return _WORD_RE.findall(q or "")

# ---------- Injection helpers ----------
def find_between_word_positions(text: str):
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

def inject_words(text: str, words: List[str], n: int, prob: float) -> str:
    """
    Inject each word in `words` up to `n` times (per-word), controlled by prob.
    """
    out = text
    clean = [w.strip() for w in words if (w or "").strip()]
    if not clean:
        return out

    for w in clean:
        for _ in range(max(0, n)):
            if rng.random() <= prob:
                out = inject_once(out, w)

    return out

# ---------- Mapping (cache) I/O ----------
def load_word_map(path: Path) -> Dict[str, str]:
    """Load word->translated map from CSV if exists (expects headers: word, translated)."""
    m: Dict[str, str] = {}
    if not path or not path.exists():
        return m
    with path.open("r", newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            w = (row.get("word") or "").strip()
            t = (row.get("translated") or "").strip()
            if w:
                m[w] = t
    return m

def save_word_map(path: Path, m: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["word", "translated"])
        w.writeheader()
        for k, v in m.items():
            w.writerow({"word": k, "translated": v})

# (kept: your existing query map helpers, unchanged)
def load_map(path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not path or not path.exists():
        return m
    with path.open("r", newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            q = (row.get("query") or "").strip()
            t = (row.get("translated") or "").strip()
            if q:
                m[q] = t
    return m

def save_map(path: Path, m: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["query", "translated"])
        w.writeheader()
        for q, t in m.items():
            w.writerow({"query": q, "translated": t})

def merge_maps(base: Dict[str, str], others: List[Dict[str, str]]) -> Dict[str, str]:
    out = dict(base)
    for om in others:
        for q, t in om.items():
            if q and q not in out and t:
                out[q] = t
    return out

# ---------- Translation ----------
def translate_word_en_to_target(w: str) -> str:
    """Translate a single EN word/token to target language."""
    if IDENTITY_LANG:
        return w
    resp = _translate.translate_text(
        Text=w,
        SourceLanguageCode="en",
        TargetLanguageCode=TARGET_LANG
    )
    return resp["TranslatedText"]

def translate_missing_words(words: Iterable[str], word_map: Dict[str, str]) -> Dict[str, str]:
    missing = [w for w in words if w and w not in word_map]
    if not missing:
        print("No missing words to translate.")
        return word_map

    print(f"Translating {len(missing)} new unique word{'s' if len(missing)!=1 else ''} to '{TARGET_LANG}'...")
    for i, w in enumerate(missing, 1):
        word_map[w] = translate_word_en_to_target(w)
        if i % 200 == 0 or i == len(missing):
            print(f"  translated {i}/{len(missing)}")
    return word_map

# ---------- Pipeline ----------
def collect_unique_query_words(files: Iterable[Path]) -> Set[str]:
    unique: Set[str] = set()
    for f in files:
        with f.open("r", newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            for row in r:
                q = (row.get("query") or "").strip()
                for w in tokenize_query_to_words(q):
                    unique.add(w)
    return unique

def process_file(in_path: Path, out_path: Path, word_map: Dict[str, str]) -> None:
    col_query_lang = "query_" + TARGET_LANG
    col_injected   = "passage_injected"

    with in_path.open("r", newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        if col_query_lang not in fieldnames:
            fieldnames.append(col_query_lang)
        if col_injected not in fieldnames:
            fieldnames.append(col_injected)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            q = (row.get("query", "") or "").strip()
            p = (row.get("passage", "") or "")

            # 1) tokenize EN query into words
            q_words = tokenize_query_to_words(q)

            # 2) translate each word via word_map (fallback to identity)
            translated_words = [word_map.get(w, w) for w in q_words]

            # 3) inject translated words
            p_inj = inject_words(p, translated_words, INJECT_COUNT, INJECT_PROB)

            # store something in query_<lang> for traceability
            row[col_query_lang] = " ".join(translated_words)
            row[col_injected] = p_inj

            valid_row = {k: v for k, v in row.items() if k in fieldnames and v is not None}
            writer.writerow(valid_row)

def parse_args():
    ap = argparse.ArgumentParser(description="Inject translated *query words* into passages using a word-level cache.")
    ap.add_argument("--fail-if-missing", action="store_true",
                    help="Exit with error if word translations are missing (skips AWS Translate).")
    return ap.parse_args()

def main():
    args = parse_args()

    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit(f"No CSV files found in: {INPUT_DIR} (pattern: {GLOB_PATTERN})")

    # Collect unique EN words across all queries
    print(f"Scanning {len(files)} file(s) for unique query words...")
    unique_words = collect_unique_query_words(files)
    print(f"Unique query words found: {len(unique_words)}")

    # Load word cache
    word_map = load_word_map(WORD_MAP_FILE)
    print(f"Word cache has {len(word_map)} entr{'y' if len(word_map)==1 else 'ies'} (file: {WORD_MAP_FILE.name})")

    if IDENTITY_LANG:
        print(f"Mode: identity injection (no AWS Translate for '{TARGET_LANG}').")
        for w in unique_words:
            if w not in word_map:
                word_map[w] = w
    else:
        missing = [w for w in unique_words if w not in word_map]
        if missing and args.fail_if_missing:
            print(f"Missing {len(missing)} word translations and --fail-if-missing set. Aborting without calling AWS Translate.")
            save_word_map(WORD_MAP_FILE, word_map)
            raise SystemExit(2)

        word_map = translate_missing_words(unique_words, word_map)

    save_word_map(WORD_MAP_FILE, word_map)
    print(f"Saved word map with {len(word_map)} entries → {WORD_MAP_FILE}")

    # (optional) keep your old full-query map save if you still want it elsewhere
    # cache_map = load_map(MAP_FILE)
    # save_map(MAP_FILE, cache_map)

    print(f"\nProcessing {len(files)} file(s) from {INPUT_DIR}")
    print(f"Writing outputs to {OUTPUT_DIR}\n")

    for i, in_path in enumerate(files, 1):
        out_path = OUTPUT_DIR / in_path.name
        print(f"[{i}/{len(files)}] {in_path.name} -> {out_path.name}")
        process_file(in_path, out_path, word_map)

    print("\nDone.")

if __name__ == "__main__":
    main()
