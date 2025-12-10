#!/usr/bin/env python3
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, Set, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from helper import allow_huge_csv_fields

# ==============================
# Config (edit as needed)
# ==============================
REGION = "ap-southeast-2"   # AWS region

# Two target languages
LANG_1 = "th"             # e.g. 'vi' for Vietnamese
LANG_2 = "vi"              # e.g. 'th' for Thai
TARGET_LANGS: List[str] = [LANG_1, LANG_2]

TRECDL_YEAR = "2022"        # for folder naming only

# Injection config:
# now we use a single injection count for the mixed query
INJECT_COUNTS: Dict[str, int] = {
    LANG_1: 2,
    LANG_2: 2,
}
MIX_INJECT_COUNT = max(INJECT_COUNTS.values()) if INJECT_COUNTS else 0
INJECT_PROB = 1.0           # probability per injection attempt (0..1)

INPUT_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/judged")       # read these CSVs
# Output subfolder can be named after the languages, e.g. "vi_th_between"
OUTPUT_DIR = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/{'_'.join(TARGET_LANGS)}_between/")

# Cache dir: one map per language (now mapping *tokens* → translated tokens)
CACHE_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/translate_cache")
MAP_FILES  = {
    lang: CACHE_DIR / f"query_map_{lang}.csv"
    for lang in TARGET_LANGS
}

# Filenames pattern to process
GLOB_PATTERN = "*.csv"

# ==============================
allow_huge_csv_fields()  # Raise CSV field size limit for giant cells
rng = random.Random(42)  # fixed seed for reproducibility; adjust/None if needed

IDENTITY_LANGS: Dict[str, bool] = {
    lang: lang.lower() in {"eng", "en"} for lang in TARGET_LANGS
}

_translate = None
if not all(IDENTITY_LANGS.values()):
    import boto3  # lazy import so script works without boto3 when not needed
    _translate = boto3.client("translate", region_name=REGION)

# ---------- Injection helpers ----------
def find_between_word_positions(text: str):
    """Return insertion indices such that inserting at that index places content BETWEEN words."""
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

# ---------- Mapping (cache) I/O ----------
def load_map(path: Path) -> Dict[str, str]:
    """Load token->translated map from CSV if exists (headers: query, translated)."""
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
    """Left-most wins: keep existing entries in base; add missing from others in order."""
    out = dict(base)
    for om in others:
        for q, t in om.items():
            if q and q not in out and t:
                out[q] = t
    return out

# ---------- Translation ----------
def translate_one(piece: str, target_lang: str) -> str:
    """Return translated text (or identity if IDENTITY_LANGS[target_lang])."""
    if IDENTITY_LANGS.get(target_lang, False):
        return piece
    resp = _translate.translate_text(
        Text=piece,
        SourceLanguageCode="auto",
        TargetLanguageCode=target_lang,
    )
    return resp["TranslatedText"]

def translate_unique_for_lang(
    pieces: Iterable[str],
    existing_map: Dict[str, str],
    target_lang: str,
) -> Dict[str, str]:
    """
    Translate only the pieces (tokens) missing from the map
    for a specific language.
    """
    missing = [q for q in pieces if q and q not in existing_map]
    if not missing:
        print(f"[{target_lang}] No missing tokens to translate.")
        return existing_map

    print(f"[{target_lang}] Translating {len(missing)} unique token{'s' if len(missing) != 1 else ''}...")
    for i, q in enumerate(missing, 1):
        t = translate_one(q, target_lang)
        existing_map[q] = t
        if i % 100 == 0 or i == len(missing):
            print(f"  [{target_lang}] translated {i}/{len(missing)}")

    return existing_map

# ---------- Pipeline ----------
def collect_unique_tokens(files: Iterable[Path]) -> Set[str]:
    """
    Collect all unique whitespace-separated tokens from the 'query' column
    across all input files.
    """
    unique: Set[str] = set()
    for f in files:
        with f.open("r", newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            for row in r:
                q = (row.get("query") or "").strip()
                if q:
                    for token in q.split():
                        if token:
                            unique.add(token)
    return unique

def build_mixed_query(q: str, qmaps: Dict[str, Dict[str, str]]) -> str:
    """
    Build a mixed-language query from the original query `q`:

      1st word  (index 1, odd)  -> LANG_1
      2nd word  (index 2, even) -> LANG_2
      3rd word  (index 3, odd)  -> LANG_1
      ...

    Using zero-based index i, that means:
      i % 2 == 0 -> LANG_1
      i % 2 == 1 -> LANG_2
    """
    if not q:
        return q

    words = q.split()
    mixed_tokens: List[str] = []

    for i, w in enumerate(words):
        lang = LANG_1 if (i % 2 == 0) else LANG_2  # 0-based -> odd(1st,3rd,...) → LANG_1
        lang_map = qmaps.get(lang, {})
        mixed_tokens.append(lang_map.get(w, w))

    return " ".join(mixed_tokens)

def process_file(in_path: Path, out_path: Path, qmaps: Dict[str, Dict[str, str]]) -> None:
    """
    For each input row:
      - Build the mixed-language query using alternating LANG_1/LANG_2 per word
      - Inject that mixed query into the passage
      - Write query_mixed and passage_injected
    """
    col_injected = "passage_injected"
    col_query_mixed = "query_mixed"

    with in_path.open("r", newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        base_fieldnames = list(reader.fieldnames or [])

        # Add query_mixed column
        if col_query_mixed not in base_fieldnames:
            base_fieldnames.append(col_query_mixed)

        # Add passage_injected
        if col_injected not in base_fieldnames:
            base_fieldnames.append(col_injected)

        writer = csv.DictWriter(fout, fieldnames=base_fieldnames)
        writer.writeheader()

        for row in reader:
            q = (row.get("query", "") or "").strip()
            p = (row.get("passage", "") or "")

            mixed_query = build_mixed_query(q, qmaps)

            # Inject the mixed query into the passage
            p_inj = p
            if mixed_query:
                p_inj = inject_n(p_inj, mixed_query, MIX_INJECT_COUNT, INJECT_PROB)

            row[col_query_mixed] = mixed_query
            row[col_injected] = p_inj

            # Ensure all keys in row are valid and not None
            valid_row = {k: ("" if v is None else v) for k, v in row.items() if k in base_fieldnames}
            writer.writerow(valid_row)

def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit(f"No CSV files found in: {INPUT_DIR} (pattern: {GLOB_PATTERN})")

    print(f"Scanning {len(files)} file(s) for unique tokens in queries...")
    unique_tokens = collect_unique_tokens(files)
    print(f"Unique tokens found: {len(unique_tokens)}")

    # For each language, load cache, translate missing tokens, save back
    qmaps: Dict[str, Dict[str, str]] = {}

    for lang in TARGET_LANGS:
        map_path = MAP_FILES[lang]
        cache_map = load_map(map_path)
        print(f"[{lang}] Cache has {len(cache_map)} entr{'y' if len(cache_map)==1 else 'ies'} ({map_path.name})")

        if IDENTITY_LANGS[lang]:
            print(f"[{lang}] Mode: identity (no AWS Translate).")
            for tok in unique_tokens:
                if tok not in cache_map:
                    cache_map[tok] = tok
        else:
            cache_map = translate_unique_for_lang(unique_tokens, cache_map, lang)

        save_map(map_path, cache_map)
        print(f"[{lang}] Saved map with {len(cache_map)} entries → {map_path}")
        qmaps[lang] = cache_map

    # Process files using the maps
    print(f"\nProcessing {len(files)} file(s) from {INPUT_DIR}")
    print(f"Writing outputs to {OUTPUT_DIR}\n")

    for i, in_path in enumerate(files, 1):
        out_path = OUTPUT_DIR / in_path.name
        print(f"[{i}/{len(files)}] {in_path.name} -> {out_path.name}")
        process_file(in_path, out_path, qmaps)

    print("\nDone.")

if __name__ == "__main__":
    main()
