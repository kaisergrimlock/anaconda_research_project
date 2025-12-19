#!/usr/bin/env python3
import csv
import random
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
TARGET_LANG = "ar"          # e.g., 'vi' for Vietnamese; 'eng'/'en' => no translation
SEED = 42                   # set None for non-deterministic injection
INJECT_COUNT = 1           # how many times to inject the translated query
INJECT_PROB = 1.0           # probability per injection attempt (0..1)
TRECDL_YEAR = "2021"        # for folder naming only

INPUT_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/judged")            # read these CSVs
OUTPUT_DIR = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/{TARGET_LANG}/")    # write mirrored CSVs

# Cache map to avoid re-translation later runs
CACHE_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/translate_cache")
MAP_FILE   = CACHE_DIR / f"query_map_{TARGET_LANG}.csv"  # cols: query, translated

# Filenames pattern to process
GLOB_PATTERN = "*.csv"

# ==============================
allow_huge_csv_fields()  # Raise CSV field size limit for giant cells
rng = random.Random(SEED)

IDENTITY_LANG = TARGET_LANG.lower() in {"eng", "en"}
_translate = None
if not IDENTITY_LANG:
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
    """Load query->translated map from CSV if exists (expects headers: query, translated)."""
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

# ---------- Harvest translations from output CSVs ----------
def harvest_from_outputs(files: Iterable[Path], col_query_lang: str) -> Dict[str, str]:
    """
    From already-produced output CSVs (that include query and query_<lang>),
    collect a map query -> query_<lang>.
    """
    m: Dict[str, str] = {}
    for f in files:
        try:
            with f.open("r", newline="", encoding="utf-8") as fh:
                r = csv.DictReader(fh)
                if not r.fieldnames or "query" not in r.fieldnames or col_query_lang not in r.fieldnames:
                    continue
                for row in r:
                    q = (row.get("query") or "").strip()
                    t = (row.get(col_query_lang) or "").strip()
                    if q and t and q not in m:
                        m[q] = t
        except FileNotFoundError:
            continue
    return m

# ---------- Translation ----------
def translate_one(q: str) -> str:
    """Return translated query (or identity if IDENTITY_LANG)."""
    if IDENTITY_LANG:
        return q
    resp = _translate.translate_text(
        Text=q,
        SourceLanguageCode="auto",
        TargetLanguageCode=TARGET_LANG
    )
    return resp["TranslatedText"]

def translate_unique(queries: Iterable[str], existing_map: Dict[str, str]) -> Dict[str, str]:
    """Translate only the queries missing from the map; return the updated map."""
    missing = [q for q in queries if q and q not in existing_map]
    if not missing:
        print("No missing queries to translate.")
        return existing_map

    print(f"Translating {len(missing)} new unique quer{'y' if len(missing)==1 else 'ies'} to '{TARGET_LANG}'...")
    for i, q in enumerate(missing, 1):
        t = translate_one(q)
        existing_map[q] = t
        if i % 100 == 0 or i == len(missing):
            print(f"  translated {i}/{len(missing)}")

    return existing_map

# ---------- Pipeline ----------
def collect_unique_queries(files: Iterable[Path]) -> Set[str]:
    unique: Set[str] = set()
    for f in files:
        with f.open("r", newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            for row in r:
                q = (row.get("query") or "").strip()
                if q:
                    unique.add(q)
    return unique

def process_file(in_path: Path, out_path: Path, qmap: Dict[str, str]) -> None:
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

            q_t = qmap.get(q, q)  # fallback to identity if somehow missing
            p_inj = inject_n(p, q_t, INJECT_COUNT, INJECT_PROB)

            row[col_query_lang] = q_t
            row[col_injected]   = p_inj

            # Ensure all keys in row are valid and not None
            valid_row = {k: v for k, v in row.items() if k in fieldnames and v is not None}
            writer.writerow(valid_row)

def parse_args():
    ap = argparse.ArgumentParser(description="Inject translated queries into passages; reuse existing translation maps if available.")
    ap.add_argument("--map", type=str, default=None,
                    help="Path to an existing translation CSV (query,translated). Merged into cache before translating.")
    ap.add_argument("--no-harvest", action="store_true",
                    help="Disable harvesting translations from existing output CSVs in OUTPUT_DIR.")
    ap.add_argument("--fail-if-missing", action="store_true",
                    help="Exit with error if translations are missing after merging maps (skips AWS Translate).")
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

    print(f"Scanning {len(files)} file(s) for unique queries...")
    unique_queries = collect_unique_queries(files)
    print(f"Unique queries found: {len(unique_queries)}")

    # Load main cache
    cache_map = load_map(MAP_FILE)
    print(f"Cache has {len(cache_map)} translated entr{'y' if len(cache_map)==1 else 'ies'} (file: {MAP_FILE.name})")

    # Load optional external map
    external_map_path = Path(args.map) if args.map else None
    external_map = load_map(external_map_path) if external_map_path else {}
    if external_map:
        print(f"Merged external map: {len(external_map)} entr{'y' if len(external_map)==1 else 'ies'} from {external_map_path}")

    # Harvest from existing outputs (if any)
    harvested_map = {}
    if not args.no_harvest and OUTPUT_DIR.exists():
        out_files = sorted(OUTPUT_DIR.glob(GLOB_PATTERN))
        if out_files:
            col_query_lang = "query_" + TARGET_LANG
            harvested_map = harvest_from_outputs(out_files, col_query_lang)
            if harvested_map:
                print(f"Harvested {len(harvested_map)} entr{'y' if len(harvested_map)==1 else 'ies'} from existing outputs in {OUTPUT_DIR}")

    # Merge: cache -> +external -> +harvested
    qmap = merge_maps(cache_map, [external_map, harvested_map])

    missing_in_cache = sorted(q for q in unique_queries if q not in cache_map)
    if missing_in_cache:
        print(f"\nQueries missing from cache ({MAP_FILE.name}): {len(missing_in_cache)}")
        for q in missing_in_cache:
            print(f"  - {q}")
    else:
        print(f"\nAll unique queries already exist in cache ({MAP_FILE.name}).")

    if IDENTITY_LANG:
        print(f"Mode: identity injection (no AWS Translate for '{TARGET_LANG}').")
        for q in unique_queries:
            if q not in qmap:
                qmap[q] = q
    else:
        # If user insists not to translate, enforce presence
        if args.fail_if_missing:
            missing = [q for q in unique_queries if q not in qmap]
            if missing:
                print(f"Missing {len(missing)} translations and --fail-if-missing set. Aborting without calling AWS Translate.")
                # still save what we have so the user can inspect/fix
                save_map(MAP_FILE, qmap)
                raise SystemExit(2)
        else:
            qmap = translate_unique(unique_queries, qmap)

    save_map(MAP_FILE, qmap)
    print(f"Saved map with {len(qmap)} entries → {MAP_FILE}")

    # Process files using the map
    print(f"\nProcessing {len(files)} file(s) from {INPUT_DIR}")
    print(f"Writing outputs to {OUTPUT_DIR}\n")

    for i, in_path in enumerate(files, 1):
        out_path = OUTPUT_DIR / in_path.name
        print(f"[{i}/{len(files)}] {in_path.name} -> {out_path.name}")
        process_file(in_path, out_path, qmap)

    print("\nDone.")

if __name__ == "__main__":
    main()
