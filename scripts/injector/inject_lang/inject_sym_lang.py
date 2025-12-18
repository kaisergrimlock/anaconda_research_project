#!/usr/bin/env python3
from __future__ import annotations

import csv
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import boto3

# ===============================================================
# Path setup
# ===============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.injector.helper import allow_huge_csv_fields

# ===============================================================
# Config
# ===============================================================
TREC_DL_YEAR = "2022"

LANG = "ru"                 # target language for synonyms translation (eng/en => identity)
SYN_SOURCE_LANG = "en"       # only used when LANG is not identity

SEED = 42
GLOB_PATTERN = "*.csv"

INPUT_DIR = PROJECT_ROOT / f"retrieved/trec_dl_{TREC_DL_YEAR}/judged"
OUTPUT_DIR = PROJECT_ROOT / f"retrieved/trec_dl_{TREC_DL_YEAR}/syn_{LANG}"

CACHE_DIR = PROJECT_ROOT / f"retrieved/trec_dl_{TREC_DL_YEAR}/translate_cache"
SYN_CACHE_DIR = CACHE_DIR / "sym_lang"
SYN_MAP_FILE = SYN_CACHE_DIR / f"symnonyms_map_{LANG}.csv"

QUERY_COL = "query"
PASSAGE_COL = "passage"

QUERY_SYN_COL = f"query_syn_{LANG}"
PASSAGE_INJECTED_COL = "passage_injected"

MAX_SYNONYMS_TO_INJECT = 8
INJECT_PROB = 1.0

AWS_REGION_TRANSLATE = "ap-southeast-2"

# ===============================================================

allow_huge_csv_fields()
rng = random.Random(SEED)

# IMPORTANT: treat eng/en as identity (no translation)
IDENTITY_LANG = LANG.lower() in {"raw", "eng", "en"}

translate_client = None
if not IDENTITY_LANG:
    translate_client = boto3.client("translate", region_name=AWS_REGION_TRANSLATE)


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def split_synonyms(s: str) -> List[str]:
    out = [normalize_spaces(x) for x in (s or "").split(",")]
    return [x for x in out if x]


def translate_text_one(text: str) -> str:
    """
    If LANG is identity (eng/en), returns text unchanged.
    Otherwise translates from SYN_SOURCE_LANG -> LANG.
    """
    if not text:
        return ""
    if IDENTITY_LANG:
        return text

    try:
        resp = translate_client.translate_text(
            Text=text,
            SourceLanguageCode=SYN_SOURCE_LANG,
            TargetLanguageCode=LANG,
        )
        return resp["TranslatedText"]
    except Exception as e:
        print(f"[WARN] Translate failed ({SYN_SOURCE_LANG}->{LANG}) for '{text}': {e}")
        return text


def inject_terms_into_passage(passage: str, terms: List[str]) -> str:
    if not passage or not terms:
        return passage
    words = passage.split()
    if not words:
        return passage

    for term in terms:
        if not term:
            continue
        if rng.random() > INJECT_PROB:
            continue
        idx = rng.randint(0, len(words))
        words.insert(idx, term)

    return " ".join(words)


def load_syn_map(path: Path) -> Dict[str, Tuple[str, str]]:
    """
    query -> (keyword, symnonyms_string)
    """
    m: Dict[str, Tuple[str, str]] = {}
    if not path.exists():
        raise SystemExit(f"Synonyms cache not found: {path} (run build_syn_cache.py first)")
    with path.open("r", newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            q = (row.get("query") or "").strip()
            k = (row.get("keyword") or "").strip()
            s = (row.get("symnonyms") or "").strip()
            if q:
                m[q] = (k, s)
    return m


def translate_syn_list(
    syn_list: List[str],
    synonym_translate_cache: Dict[str, str],
) -> List[str]:
    """
    If LANG is identity (eng/en), returns the cache synonyms as-is.
    Otherwise translates each synonym term.
    """
    if IDENTITY_LANG:
        # no translate, no cache required, but keep behavior consistent
        return [syn.strip() for syn in syn_list if (syn or "").strip()]

    out: List[str] = []
    for syn in syn_list:
        syn = (syn or "").strip()
        if not syn:
            continue

        if syn in synonym_translate_cache:
            out.append(synonym_translate_cache[syn])
            continue

        syn_t = translate_text_one(syn)
        synonym_translate_cache[syn] = syn_t
        out.append(syn_t)

    return out


def process_file(
    in_path: Path,
    out_path: Path,
    syn_map: Dict[str, Tuple[str, str]],
    synonym_translate_cache: Dict[str, str],
) -> None:
    with in_path.open("r", newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])

        if QUERY_SYN_COL not in fieldnames:
            fieldnames.append(QUERY_SYN_COL)
        if PASSAGE_INJECTED_COL not in fieldnames:
            fieldnames.append(PASSAGE_INJECTED_COL)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            q = (row.get(QUERY_COL) or "").strip()
            p = (row.get(PASSAGE_COL) or "").strip()

            _kw, syns_str = syn_map.get(q, ("", ""))
            syn_list = split_synonyms(syns_str)[:MAX_SYNONYMS_TO_INJECT]

            # KEY CHANGE: if LANG is eng/en, this returns original cache synonyms unchanged
            terms = translate_syn_list(syn_list, synonym_translate_cache)

            row[QUERY_SYN_COL] = ", ".join(terms)
            row[PASSAGE_INJECTED_COL] = inject_terms_into_passage(p, terms)

            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")

    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit(f"No CSV files found in: {INPUT_DIR} (pattern: {GLOB_PATTERN})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    syn_map = load_syn_map(SYN_MAP_FILE)
    print(f"[LOAD] Syn cache entries: {len(syn_map)} from {SYN_MAP_FILE}")
    print(f"[MODE] LANG={LANG}  identity={IDENTITY_LANG}")

    print(f"[RUN] Injecting into {len(files)} file(s) → {OUTPUT_DIR}")

    synonym_translate_cache: Dict[str, str] = {}

    for i, in_path in enumerate(files, 1):
        out_path = OUTPUT_DIR / in_path.name
        print(f"[{i}/{len(files)}] {in_path.name} -> {out_path.name}")
        process_file(in_path, out_path, syn_map, synonym_translate_cache)

    print("[DONE]")


if __name__ == "__main__":
    main()
