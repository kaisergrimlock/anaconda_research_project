#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


# ==============================
# Repo imports
# ==============================
THIS_FILE = Path(__file__).resolve()

# Works when the script is inside the project OR when run from the project root.
CWD = Path.cwd().resolve()
if (CWD / "retrieved").exists() and (CWD / "scripts").exists():
    PROJECT_ROOT = CWD
else:
    PROJECT_ROOT = THIS_FILE.parents[2] if len(THIS_FILE.parents) > 2 else THIS_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.csv_helpers import bump_field_limit
except Exception:
    bump_field_limit = None

try:
    from scripts.helper import allow_huge_csv_fields
except Exception:
    allow_huge_csv_fields = None


# ==============================
# Config
# ==============================
REGION = "ap-southeast-2"

# Multiple languages are supported.
# Example:
# TARGET_LANGS = ["vi", "ru"]
TARGET_LANGS = ["raw", "eng", "ru", "ar", "vi", "th", "sw", "he", "zh", "fr", "hi", "ga"]

TRECDL_YEAR = "2021"

# Judged part files to inject into:
# retrieved/trec_dl_2021/judged/all_topics_trecdl_2021_part1.csv
INPUT_PART_DIR = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/judged")
PART_PATTERN = f"all_topics_trecdl_{TRECDL_YEAR}_part{{n}}.csv"
START_PART = 1
END_PART = 6

# Query source created earlier.
# Preferred:
#   retrieved/queries/variations/trec_dl2021_query_variations.csv
#
# Columns expected:
#   qid,query,query_variation
#
# If query_variation exists and is non-empty, the script uses query_variation.
# Otherwise it falls back to query.
QUERY_VARIATION_CSV = Path(f"retrieved/queries/variations/trec_dl{TRECDL_YEAR}_query_variations.csv")
QUERY_CSV = Path(f"retrieved/queries/trec_dl{TRECDL_YEAR}.csv")

# Translation cache
CACHE_DIR = Path(f"retrieved/queries/translate_cache/trec_dl{TRECDL_YEAR}")

# Injection settings
SEED = 42
INJECT_COUNT = 1
INJECT_PROB = 1.0

IDENTITY_LANGS = {"eng", "en", "raw"}

if bump_field_limit:
    bump_field_limit()
elif allow_huge_csv_fields:
    allow_huge_csv_fields()

rng = random.Random(SEED)


# ==============================
# Basic helpers
# ==============================
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def normalize_lang(lang: str) -> str:
    return normalize_text(lang).lower()


def sort_qid(qid: str):
    qid = str(qid)
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def iter_part_files(input_part_dir: Path, year: str, start: int, end: int):
    pattern = f"all_topics_trecdl_{year}_part{{n}}.csv"

    for n in range(start, end + 1):
        path = input_part_dir / pattern.format(n=n)

        if path.exists():
            yield path
        else:
            print(f"[WARN] Missing part file: {path}")


# ==============================
# Query variation source
# ==============================
def read_query_variations(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Read query variation source.

    Required columns:
      qid, query, query_variation

    IMPORTANT:
      This function uses query_variation only.
      It does NOT fall back to the original query.

    Returns:
      qid -> {
        "query": original query,
        "variant": query_variation
      }
    """
    if not path.exists():
        raise FileNotFoundError(f"Query variation CSV not found: {path}")

    qid_map: Dict[str, Dict[str, str]] = {}
    skipped_empty_variation = 0

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)

        if not reader.fieldnames:
            raise RuntimeError(f"Query variation source has no header: {path}")

        required = ["qid", "query", "query_variation"]
        missing = [col for col in required if col not in reader.fieldnames]

        if missing:
            raise RuntimeError(
                f"Query variation source is missing required columns {missing}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            qid = normalize_text(row.get("qid", ""))
            query = normalize_text(row.get("query", ""))
            variation = normalize_text(row.get("query_variation", ""))

            if not qid or not query:
                continue

            if not variation:
                skipped_empty_variation += 1
                continue

            qid_map[qid] = {
                "query": query,
                "variant": variation,
            }

    if skipped_empty_variation:
        print(
            f"[WARN] Skipped {skipped_empty_variation} row(s) with empty query_variation. "
            f"Run your query variation generator again if these should be included."
        )

    return qid_map


def pick_query_source(year: str, explicit_path: Path | None = None) -> Path:
    if explicit_path:
        return explicit_path

    return Path(f"retrieved/queries/variations/trec_dl{year}_query_variations.csv")


# ==============================
# Translation cache
# ==============================
def load_map(path: Path) -> Dict[str, str]:
    """
    Translation cache format:
      source_text,translated

    Also accepts old cache format:
      query,translated
    """
    mapping: Dict[str, str] = {}

    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)

        if not reader.fieldnames:
            return mapping

        source_col = "source_text" if "source_text" in reader.fieldnames else "query"

        if source_col not in reader.fieldnames or "translated" not in reader.fieldnames:
            print(f"[WARN] Ignoring cache with unexpected columns: {path}")
            print(f"[WARN] Found columns: {reader.fieldnames}")
            return mapping

        for row in reader:
            source = normalize_text(row.get(source_col, ""))
            translated = normalize_text(row.get("translated", ""))

            if source and translated:
                mapping[source] = translated

    return mapping


def save_map(path: Path, mapping: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_text", "translated"])
        writer.writeheader()

        for source in sorted(mapping.keys()):
            writer.writerow({
                "source_text": source,
                "translated": mapping[source],
            })


def make_translate_client(region: str):
    import boto3
    return boto3.client("translate", region_name=region)


def translate_one(client, text: str, lang: str) -> str:
    if lang in IDENTITY_LANGS:
        return text

    response = client.translate_text(
        Text=text,
        SourceLanguageCode="auto",
        TargetLanguageCode=lang,
    )

    return normalize_text(response["TranslatedText"])


def translate_unique(
    source_texts: Iterable[str],
    lang: str,
    cache_map: Dict[str, str],
    region: str,
    fail_if_missing: bool = False,
) -> Dict[str, str]:
    lang = normalize_lang(lang)

    missing = [text for text in source_texts if text and text not in cache_map]

    if not missing:
        print(f"[{lang}] No missing translations.")
        return cache_map

    if fail_if_missing:
        print(f"[{lang}] Missing {len(missing)} translations and --fail-if-missing is set.")
        for text in missing:
            print(f"  - {text}")
        raise SystemExit(2)

    if lang in IDENTITY_LANGS:
        print(f"[{lang}] Identity mode. Copying source text.")
        for text in missing:
            cache_map[text] = text
        return cache_map

    print(f"[{lang}] Translating {len(missing)} missing query variation(s)...")

    client = make_translate_client(region)

    for i, text in enumerate(missing, start=1):
        cache_map[text] = translate_one(client, text, lang)

        if i % 50 == 0 or i == len(missing):
            print(f"[{lang}] translated {i}/{len(missing)}")

    return cache_map


# ==============================
# Injection helpers
# ==============================
def find_between_word_positions(text: str) -> List[int]:
    positions: List[int] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i].isspace():
            j = i

            while j < n and text[j].isspace():
                j += 1

            if i > 0 and j < n and not text[i - 1].isspace() and not text[j].isspace():
                positions.append(j)

            i = j
        else:
            i += 1

    return positions


def inject_once(text: str, snippet: str) -> str:
    if not snippet.strip():
        return text

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


# ==============================
# Part processing
# ==============================
def process_part_file(
    in_path: Path,
    out_path: Path,
    qid_map: Dict[str, Dict[str, str]],
    translated_by_source: Dict[str, str],
    lang: str,
    inject_count: int,
    inject_prob: float,
) -> None:
    col_query_lang = f"query_{lang}"
    col_query_variant = "query_variation"
    col_injected = "passage_injected"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8-sig", newline="") as fin, \
         out_path.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)

        if not reader.fieldnames:
            raise RuntimeError(f"No header found in input part file: {in_path}")

        required = ["qid", "query", "passage"]

        missing = [col for col in required if col not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"{in_path.name} missing required columns {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        fieldnames = list(reader.fieldnames)

        if col_query_variant not in fieldnames:
            fieldnames.append(col_query_variant)

        if col_query_lang not in fieldnames:
            fieldnames.append(col_query_lang)

        if col_injected not in fieldnames:
            fieldnames.append(col_injected)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        missing_qids = 0

        for row in reader:
            qid = normalize_text(row.get("qid", ""))
            passage = row.get("passage", "") or ""

            query_info = qid_map.get(qid)

            if not query_info:
                # Do not inject the original query as fallback.
                missing_qids += 1
                source_variant = ""
                translated_variant = ""
                passage_injected = passage
            else:
                source_variant = query_info["variant"]
                translated_variant = translated_by_source.get(source_variant, source_variant)

                passage_injected = inject_n(
                    text=passage,
                    snippet=translated_variant,
                    n=inject_count,
                    prob=inject_prob,
                )

            row[col_query_variant] = source_variant
            row[col_query_lang] = translated_variant
            row[col_injected] = passage_injected

            valid_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(valid_row)

    if missing_qids:
        print(
            f"[WARN] {in_path.name}: {missing_qids} qid(s) had no non-empty query_variation; "
            f"left passage_injected unchanged for those rows."
        )


def run_for_language(
    lang: str,
    year: str,
    input_part_dir: Path,
    qid_map: Dict[str, Dict[str, str]],
    output_suffix: str,
    region: str,
    inject_count: int,
    inject_prob: float,
    fail_if_missing: bool,
) -> None:
    lang = normalize_lang(lang)

    source_texts = sorted({info["variant"] for info in qid_map.values() if info.get("variant")})

    cache_file = Path(f"retrieved/queries/translate_cache/trec_dl{year}") / f"query_variation_map_{lang}.csv"
    cache_map = load_map(cache_file)

    print("\n" + "=" * 70)
    print(f"[LANG] {lang}")
    print(f"[CACHE] {cache_file}")
    print(f"[CACHE] Existing entries: {len(cache_map)}")

    translated_by_source = translate_unique(
        source_texts=source_texts,
        lang=lang,
        cache_map=cache_map,
        region=region,
        fail_if_missing=fail_if_missing,
    )

    save_map(cache_file, translated_by_source)
    print(f"[CACHE] Saved entries: {len(translated_by_source)}")

    # Output folder change requested:
    #   vi      -> vi_var
    #   ru      -> ru_var
    #   eng     -> eng_var
    output_dir = Path(f"retrieved/trec_dl_{year}/{lang}{output_suffix}")

    part_files = list(iter_part_files(
        input_part_dir=input_part_dir,
        year=year,
        start=START_PART,
        end=END_PART,
    ))

    if not part_files:
        raise SystemExit(f"No part files found in {input_part_dir}")

    print(f"[INPUT PARTS] {input_part_dir}")
    print(f"[OUTPUT DIR] {output_dir}")
    print(f"[PARTS] {len(part_files)}")

    for i, in_path in enumerate(part_files, start=1):
        out_path = output_dir / in_path.name
        print(f"[{lang}] [{i}/{len(part_files)}] {in_path.name} -> {out_path}")
        process_part_file(
            in_path=in_path,
            out_path=out_path,
            qid_map=qid_map,
            translated_by_source=translated_by_source,
            lang=lang,
            inject_count=inject_count,
            inject_prob=inject_prob,
        )


# ==============================
# Args / main
# ==============================
def parse_langs(items: List[str] | None) -> List[str]:
    if not items:
        return [normalize_lang(lang) for lang in TARGET_LANGS]

    langs: List[str] = []

    for item in items:
        for part in item.split(","):
            lang = normalize_lang(part)

            if lang:
                langs.append(lang)

    seen = set()
    return [lang for lang in langs if not (lang in seen or seen.add(lang))]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inject translated query variations into judged TREC-DL part files. "
            "Outputs mirrored part files into retrieved/trec_dl_{year}/{lang}_var."
        )
    )

    parser.add_argument(
        "--year",
        default=TRECDL_YEAR,
        help="TREC-DL year, e.g. 2021 or 2022.",
    )

    parser.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="Target languages. Examples: --langs vi ru OR --langs vi,ru",
    )

    parser.add_argument(
        "--query-source",
        type=Path,
        default=None,
        help=(
            "Query source CSV. Default: uses "
            "retrieved/queries/variations/trec_dl{year}_query_variations.csv"
        ),
    )

    parser.add_argument(
        "--input-part-dir",
        type=Path,
        default=None,
        help="Input judged part folder. Default: retrieved/trec_dl_{year}/judged",
    )

    parser.add_argument(
        "--region",
        default=REGION,
        help="AWS Translate region.",
    )

    parser.add_argument(
        "--inject-count",
        type=int,
        default=INJECT_COUNT,
        help="How many times to inject the translated query variation into each passage.",
    )

    parser.add_argument(
        "--inject-prob",
        type=float,
        default=INJECT_PROB,
        help="Probability for each injection attempt.",
    )

    parser.add_argument(
        "--output-suffix",
        default="_var",
        help='Output folder suffix. Default: "_var", producing folders like vi_var.',
    )

    parser.add_argument(
        "--fail-if-missing",
        action="store_true",
        help="Do not call AWS Translate. Exit if cache is missing translations.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    year = str(args.year)
    langs = parse_langs(args.langs)

    query_source = pick_query_source(year, args.query_source)
    input_part_dir = args.input_part_dir or Path(f"retrieved/trec_dl_{year}/judged")

    if not input_part_dir.exists():
        raise FileNotFoundError(f"Input judged part folder not found: {input_part_dir}")

    qid_map = read_query_variations(query_source)

    if not qid_map:
        raise RuntimeError(
            f"No rows with non-empty query_variation found in query source: {query_source}"
        )

    print(f"[QUERY SOURCE] {query_source}")
    print(f"[QUERY SOURCE ROWS] {len(qid_map)}")
    print(f"[LANGS] {langs}")
    print(f"[INPUT PART DIR] {input_part_dir}")

    for lang in langs:
        run_for_language(
            lang=lang,
            year=year,
            input_part_dir=input_part_dir,
            qid_map=qid_map,
            output_suffix=args.output_suffix,
            region=args.region,
            inject_count=args.inject_count,
            inject_prob=args.inject_prob,
            fail_if_missing=args.fail_if_missing,
        )

    print("\n[DONE] All languages processed.")


if __name__ == "__main__":
    main()
