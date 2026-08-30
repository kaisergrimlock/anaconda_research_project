#!/usr/bin/env python3
"""Create multilingual query, word, passage, and synonym variants for TREC DL."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import boto3
    from botocore.config import Config
except ImportError:  # Identity-language and cached runs do not need AWS packages.
    boto3 = None
    Config = None

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

IDENTITY_LANGUAGES = {"raw", "eng", "en"}
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
QUERY_COLUMN = "query"
PASSAGE_COLUMN = "passage"
INJECTED_COLUMN = "passage_injected"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name].astype(str)
    return pd.Series("", index=frame.index, dtype=str)


class _Translator:
    def __init__(self, region: str) -> None:
        self.region = region
        self._client: Any | None = None

    def translate(self, text: str, *, source: str, target: str) -> str:
        if not text or _is_identity_language(target):
            return text
        if boto3 is None:
            raise RuntimeError(
                "boto3 is required for uncached translations. "
                "Install it or rerun with --fail-if-missing."
            )
        if self._client is None:
            self._client = boto3.client("translate", region_name=self.region)
        response = self._client.translate_text(
            Text=text,
            SourceLanguageCode=source,
            TargetLanguageCode=target,
        )
        return str(response["TranslatedText"])


def _is_identity_language(language: str) -> bool:
    return language.lower() in IDENTITY_LANGUAGES


def _data_root(year: str) -> Path:
    return PROJECT_ROOT / "retrieved" / f"trec_dl_{year}"


def _resolve_input_dir(args: argparse.Namespace) -> Path:
    return args.input_dir or _data_root(args.year) / "judged"


def _resolve_cache_dir(args: argparse.Namespace) -> Path:
    return args.cache_dir or _data_root(args.year) / "translate_cache"


def _input_files(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No CSV files found in {input_dir} with pattern {pattern!r}")
    return files


def _check_output_targets(
    files: Iterable[Path], output_dir: Path, *, overwrite: bool
) -> None:
    conflicts = [
        output_dir / path.name for path in files if (output_dir / path.name).exists()
    ]
    if conflicts and not overwrite:
        preview = "\n".join(f"  {path}" for path in conflicts[:5])
        extra = f"\n  ... and {len(conflicts) - 5} more" if len(conflicts) > 5 else ""
        raise SystemExit(
            "Refusing to overwrite existing outputs. "
            "Pass --overwrite to replace them:\n"
            f"{preview}{extra}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_map(path: Path, *, key_field: str) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = _read_csv(path)
    values: dict[str, str] = {}
    for key, translated in zip(
        _column(frame, key_field), _column(frame, "translated"), strict=True
    ):
        if (key := key.strip()) and (translated := translated.strip()):
            values.setdefault(key, translated)
    return values


def _save_map(path: Path, values: dict[str, str], *, key_field: str) -> None:
    keys = sorted(values)
    _write_csv(
        pd.DataFrame({key_field: keys, "translated": [values[key] for key in keys]}),
        path,
    )


def _merge_maps(*maps: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for values in maps:
        for key, translated in values.items():
            if key and translated:
                merged.setdefault(key, translated)
    return merged


def _parse_path_assignments(values: Sequence[str]) -> dict[str, Path]:
    assignments: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise SystemExit(f"Expected LANGUAGE=PATH, received {value!r}")
        assignments[name.strip()] = Path(raw_path.strip())
    return assignments


def _parse_count_assignments(values: Sequence[str]) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for value in values:
        name, separator, raw_count = value.partition("=")
        if not separator or not name.strip():
            raise SystemExit(f"Expected LANGUAGE=COUNT, received {value!r}")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise SystemExit(f"Invalid injection count in {value!r}") from error
        if count < 0:
            raise SystemExit("Injection counts must be non-negative")
        assignments[name.strip()] = count
    return assignments


def _fill_translation_map(
    items: Iterable[str],
    values: dict[str, str],
    *,
    key_field: str,
    path: Path,
    source_language: str,
    target_language: str,
    translator: _Translator,
    fail_if_missing: bool,
) -> dict[str, str]:
    unique_items = sorted({item.strip() for item in items if item and item.strip()})
    missing = [item for item in unique_items if item not in values]
    if _is_identity_language(target_language):
        values.update({item: item for item in missing})
        _save_map(path, values, key_field=key_field)
        return values
    if missing and fail_if_missing:
        _save_map(path, values, key_field=key_field)
        raise SystemExit(
            f"Missing {len(missing)} {key_field} translation(s) for "
            f"{target_language}; no AWS call was made."
        )
    if missing:
        print(
            f"[{target_language}] Translating {len(missing)} missing "
            f"{key_field} value(s)"
        )
    for index, item in enumerate(missing, 1):
        values[item] = translator.translate(
            item,
            source=source_language,
            target=target_language,
        )
        if index % 100 == 0 or index == len(missing):
            _save_map(path, values, key_field=key_field)
            print(f"[{target_language}] translated {index}/{len(missing)}")
    _save_map(path, values, key_field=key_field)
    return values


def _collect_column_values(files: Iterable[Path], column: str) -> set[str]:
    return {
        value
        for path in files
        for value in _column(_read_csv(path), column).str.strip()
        if value
    }


def _collect_query_words(
    files: Iterable[Path], *, whitespace: bool = False
) -> set[str]:
    words: set[str] = set()
    for query in _collect_column_values(files, QUERY_COLUMN):
        words.update(query.split() if whitespace else WORD_PATTERN.findall(query))
    return {word for word in words if word}


def _harvest_query_map(files: Iterable[Path], language: str) -> dict[str, str]:
    query_column = f"query_{language}"
    harvested: dict[str, str] = {}
    for path in files:
        if not path.exists():
            continue
        frame = _read_csv(path)
        if query_column not in frame:
            continue
        for query, translated in zip(
            _column(frame, QUERY_COLUMN).str.strip(),
            _column(frame, query_column).str.strip(),
            strict=True,
        ):
            if query and translated:
                harvested.setdefault(query, translated)
    return harvested


def _word_boundaries(text: str) -> list[int]:
    boundaries: list[int] = []
    index = 0
    while index < len(text):
        if not text[index].isspace():
            index += 1
            continue
        end = index
        while end < len(text) and text[end].isspace():
            end += 1
        if index > 0 and end < len(text):
            boundaries.append(end)
        index = end
    return boundaries


def _sentence_boundaries(text: str) -> list[int]:
    boundaries: list[int] = []
    for index, character in enumerate(text):
        if character not in ".?!":
            continue
        end = index + 1
        while end < len(text) and text[end].isspace():
            end += 1
        if end < len(text):
            boundaries.append(end)
    return boundaries


def _inject_once(
    text: str,
    snippet: str,
    *,
    placement: str,
    rng: random.Random,
) -> str:
    clean_text = text.strip()
    clean_snippet = snippet.strip()
    if not clean_snippet:
        return text
    if not clean_text:
        return clean_snippet
    if placement == "first":
        return f"{clean_snippet}. {clean_text}"
    if placement == "last":
        return f"{clean_text} {clean_snippet}"

    boundaries = (
        _sentence_boundaries(text)
        if placement == "sentence"
        else _word_boundaries(text)
    )
    inserted = f"({clean_snippet})" if placement == "brackets" else clean_snippet
    if not boundaries:
        return f"{clean_text} {inserted}"
    index = rng.choice(boundaries)
    return f"{text[:index]}{inserted} {text[index:]}"


def _inject_repeated(
    text: str,
    snippets: Iterable[str],
    *,
    count: int,
    probability: float,
    placement: str,
    rng: random.Random,
) -> str:
    result = text
    for snippet in snippets:
        for _ in range(count):
            if rng.random() <= probability:
                result = _inject_once(result, snippet, placement=placement, rng=rng)
    return result


def _validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise SystemExit("--probability must be between 0 and 1")


def _default_query_suffix(
    languages: Sequence[str], placement: str, combine: str, count: int
) -> str:
    joined = "_".join(languages)
    if combine == "alternating":
        return f"{joined}_between"
    if placement == "none":
        return f"{joined}_trans_q"
    if placement == "word":
        return joined if count == 1 else f"{joined}_mult_{count}"
    return f"{joined}_{placement}"


def _run_query(args: argparse.Namespace) -> int:
    languages = list(dict.fromkeys(args.language))
    if not languages:
        raise SystemExit("At least one --language is required")
    if args.combine == "alternating" and len(languages) < 2:
        raise SystemExit("Alternating mode requires at least two languages")
    if args.combine == "alternating" and args.placement == "none":
        raise SystemExit("Alternating mode requires an injection placement")
    _validate_probability(args.probability)

    input_dir = _resolve_input_dir(args)
    cache_dir = _resolve_cache_dir(args)
    suffix = _default_query_suffix(languages, args.placement, args.combine, args.count)
    output_dir = args.output_dir or _data_root(args.year) / suffix
    files = _input_files(input_dir, args.glob)
    _check_output_targets(files, output_dir, overwrite=args.overwrite)

    external_paths = _parse_path_assignments(args.map)
    language_counts = _parse_count_assignments(args.language_count)
    translator = _Translator(args.region)
    map_kind = "word" if args.combine == "alternating" else "query"
    items = (
        _collect_query_words(files, whitespace=True)
        if args.combine == "alternating"
        else _collect_column_values(files, QUERY_COLUMN)
    )
    maps: dict[str, dict[str, str]] = {}
    output_files = [output_dir / path.name for path in files]
    for language in languages:
        map_path = cache_dir / f"{map_kind}_map_{language}.csv"
        existing = _load_map(map_path, key_field=map_kind)
        external = (
            _load_map(external_paths[language], key_field=map_kind)
            if language in external_paths
            else {}
        )
        harvested = (
            _harvest_query_map(output_files, language)
            if not args.no_harvest and map_kind == "query"
            else {}
        )
        values = _merge_maps(existing, external, harvested)
        maps[language] = _fill_translation_map(
            items,
            values,
            key_field=map_kind,
            path=map_path,
            source_language=args.source_language,
            target_language=language,
            translator=translator,
            fail_if_missing=args.fail_if_missing,
        )

    rng = random.Random(args.seed)
    for index, input_path in enumerate(files, 1):
        output_path = output_dir / input_path.name
        frame = _read_csv(input_path)
        queries = _column(frame, QUERY_COLUMN).str.strip()
        passages = _column(frame, PASSAGE_COLUMN)
        if args.combine == "alternating":
            mixed_queries = [
                " ".join(
                    maps[languages[token_index % len(languages)]].get(token, token)
                    for token_index, token in enumerate(query.split())
                )
                for query in queries
            ]
            frame["query_mixed"] = mixed_queries
            frame[INJECTED_COLUMN] = [
                _inject_repeated(
                    passage,
                    [mixed],
                    count=args.count,
                    probability=args.probability,
                    placement=args.placement,
                    rng=rng,
                )
                for passage, mixed in zip(passages, mixed_queries, strict=True)
            ]
        else:
            translations = {
                language: [maps[language].get(query, query) for query in queries]
                for language in languages
            }
            for language, translated in translations.items():
                frame[f"query_{language}"] = translated
            if args.placement != "none":
                injected_passages = []
                for row_index, passage in enumerate(passages):
                    for language in languages:
                        passage = _inject_repeated(
                            passage,
                            [translations[language][row_index]],
                            count=language_counts.get(language, args.count),
                            probability=args.probability,
                            placement=args.placement,
                            rng=rng,
                        )
                    injected_passages.append(passage)
                frame[INJECTED_COLUMN] = injected_passages
        _write_csv(frame, output_path)
        print(f"[{index}/{len(files)}] {input_path.name} -> {output_path}")
    return 0


def _run_words(args: argparse.Namespace) -> int:
    _validate_probability(args.probability)
    input_dir = _resolve_input_dir(args)
    cache_dir = _resolve_cache_dir(args)
    output_dir = args.output_dir or _data_root(args.year) / (
        f"{args.language}_word"
        if args.placement == "word"
        else f"{args.language}_word_sentence"
    )
    files = _input_files(input_dir, args.glob)
    _check_output_targets(files, output_dir, overwrite=args.overwrite)
    words = _collect_query_words(files)
    map_path = cache_dir / f"word_map_{args.language}.csv"
    values = _load_map(map_path, key_field="word")
    values = _fill_translation_map(
        words,
        values,
        key_field="word",
        path=map_path,
        source_language="en",
        target_language=args.language,
        translator=_Translator(args.region),
        fail_if_missing=args.fail_if_missing,
    )

    rng = random.Random(args.seed)
    translated_column = f"query_{args.language}"
    for index, input_path in enumerate(files, 1):
        output_path = output_dir / input_path.name
        frame = _read_csv(input_path)
        translated_words = [
            [values.get(word, word) for word in WORD_PATTERN.findall(query)]
            for query in _column(frame, QUERY_COLUMN)
        ]
        frame[translated_column] = [" ".join(words) for words in translated_words]
        frame[INJECTED_COLUMN] = [
            _inject_repeated(
                passage,
                words,
                count=args.count,
                probability=args.probability,
                placement=args.placement,
                rng=rng,
            )
            for passage, words in zip(
                _column(frame, PASSAGE_COLUMN), translated_words, strict=True
            )
        ]
        _write_csv(frame, output_path)
        print(f"[{index}/{len(files)}] {input_path.name} -> {output_path}")
    return 0


def _run_translate(args: argparse.Namespace) -> int:
    input_dir = _resolve_input_dir(args)
    cache_dir = _resolve_cache_dir(args)
    suffix = f"{args.language}_trans_p" if args.passages else f"{args.language}_trans_q"
    output_dir = args.output_dir or _data_root(args.year) / suffix
    files = _input_files(input_dir, args.glob)
    _check_output_targets(files, output_dir, overwrite=args.overwrite)
    translator = _Translator(args.region)

    queries = _collect_column_values(files, QUERY_COLUMN)
    query_path = cache_dir / f"query_map_{args.language}.csv"
    query_map = _fill_translation_map(
        queries,
        _load_map(query_path, key_field="query"),
        key_field="query",
        path=query_path,
        source_language=args.source_language,
        target_language=args.language,
        translator=translator,
        fail_if_missing=args.fail_if_missing,
    )
    passage_map: dict[str, str] = {}
    if args.passages:
        passages = {
            passage
            for passage in _collect_column_values(files, PASSAGE_COLUMN)
            if len(passage) <= args.max_passage_chars
        }
        passage_path = cache_dir / f"passage_map_{args.language}.csv"
        passage_map = _fill_translation_map(
            passages,
            _load_map(passage_path, key_field="passage"),
            key_field="passage",
            path=passage_path,
            source_language=args.source_language,
            target_language=args.language,
            translator=translator,
            fail_if_missing=args.fail_if_missing,
        )

    query_column = f"query_{args.language}"
    passage_column = f"passage_{args.language}"
    for index, input_path in enumerate(files, 1):
        output_path = output_dir / input_path.name
        frame = _read_csv(input_path)
        queries = _column(frame, QUERY_COLUMN).str.strip()
        frame[query_column] = [query_map.get(query, query) for query in queries]
        if args.passages:
            passages = _column(frame, PASSAGE_COLUMN).str.strip()
            frame[passage_column] = [passage_map.get(text, "") for text in passages]
        _write_csv(frame, output_path)
        print(f"[{index}/{len(files)}] {input_path.name} -> {output_path}")
    return 0


def _render_prompt(template: str, key: str, value: str) -> str:
    sentinel = "<<<VALUE>>>"
    escaped = template.replace(f"{{{key}}}", sentinel)
    escaped = escaped.replace("{", "{{").replace("}", "}}")
    return escaped.replace(sentinel, f"{{{key}}}").format(**{key: value})


def _extract_json(text: str) -> dict[str, Any] | None:
    clean = (text or "").strip()
    if not clean:
        return None
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            return None
        blob = re.sub(
            r"(\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:",
            r'\1 "\2":',
            match.group(0).replace("“", '"').replace("”", '"').replace("’", "'"),
        )
        try:
            value = json.loads(blob)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _load_synonym_map(
    path: Path, *, required: bool = False
) -> dict[str, tuple[str, str]]:
    if not path.exists():
        if required:
            raise SystemExit(f"Synonym cache not found: {path}")
        return {}
    frame = _read_csv(path)
    synonym_column = "synonyms" if "synonyms" in frame else "symnonyms"
    return {
        query: (keyword, synonyms)
        for query, keyword, synonyms in zip(
            _column(frame, "query").str.strip(),
            _column(frame, "keyword").str.strip(),
            _column(frame, synonym_column).str.strip(),
            strict=True,
        )
        if query
    }


def _save_synonym_map(path: Path, values: dict[str, tuple[str, str]]) -> None:
    rows = [
        {"query": query, "keyword": values[query][0], "symnonyms": values[query][1]}
        for query in sorted(values)
    ]
    _write_csv(pd.DataFrame(rows, columns=["query", "keyword", "symnonyms"]), path)


def _bedrock_text(client: Any, model: str, prompt: str) -> str:
    response = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0.0, "topP": 1.0},
        system=[{"text": "You are an SEO assistant. Return ONLY valid JSON."}],
    )
    blocks = response.get("output", {}).get("message", {}).get("content", []) or []
    return "".join(block.get("text", "") for block in blocks).strip()


def _run_synonym_cache(args: argparse.Namespace) -> int:
    if boto3 is None or Config is None:
        raise SystemExit("boto3 and botocore are required to build a synonym cache")
    input_dir = _resolve_input_dir(args)
    cache_dir = _resolve_cache_dir(args) / "sym_lang"
    files = _input_files(input_dir, args.glob)
    map_path = args.synonym_map or cache_dir / f"symnonyms_map_{args.cache_name}.csv"
    values = _load_synonym_map(map_path)
    queries = sorted(_collect_column_values(files, QUERY_COLUMN))
    query_template = args.query_prompt.read_text(encoding="utf-8")
    word_template = args.word_prompt.read_text(encoding="utf-8")
    client = boto3.client(
        "bedrock-runtime",
        config=Config(
            region_name=args.region,
            connect_timeout=10,
            read_timeout=300,
            retries={"max_attempts": 8, "mode": "standard"},
        ),
    )
    for index, query in enumerate(queries, 1):
        if query not in values:
            response = _bedrock_text(
                client, args.model, _render_prompt(query_template, "query", query)
            )
            parsed = _extract_json(response) or {}
            keyword = str(parsed.get("keyword", "") or "").strip()
            synonyms = str(
                parsed.get("synonyms", parsed.get("symnonyms", "")) or ""
            ).strip()
            values[query] = (keyword, synonyms)
        else:
            keyword, synonyms = values[query]
            if keyword and not synonyms:
                response = _bedrock_text(
                    client,
                    args.model,
                    _render_prompt(word_template, "keyword", keyword),
                )
                parsed = _extract_json(response) or {}
                synonyms = str(
                    parsed.get("synonyms", parsed.get("symnonyms", "")) or ""
                ).strip()
                values[query] = (keyword, synonyms)
            else:
                continue
        _save_synonym_map(map_path, values)
        print(f"[{index}/{len(queries)}] cached synonyms for {query!r}")
    _save_synonym_map(map_path, values)
    print(f"Saved {len(values)} synonym entries -> {map_path}")
    return 0


def _split_synonyms(value: str) -> list[str]:
    return [
        normalized
        for item in (value or "").split(",")
        if (normalized := re.sub(r"\s+", " ", item).strip())
    ]


def _run_synonym_inject(args: argparse.Namespace) -> int:
    _validate_probability(args.probability)
    input_dir = _resolve_input_dir(args)
    cache_dir = _resolve_cache_dir(args)
    files = _input_files(input_dir, args.glob)
    output_dir = args.output_dir or _data_root(args.year) / f"syn_{args.language}"
    _check_output_targets(files, output_dir, overwrite=args.overwrite)
    synonym_path = args.synonym_map or (
        cache_dir / "sym_lang" / f"symnonyms_map_{args.cache_name or args.language}.csv"
    )
    synonyms = _load_synonym_map(synonym_path, required=True)
    terms = {
        term
        for _keyword, raw_synonyms in synonyms.values()
        for term in _split_synonyms(raw_synonyms)[: args.max_synonyms]
    }
    term_path = cache_dir / f"synonym_map_{args.language}.csv"
    term_map = _fill_translation_map(
        terms,
        _load_map(term_path, key_field="term"),
        key_field="term",
        path=term_path,
        source_language=args.source_language,
        target_language=args.language,
        translator=_Translator(args.region),
        fail_if_missing=args.fail_if_missing,
    )
    rng = random.Random(args.seed)
    synonym_column = f"query_syn_{args.language}"
    for index, input_path in enumerate(files, 1):
        output_path = output_dir / input_path.name
        frame = _read_csv(input_path)
        translated_terms = [
            [
                term_map.get(term, term)
                for term in _split_synonyms(synonyms.get(query.strip(), ("", ""))[1])[
                    : args.max_synonyms
                ]
            ]
            for query in _column(frame, QUERY_COLUMN)
        ]
        frame[synonym_column] = [", ".join(terms) for terms in translated_terms]
        frame[INJECTED_COLUMN] = [
            _inject_repeated(
                passage,
                terms,
                count=1,
                probability=args.probability,
                placement=args.placement,
                rng=rng,
            )
            for passage, terms in zip(
                _column(frame, PASSAGE_COLUMN), translated_terms, strict=True
            )
        ]
        _write_csv(frame, output_path)
        print(f"[{index}/{len(files)}] {input_path.name} -> {output_path}")
    return 0


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--year", default="2022", help="TREC DL dataset year")
    parser.add_argument("--input-dir", type=Path, help="Input CSV directory")
    parser.add_argument("--output-dir", type=Path, help="Output CSV directory")
    parser.add_argument("--cache-dir", type=Path, help="Translation cache directory")
    parser.add_argument("--glob", default="*.csv", help="Input filename pattern")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace matching output CSV files"
    )


def _add_translation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--source-language", default="auto")
    parser.add_argument(
        "--fail-if-missing",
        action="store_true",
        help="Abort rather than call AWS when a cache entry is missing",
    )


def _add_injection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--probability", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create all multilingual TREC DL passage variants from one script."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser(
        "query", help="Translate and optionally inject complete queries"
    )
    _add_data_arguments(query)
    _add_translation_arguments(query)
    _add_injection_arguments(query)
    query.add_argument("--language", action="append", required=True)
    query.add_argument(
        "--placement",
        choices=("none", "first", "last", "word", "sentence", "brackets"),
        default="word",
    )
    query.add_argument(
        "--combine", choices=("sequential", "alternating"), default="sequential"
    )
    query.add_argument(
        "--language-count",
        action="append",
        default=[],
        metavar="LANGUAGE=COUNT",
    )
    query.add_argument("--map", action="append", default=[], metavar="LANGUAGE=PATH")
    query.add_argument("--no-harvest", action="store_true")
    query.set_defaults(handler=_run_query)

    words = subparsers.add_parser(
        "words", help="Translate and inject individual query words"
    )
    _add_data_arguments(words)
    _add_translation_arguments(words)
    _add_injection_arguments(words)
    words.add_argument("--language", required=True)
    words.add_argument("--placement", choices=("word", "sentence"), default="word")
    words.set_defaults(handler=_run_words)

    translate = subparsers.add_parser(
        "translate", help="Add translated query and optional passage columns"
    )
    _add_data_arguments(translate)
    _add_translation_arguments(translate)
    translate.add_argument("--language", required=True)
    translate.add_argument("--passages", action="store_true")
    translate.add_argument("--max-passage-chars", type=int, default=10_000)
    translate.set_defaults(handler=_run_translate)

    synonym_cache = subparsers.add_parser(
        "synonym-cache", help="Build or resume a Bedrock-generated synonym cache"
    )
    _add_data_arguments(synonym_cache)
    synonym_cache.add_argument("--cache-name", required=True)
    synonym_cache.add_argument("--synonym-map", type=Path)
    synonym_cache.add_argument("--model", default="meta.llama3-8b-instruct-v1:0")
    synonym_cache.add_argument("--region", default="us-west-2")
    synonym_cache.add_argument(
        "--query-prompt", type=Path, default=PROJECT_ROOT / "prompts" / "symnonyms.txt"
    )
    synonym_cache.add_argument(
        "--word-prompt",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "symnonyms_word.txt",
    )
    synonym_cache.set_defaults(handler=_run_synonym_cache)

    synonym_inject = subparsers.add_parser(
        "synonym-inject", help="Translate and inject terms from a synonym cache"
    )
    _add_data_arguments(synonym_inject)
    _add_translation_arguments(synonym_inject)
    synonym_inject.add_argument("--language", required=True)
    synonym_inject.add_argument("--cache-name")
    synonym_inject.add_argument("--synonym-map", type=Path)
    synonym_inject.add_argument("--max-synonyms", type=int, default=8)
    synonym_inject.add_argument("--probability", type=float, default=1.0)
    synonym_inject.add_argument("--seed", type=int, default=42)
    synonym_inject.add_argument(
        "--placement",
        choices=("first", "last", "word", "sentence", "brackets"),
        default="word",
    )
    synonym_inject.set_defaults(handler=_run_synonym_inject)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected multilingual data transformation.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments without the program name. Uses ``sys.argv`` when
        omitted.

    Returns
    -------
    int
        Process exit code; zero indicates a completed transformation.
    """
    args = _build_parser().parse_args(argv)
    if getattr(args, "count", 0) < 0:
        raise SystemExit("--count must be non-negative")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
