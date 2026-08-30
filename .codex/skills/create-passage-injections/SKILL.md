---
name: create-passage-injections
description: Create controlled passage-injection variants from retrieved TREC DL CSVs in this repository. Use for multilingual query insertion, placement/repetition variants, synonyms, leetspeak, dates, ASCII/word art, or distracting text; do not use for retrieval, Bedrock relevance judging, or report generation.
---

# Create Passage Injections

Work from the repository root. Read `docs/REPOSITORY_OVERVIEW.md`, then inspect `scripts/injector/inject_lang.py` and a representative input CSV before running anything.

## Select a variant

Use the unified `inject_lang.py` CLI for language-related variants:

- `query` translates complete queries and supports `none`, `first`, `last`, `word`, `sentence`, and `brackets` placement. Multiple languages can be injected sequentially or alternated word-by-word.
- `words` translates and injects individual query words at word or sentence boundaries.
- `translate` adds translated query columns and can also translate passages.
- `synonym-cache` builds/resumes the Bedrock-generated synonym cache.
- `synonym-inject` translates and injects cached synonym terms.

Run `uv sync` once, then use `uv run python scripts/injector/inject_lang.py <command> --help` to inspect the current interface. Scripts such as `inject_leet_parts.py`, `inject_date.py`, `inject_ascii_box.py`, `inject_wordart.py`, and `inject_distraction.py` remain separate because they are not language workflows.

Preserve the user's requested intervention exactly: language, inserted text source, placement, repetition count, probability, and seed are experimental conditions. Add a new CLI option or internal strategy to the unified script instead of creating another language-specific script.

## Protect provenance

Read from `retrieved/trec_dl_<year>/judged/` or the explicitly requested source. Write a materially new condition to its own descriptive directory under `retrieved/trec_dl_<year>/`; do not overwrite another condition or its translation cache.

Preserve `qid`, `pid`, `query`, `passage`, and `relevance`. Add the transformed text in `passage_injected` and retain any translated-query field used to construct it. Keep the existing filename/chunk structure when processing a directory.

Translation scripts call Amazon Translate and cache mappings in `translate_cache/`. Before a call, count uncached unique queries, identify the source/target language and AWS region, and obtain authorization for the external operation if it was not already granted. LLM-generated distractions and synonym caches may call Bedrock; identify the model and estimated scope before seeking the same authorization.

## Validate the intervention

For a new or changed configuration, first process a small representative sample when the script supports a safe bounded run. Check that:

- output row count and `(qid, pid)` identity match the input;
- original `passage` text is retained;
- `passage_injected` differs only as intended;
- deterministic variants reproduce with the configured seed;
- translation-cache keys are unique and non-empty; and
- output encoding preserves multilingual text.

Then run the requested scope and report the variant definition, source and destination paths, row/file counts, cache additions, and any skipped or malformed rows. Stop before labeling unless separately requested.
