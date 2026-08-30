# Repository overview

## Summary

This repository is a research workspace for studying the robustness of large-language-model relevance judgments. It starts with judged TREC Deep Learning passage-retrieval data, creates altered versions of passages, asks several LLMs to assign relevance labels, and compares those labels with the original human/NIST judgments.

The main intervention is **passage injection**: text related to the query is inserted into an otherwise retrieved passage. The variants include translated query text, English query text, synonyms, repeated or repositioned text, leetspeak, dates, ASCII art, and generated distracting passages. The stored data and figures indicate that the study examines whether the injected text changes an LLM judge's score, and how that effect varies by language, injection strategy, dataset year, and model.

This is primarily a collection of executable research scripts and generated artifacts, rather than an installable Python package. At the time of this overview, it contains about 6,100 tracked files, most of them CSV results and generated figures.

## End-to-end workflow

The repository implements the following loose pipeline:

1. **Retrieve TREC DL topics and passages.** Scripts under `scripts/retrieval/` use Pyserini's prebuilt MS MARCO indexes and TREC DL topics/qrels. The main retrieval script supports TREC DL 2019–2023, passage or document indexes, judged-qrel export, and BM25 top-k retrieval.
2. **Create passage variants.** Scripts under `scripts/injector/` translate or transform query text and insert it into passages. Translation mappings are cached under `retrieved/trec_dl_<year>/translate_cache/` so repeated runs do not need to translate the same queries again.
3. **Run LLM relevance judging.** Scripts under `scripts/label/` render prompts from `prompts/`, call models through Amazon Bedrock, parse 0–3 relevance scores, and write per-part and combined CSV outputs. Some variants ask for a single overall score; criterion variants collect dimensions such as contextuality, coverage, exactness, and topicality before deriving or recording relevance.
4. **Normalize model responses.** `scripts/parser/` contains conversion and cleanup utilities for model-specific outputs.
5. **Analyse agreement and robustness.** Scripts under `scripts/report/` compare LLM labels with NIST judgments and with other experimental variants. They calculate confusion matrices, mean absolute error, Cohen's kappa, Krippendorff's alpha, token-count/fertility changes, perplexity changes, confidence intervals, and Tukey HSD comparisons.
6. **Persist research artifacts.** Intermediate datasets live mainly in `retrieved/`; LLM labels and derived tables in `outputs/`; and publication-oriented charts and metric tables in `figures/`.

There is no single orchestration command. Most scripts expose experiment settings as constants near the top of each file, so a run is assembled by selecting a script and editing values such as year, language/variant, model, input directory, and output directory.

## Repository layout

| Path | Role |
| --- | --- |
| `scripts/retrieval/` | Downloads/opens TREC topics, qrels, and MS MARCO passages; performs BM25 retrieval; samples non-relevant passages; and generates synthetic passages. |
| `scripts/injector/` | Builds adversarial or diagnostic passage variants. Multilingual query/word/passage/synonym workflows share `inject_lang.py`; separate scripts cover leetspeak, dates, word art, ASCII boxes, and distractions. |
| `scripts/label/` | Runs Bedrock-hosted LLM judges, handles concurrency and batching, parses model scores, tracks token usage and estimated cost, and writes label CSVs. |
| `scripts/parser/` | Cleans or converts raw/model-specific response files into consistent tabular results. |
| `scripts/report/` | Produces summaries, agreement metrics, confusion matrices, statistical comparisons, tokenization analyses, perplexity analyses, and plots. |
| `scripts/bedrock_client.py` | Shared Bedrock client and response parsing for score-based labeling. |
| `scripts/csv_helpers.py`, `scripts/log_helpers.py` | Shared large-CSV, chunking, normalization, logging, and run-metadata helpers. |
| `prompts/` | Prompt templates for 0–3 relevance assessment, few-shot/one-shot judging, leetspeak conversion, and synonym generation. |
| `retrieved/` | Source and transformed TREC DL datasets, organized mostly by year and injection variant. This is the largest input/intermediate-data area. |
| `outputs/` | LLM labels, baseline analyses, diagnostics, tokenization tables, extracted queries, and JSONL passage exports. |
| `figures/` | Generated SVG/PNG/PDF charts and the CSV/LaTeX metric tables that support them, organized by year, model, and experiment variant. |
| `unjudged/` | Untracked-at-review-time TREC DL CSVs for unjudged data from 2019 and 2021–2023. |
| `sample_label0_judged.csv` | A small sample of human-judged, relevance-0 passages. |
| `token_usage.csv` | Run-level model token accounting and references to generated labels/logs. |
| `passage_pair_card.*`, `passage_injection_example.pdf` | Explanatory visual artifacts for passage-pair and injection examples. |

## Data conventions

The core retrieved CSV schema is:

```text
qid, query, pid, passage, relevance
```

Injected datasets generally preserve these identifiers and add fields such as a translated query and `passage_injected`. Label outputs add `llm_relevance`; criterion-based runs may also include `contextuality`, `coverage`, `exactness`, and `topicality`.

Important naming dimensions are encoded in directory and file names:

- **Dataset year:** primarily TREC DL 2021 and 2022, with some 2019 and 2023 material.
- **Language:** examples include Arabic (`ar`), French (`fr`), Irish (`ga`), Hebrew (`he`), Hindi (`hi`), Russian (`ru`), Swahili (`sw`), Thai (`th`), Vietnamese (`vi`), Chinese (`zh`), and English (`eng`), alongside additional languages in parts of the retrieved data.
- **Injection placement/type:** names such as `first`, `last`, `word`, `brackets`, `between`, `mult_2`, `mult_3`, `crit`, `syn_*`, `leet`, and `date_2024` identify experimental variants.
- **Model:** stored results are grouped around GPT-OSS 20B, Llama 3 8B, and Qwen 3 32B, with code references to GPT-OSS 120B and Llama 3 70B as well.

The directory names are experimental provenance. Renaming them without updating script constants and report paths will break the implicit links between retrieved variants, labels, and figures.

## External systems and dependencies

The scripts depend on a broad research stack:

- Amazon Bedrock via `boto3`/`botocore` for LLM inference.
- Amazon Translate via `boto3` for multilingual query variants.
- Pyserini for TREC topics/qrels and MS MARCO Lucene indexes.
- pandas and NumPy for tabular processing.
- scikit-learn, statsmodels, and `krippendorff` for agreement and statistical analysis.
- Matplotlib and Seaborn for plots.
- Transformers, PyTorch, Tiktoken, Hugging Face Hub, spaCy, and TextDescriptives for tokenization and perplexity-related analyses.
- PyFiglet for word-art injection.

Bedrock code defaults to AWS region `us-west-2` and first attempts to use a local AWS profile named `rmit`, falling back to the default credential chain. Some translation scripts use `ap-southeast-2`. Credentials are therefore expected to be configured outside the repository.

Model IDs visible in active and historical scripts include:

```text
openai.gpt-oss-20b-1:0
openai.gpt-oss-120b-1:0
meta.llama3-8b-instruct-v1:0
meta.llama3-70b-instruct-v1:0
qwen.qwen3-32b-v1:0
```

## Running and extending the work

Before running a script, review its configuration block. In particular, verify:

- the TREC DL year and input variant;
- input and output paths;
- the Bedrock model ID and AWS region/profile;
- concurrency, chunk size, sampling, and random-seed settings;
- the passage/query columns expected by the script; and
- whether the script writes a new output or resumes/combines partial results.

Run scripts from the repository root because many paths are repository-relative and imports use the `scripts` namespace. Retrieval may also require a local Java runtime and a large Pyserini index download/cache. One retrieval script currently hard-codes `D:\PyseriniCache`.

When adding a new experiment variant, the existing convention is to:

1. create the transformed CSVs under `retrieved/trec_dl_<year>/<variant>/`;
2. configure a labeling script to read that variant and write under `outputs/llm_label/trec_dl_<year>/<model>/`;
3. generate agreement tables/plots under `figures/<year>/<model>/`; and
4. keep the same `qid` and `pid` fields so report helpers can pair model and NIST rows.

## Current engineering characteristics

- **No environment manifest:** there is no `pyproject.toml`, `requirements.txt`, Conda environment file, or equivalent, so the exact reproducible dependency set is not recorded.
- **No automated test or CI suite:** there is no `tests/` or `.github/` workflow directory. Validation currently comes from script-level checks and inspection of generated artifacts.
- **Configuration is mostly in source:** experiment scripts rely on edited module constants rather than a shared CLI/configuration layer.
- **Research history is retained in place:** there are several copied or superseded scripts and many temporary output directories. This preserves provenance but makes it hard to identify the canonical script for a task from its filename alone.
- **Generated artifacts are committed:** CSVs, JSONL files, charts, and some Python bytecode are versioned alongside code. The repository behaves more like a complete research snapshot than a lightweight source distribution.
- **Portability is limited:** some paths and cloud settings are machine-specific, and the scripts assume particular directory names.
- **README/license are absent:** this overview is the first central map of the repository, but it does not establish licensing or a fully reproducible runbook.

## Practical starting points

- To understand retrieval, begin with `scripts/retrieval/retrieve_doc_trec_dl_all_20XX.py`.
- To understand multilingual injection, begin with `scripts/injector/inject_lang.py`; its subcommands cover complete-query, word-level, field-translation, and synonym workflows.
- The root Python environment is managed by uv. Run `uv sync` after cloning and invoke the unified injector with `uv run python scripts/injector/inject_lang.py ...`.
- To understand LLM labeling, begin with `scripts/label/trec_label_criterion_concurrent.py` and `scripts/bedrock_client.py`.
- To understand evaluation, begin with `scripts/report/seaborn_script/confusion_matrix/confusion_matrix_lang_2.py` and the helpers beside it.
- To inspect concrete results, browse `figures/<year>/<model>/confusion_matrix/<variant>/metrics_llm_vs_nist.csv` together with the corresponding SVG.

## Scope note

The high-level research purpose above is inferred from the repository's prompts, pipeline code, filenames, and stored artifacts because the repository previously had no README or formal project description. The document describes the checked-in state; it does not claim that every historical script or generated result remains reproducible on a clean machine.
