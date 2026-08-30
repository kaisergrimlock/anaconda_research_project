---
name: run-llm-relevance-labeling
description: Run this repository's Amazon Bedrock LLM relevance-labeling scripts over raw or injected TREC DL passages. Use when generating, resuming, or validating model label CSVs and token/cost logs; do not use for retrieval, passage construction, or statistical reporting.
---

# Run LLM Relevance Labeling

This workflow can incur cloud cost. Work from the repository root, read `docs/REPOSITORY_OVERVIEW.md`, and inspect the selected script, prompt template, input header, and `scripts/bedrock_client.py` before a run.

## Select the labeling mode

- Use `scripts/label/trec_label concurrent.py` for ordinary concurrent 0–3 relevance labeling.
- Use `scripts/label/trec_label_criterion_concurrent.py` for criterion-level labeling when the requested output includes contextuality, coverage, exactness, and topicality.
- Use `trec_label_criterion_batch.py` or `trec_label_criteria_composition_batch.py` only when the user requests Bedrock batch inference.
- Use `trec_label_logprob.py`, `trec_label_query_adjust.py`, or a decomposition/composition script only for the specifically named experimental method.

Historical `copy` scripts are not canonical. Do not choose one only because its current constants happen to match the request.

## Pre-run contract

Resolve and report:

- TREC DL year and raw/injection variant;
- input files and row count;
- prompt file and passage column (`passage` for raw, normally `passage_injected` otherwise);
- exact Bedrock model ID;
- inference settings, concurrency/batch mode, and resume behavior;
- output and log directories; and
- estimated token volume/cost when it can be calculated from a sample and `scripts/report/llm_cost.csv`.

The shared client tries AWS profile `rmit` in `us-west-2`, then the default credential chain. Confirm credentials without printing secrets. Obtain explicit authorization immediately before sending prompts to Bedrock unless the user has already authorized that run and scope.

Never overwrite completed labels for a different configuration. Use the repository's year/model/variant naming pattern, and preserve partial outputs when resuming. A retry must be bounded; stop and report repeated authentication, quota, throttling, schema, or parsing failures instead of launching an unbounded rerun.

## Run and verify

Run the selected entry script from the repository root. Monitor processed rows, failures, input/output token counts, and estimated cost. Do not expose credentials or raw sensitive logs in the response.

After completion, verify:

- each input `(qid, pid)` has at most one corresponding result;
- required identity, NIST `relevance`, and `llm_relevance` fields are present;
- `llm_relevance` parses to the intended 0–3 scale;
- missing/invalid responses are counted and retained for diagnosis rather than silently dropped;
- combined outputs match the sum of successfully processed parts; and
- token-usage and run metadata point to the generated files.

Report model, prompt/mode, variant, processed/success/invalid counts, tokens/cost, and output locations. Stop before statistical analysis unless separately requested.
