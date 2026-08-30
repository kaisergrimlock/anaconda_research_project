---
name: retrieve-trec-data
description: Retrieve or sample TREC Deep Learning topics, qrels, and MS MARCO passages for this repository. Use when preparing judged/top-k data for a TREC DL year, rebuilding a retrieval dataset, or sampling non-relevant passages; do not use for passage injection, LLM labeling, or result analysis.
---

# Retrieve TREC Data

Work from the repository root. Read `docs/REPOSITORY_OVERVIEW.md` and inspect the selected retrieval script before changing or running it.

## Route the request

- Use `scripts/retrieval/retrieve_doc_trec_dl_all_20XX.py` for TREC DL 2019–2023 qrels export or BM25 top-k retrieval.
- Use `scripts/retrieval/retrieve_doc_trec_dl_all_2019.py` only for the older 2019-specific workflow.
- Use `scripts/retrieval/retrieve_non_relevant.py` to sample human-judged relevance-0 passages from an existing retrieved dataset.
- Use `scripts/retrieval/bm_25_score.py` only when the request is specifically to add or inspect BM25 scores.
- Treat the fake-passage generators as synthetic-data creation, not retrieval. Do not run them under this skill unless the user explicitly requests generated passages.

## Prepare the run

Inspect the configuration block and resolve the requested year, level (`passage` or `document`), mode (`qrels` or `topk`), query selection, depth, chunk size, and output directory. Do not silently use the file's current experimental constants when the request specifies different values.

Before downloading or opening a prebuilt Pyserini index, report the selected index and expected output location. Pyserini may download a large index and needs Java; obtain authorization immediately before any network-heavy download when it has not already been authorized.

The general retrieval script hard-codes `D:\PyseriniCache`. Confirm that the path is usable or change it to a workspace-appropriate cache. Never overwrite a populated retrieval variant unless the user explicitly asks; choose a new output directory for a materially different configuration.

## Run and verify

Run the chosen script from the repository root. Preserve the canonical row identity and schema:

```text
qid,query,pid,passage,relevance
```

After the run, verify the output files exist, headers are consistent across chunks, row counts are plausible, `qid`/`pid` are populated, and relevance values are in the expected TREC scale. Report the exact configuration, output path, file/row count, and any missing passages or retrieval warnings.

Stop once the requested retrieval dataset has been produced and checked. Do not start injection or labeling without a separate user request.
