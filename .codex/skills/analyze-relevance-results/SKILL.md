---
name: analyze-relevance-results
description: Analyze and visualize agreement between LLM relevance labels, NIST judgments, and experiment variants in this repository. Use for confusion matrices, MAE, kappa/alpha, token fertility, perplexity, confidence intervals, or Tukey HSD outputs; do not use to retrieve data or invoke cloud models.
---

# Analyze Relevance Results

Work from the repository root. Read `docs/REPOSITORY_OVERVIEW.md`, inspect the requested label files, and choose the narrowest existing script under `scripts/report/` that answers the question.

## Route the analysis

- Use `scripts/report/seaborn_script/confusion_matrix/` for LLM-vs-NIST or variant-vs-variant pairing, confusion matrices, MAE, weighted/unweighted kappa, and Krippendorff's alpha.
- Use `scripts/report/process_baseline/summary.py` or `process_llm/` for dataset/model summaries and label distributions.
- Use `tokenizer_gpt.py`, `tokenizer_llama.py`, or `tokenizer_qwen.py` for original/injected token counts and fertility; use the matching plotting scripts for cross-language summaries.
- Use `qwen_perplexity_lm.py`, `textdescriptives_demo.py`, and the perplexity plotting/table scripts only for perplexity questions.
- Use `seaborn_script/tukey's hsd/` only when the requested comparison calls for Tukey HSD and its assumptions are appropriate.

## Pair data correctly

Never compare files by row position. Pair on stable identity fields, normally `(qid, pid)`, after normalizing identifier types. Keep NIST `relevance` distinct from `llm_relevance`. Before calculating metrics, report duplicate keys, missing counterparts, invalid labels, the number of paired rows, and any rows excluded.

For the 0–3 relevance scale, preserve graded metrics unless the question explicitly needs a binary view. Existing binary reports use the threshold `<=1 -> 0`, `>1 -> 1`; state this threshold next to binary results. Do not present binary and four-point kappa as interchangeable.

Use existing output conventions under `figures/<year>/<model>/` or `outputs/` and avoid overwriting another variant's tables or plots. Generated tables must identify year, model, variant(s), sample size, pairing/exclusion rules, and metric definition.

## Verify and interpret

Check that confusion-matrix totals equal the paired sample count and that percentage matrices use the intended denominator. Treat NaN/undefined agreement statistics as diagnostic outcomes, not zeros. For multi-variant claims, distinguish descriptive differences from statistically supported comparisons.

Report the main result in plain language, link the generated CSV and figure files, and call out missing/invalid rows or comparability limitations. This skill is local and read/compute oriented: do not invoke Bedrock, Translate, or retrieval downloads as part of analysis.
