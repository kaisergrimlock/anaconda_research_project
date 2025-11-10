#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit
from helpers.llm_nist_pairing import PairingConfig, pair_labels
from helpers.output_writer import (
    write_confusion_outputs, write_metrics, write_unparseable_rows,
    write_missing_nist, write_llm_extra, save_heatmap, write_df
)

from helpers.metrics_llm import (
    compute_mae, compute_weighted_kappa_ordinal,
    compute_unweighted_kappa, binarize_labels,
)

# -------- Config --------
TREC_DL_YEAR = "2023"
MODEL = "gpt-oss-20b"
LANG  = "ru"  # "raw","eng","vi","fr"

if LANG == "raw":
    TOPIC_QUERY_COL, TOPIC_PASSAGE_COL = "query", "passage"
else:
    TOPIC_QUERY_COL, TOPIC_PASSAGE_COL = f"query_{LANG}", "passage_injected"

TOPIC_PID_COL = "pid_resolved"

NIST_DIR   = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
TOPICS_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / ( "judged" if LANG == "raw" else LANG )
LLM_FILE   = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{'raw' if LANG=='raw' else LANG}.csv"
TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

OUT_DIR    = Path("outputs/baseline") / TREC_DL_YEAR / LANG
OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT    = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_SVG    = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"
OUT_UNPARSEABLE = OUT_DIR / "llm_unparseable_labels.csv"
OUT_UNRESOLVED  = OUT_DIR / "llm_unresolved_qid.csv"
OUT_NIST_MISSING= OUT_DIR / "nist_not_joined_by_llm.csv"
OUT_LLM_EXTRA   = OUT_DIR / "llm_not_in_nist.csv"

LABELS = [0,1,2,3]

def main():
    bump_field_limit()

    cfg = PairingConfig(
        nist_dir=NIST_DIR,
        topics_dir=TOPICS_DIR,
        topics_glob=TOPICS_GLOB,
        llm_file=LLM_FILE,
        lang=LANG,
        topic_pid_col=TOPIC_PID_COL,
        topic_query_col=TOPIC_QUERY_COL,
        topic_passage_col=TOPIC_PASSAGE_COL,
        nist_label_choices=["relevance","label","nist"],
        llm_label_choices=["llm_relevance","label"],
        allow_pid_only_fallback=True,
        map_invalid_to_zero=False,
    )

    # 1) Pairing (pure)
    res = pair_labels(cfg)

    # 2) (Optional) write diagnostics
    if not res.unparseable_rows.empty:
        write_unparseable_rows(res.unparseable_rows, OUT_UNPARSEABLE, chunk_dir=OUT_DIR / "unparseable")
    if not res.unresolved_qid_rows.empty:
        write_df(res.unresolved_qid_rows, OUT_UNRESOLVED)
    if not res.nist_missing_df.empty:
        write_missing_nist(res.nist_missing_df, OUT_NIST_MISSING, chunk_dir=OUT_DIR / "missing_nist")
    if not res.llm_extra_df.empty:
        write_llm_extra(res.llm_extra_df, OUT_LLM_EXTRA, chunk_dir=OUT_DIR / "missing_llm")

    # 3) Confusions + metrics (pure in main)
    paired = res.paired
    cm = pd.crosstab(
        index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM"],  categories=LABELS, ordered=True),
        dropna=False,
    )
    cm.index.name = "NIST"; cm.columns.name = "LLM"
    cm_pct = cm.div(cm.sum(axis=1).replace(0,1), axis=0) * 100.0

    mae = compute_mae(paired["NIST"], paired["LLM"])
    kappa_weighted = compute_weighted_kappa_ordinal(cm)

    paired_bin = paired.copy()
    paired_bin["NIST_bin"] = binarize_labels(paired_bin["NIST"])
    paired_bin["LLM_bin"]  = binarize_labels(paired_bin["LLM"])
    cm_bin = pd.crosstab(
        index=pd.Categorical(paired_bin["NIST_bin"], categories=[0,1], ordered=True),
        columns=pd.Categorical(paired_bin["LLM_bin"],  categories=[0,1], ordered=True),
        dropna=False,
    )
    kappa_binary = compute_unweighted_kappa(cm_bin)

    # 4) Write outputs (effects)
    write_confusion_outputs(cm, cm_pct, OUT_COUNTS, OUT_PCT)

    metrics_df = pd.DataFrame(
        [
            {"metric":"mae", "value": float(mae)},
            {"metric":"kappa_weighted_4pt", "value": float(kappa_weighted)},
            {"metric":"kappa_binary_2pt", "value": float(kappa_binary)},
            {"metric":"pairs", "value": float(len(paired))},
        ]
    )
    write_metrics(metrics_df, OUT_DIR / "metrics_llm_vs_nist.csv")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title(f"Confusion Matrix: NIST vs LLM — {MODEL} {TREC_DL_YEAR} {LANG}")
    plt.ylabel("NIST label"); plt.xlabel("LLM label")
    save_heatmap(plt, OUT_SVG, dpi=200, tight=True, show=True)

if __name__ == "__main__":
    main()
