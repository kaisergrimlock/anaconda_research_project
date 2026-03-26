#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

THIS_FILE = Path(__file__).resolve()
SEABORN_SCRIPT_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[4]

if str(SEABORN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SEABORN_SCRIPT_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit
from helpers.metrics_llm import (
    compute_mae,
    compute_weighted_kappa_ordinal,
    compute_unweighted_kappa,
    compute_krippendorff_alpha_paired,
    binarize_labels,
)

TREC_DL_YEAR = "2022"
LABELS = [0, 1, 2, 3]

MODELS = [
    ("gpt-oss-20b", "GPT-OSS-20B"),
    ("qwen3-32b-v1", "QWEN-3-32B"),
    ("llama3-8b-instruct", "LLAMA-3-8B"),
]

PROMPTS = [
    ("eng", "Utility"),
    ("eng_crit_2", "Criteria"),
]


def build_llm_file(model_key: str, lang: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "llm_label"
        / f"trec_dl_{TREC_DL_YEAR}"
        / model_key
        / f"{model_key}_trecdl_{TREC_DL_YEAR}_{lang}_labels.csv"
    )


def load_valid_pairs(llm_file: Path) -> pd.DataFrame:
    bump_field_limit()

    if not llm_file.exists():
        raise FileNotFoundError(f"Missing file: {llm_file}")

    df = pd.read_csv(llm_file)

    if "relevance" not in df.columns or "llm_relevance" not in df.columns:
        raise ValueError(
            f"Expected columns 'relevance' and 'llm_relevance' in {llm_file}, "
            f"but got: {list(df.columns)}"
        )

    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    valid_mask = df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)
    paired = df.loc[valid_mask, ["NIST", "LLM"]].copy()

    if paired.empty:
        raise RuntimeError(f"No valid label pairs found in {llm_file}")

    return paired


def compute_metrics(paired: pd.DataFrame) -> dict[str, float]:
    cm_4pt = pd.crosstab(
        index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM"], categories=LABELS, ordered=True),
        dropna=False,
    )

    mae_ord = compute_mae(paired["NIST"], paired["LLM"])
    kappa_ord = compute_weighted_kappa_ordinal(cm_4pt)
    alpha_ord = compute_krippendorff_alpha_paired(
        paired["NIST"],
        paired["LLM"],
        level="ordinal",
    )

    paired_bin = paired.copy()
    paired_bin["NIST_bin"] = binarize_labels(paired_bin["NIST"])
    paired_bin["LLM_bin"] = binarize_labels(paired_bin["LLM"])

    cm_bin = pd.crosstab(
        index=pd.Categorical(paired_bin["NIST_bin"], categories=[0, 1], ordered=True),
        columns=pd.Categorical(paired_bin["LLM_bin"], categories=[0, 1], ordered=True),
        dropna=False,
    )

    mae_bin = compute_mae(paired_bin["NIST_bin"], paired_bin["LLM_bin"])
    kappa_bin = compute_unweighted_kappa(cm_bin)

    return {
        "mae_ord": float(mae_ord),
        "mae_bin": float(mae_bin),
        "kappa_ord": float(kappa_ord),
        "kappa_bin": float(kappa_bin),
        "alpha_ord": float(alpha_ord),
    }


def format_row(model_display: str, prompt_display: str, m: dict[str, float]) -> str:
    return (
        f"{model_display} ({prompt_display})"
        f" & {m['mae_ord']:.3f}"
        f" & {m['mae_bin']:.3f}"
        f" & {m['kappa_ord']:.3f}"
        f" & {m['kappa_bin']:.3f}"
        f" & {m['alpha_ord']:.3f} \\\\"
    )


def build_latex_table(rows: list[str]) -> str:
    body = "\n".join(rows)
    return rf"""
\begin{{tabular}}{{
    @{{}}l
    S[table-format=1.3]
    S[table-format=1.3]
    S[table-format=1.3]
    S[table-format=1.3]
    S[table-format=1.3]
    @{{}}
}}
\toprule
\textbf{{Model / Prompt}}
& \multicolumn{{2}}{{c}}{{\textbf{{MAE}}}}
& \multicolumn{{2}}{{c}}{{\textbf{{Cohen's $\kappa$}}}}
& \multicolumn{{1}}{{c}}{{\textbf{{Krippendorff's $\alpha$}}}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(l){{6-6}}
& {{\textbf{{Ordinal}}}}
& {{\textbf{{Binary}}}}
& {{\textbf{{Ordinal}}}}
& {{\textbf{{Binary}}}}
& {{\textbf{{Ordinal}}}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""".strip()


def main() -> None:
    rows: list[str] = []

    for model_key, model_display in MODELS:
        for lang, prompt_display in PROMPTS:
            llm_file = build_llm_file(model_key, lang)
            print(f"[INFO] Reading: {llm_file}")
            paired = load_valid_pairs(llm_file)
            metrics = compute_metrics(paired)
            rows.append(format_row(model_display, prompt_display, metrics))

    print()
    print(build_latex_table(rows))


if __name__ == "__main__":
    main()