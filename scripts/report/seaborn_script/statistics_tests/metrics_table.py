#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


# -------------------------------------------------------------------
# Project-root setup
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# metrics_table.py is in:
# scripts/report/seaborn_script/statistics_tests
PROJECT_ROOT = SCRIPT_DIR.parents[3]

os.chdir(PROJECT_ROOT)

SEABORN_ROOT = PROJECT_ROOT / "scripts" / "report" / "seaborn_script"

# Put SEABORN_ROOT first so helpers.report_metrics resolves from seaborn_script/helpers
if str(SEABORN_ROOT) not in sys.path:
    sys.path.insert(0, str(SEABORN_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit

from helpers.metrics_llm import (
    compute_mae,
    compute_weighted_kappa_ordinal,
    compute_krippendorff_alpha_paired,
)


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
TREC_DL_YEARS = ["2021", "2022"]

MODEL_ORDER = [
    "gpt-oss-20b",
    "llama3-8b-instruct",
    "qwen3-32b-v1",
]

GOLD_COL = "relevance"
LLM_COL = "llm_relevance"

OUTPUT_CSV_FILE = Path("outputs") / "tables" / "metric_summary_raw.csv"
OUTPUT_TEX_FILE = Path("outputs") / "tables" / "metric_summary_raw.tex"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def model_display_name(model: str) -> str:
    names = {
        "gpt-oss-20b": "gpt-20b",
        "llama3-8b-instruct": "llama3-8b",
        "qwen3-32b-v1": "qwen-32b",
    }
    return names.get(model, model)


def find_raw_label_file(year: str, model: str) -> Path | None:
    model_dir = Path("outputs") / "llm_label" / f"trec_dl_{year}" / model

    if not model_dir.exists():
        return None

    candidates = list(
        model_dir.glob(f"{model}_trecdl_{year}_raw_labels.csv")
    )

    if candidates:
        return candidates[0]

    candidates = list(
        model_dir.glob(f"{model}_trecdl_{year}_raw*_labels.csv")
    )

    if candidates:
        return candidates[0]

    return None


def build_confusion_matrix_4pt(
    gold: pd.Series,
    llm: pd.Series,
) -> pd.DataFrame:
    cm = pd.crosstab(
        index=pd.Categorical(gold, categories=[0, 1, 2, 3], ordered=True),
        columns=pd.Categorical(llm, categories=[0, 1, 2, 3], ordered=True),
        dropna=False,
    )

    cm.index.name = "NIST"
    cm.columns.name = "LLM"

    return cm


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    missing = {GOLD_COL, LLM_COL} - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gold = pd.to_numeric(df[GOLD_COL], errors="coerce")
    llm = pd.to_numeric(df[LLM_COL], errors="coerce")

    valid = gold.notna() & llm.notna()

    gold = gold[valid].astype(int)
    llm = llm[valid].astype(int)

    valid_labels = gold.isin([0, 1, 2, 3]) & llm.isin([0, 1, 2, 3])

    gold = gold[valid_labels]
    llm = llm[valid_labels]

    if len(gold) == 0:
        raise ValueError("No valid rows after cleaning.")

    cm_4pt = build_confusion_matrix_4pt(gold, llm)

    mae = compute_mae(gold, llm)

    kappa = compute_weighted_kappa_ordinal(cm_4pt)

    alpha = compute_krippendorff_alpha_paired(
        gold,
        llm,
        level="ordinal",
        value_domain=[0, 1, 2, 3],
    )

    mean_diff = (llm - gold).mean()

    return {
        "MAE": mae,
        "Alpha": alpha,
        "Kappa": kappa,
        "Mean-Diff": mean_diff,
    }


def latex_float(value: float) -> str:
    return f"{value:.4f}"


# -------------------------------------------------------------------
# LaTeX writer
# -------------------------------------------------------------------
def write_latex_table(df: pd.DataFrame, output_file: Path):
    lines = []

    lines.append(r"\begin{wraptable}{r}{0.48\textwidth}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Agreement between LLM-generated relevance labels and NIST judgments for the non-injected dataset on TREC-DL 2021 and 2022.}"
    )
    lines.append(r"\label{tab:metrics_raw}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\footnotesize")
    lines.append("")

    for year in TREC_DL_YEARS:
        year_df = df[df["year"] == year]

        lines.append(rf"\textbf{{TREC-DL {year}}} \\")
        lines.append(r"\vspace{0.2em}")
        lines.append("")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")

        header = [""]

        for model in MODEL_ORDER:
            header.append(rf"\textbf{{{model_display_name(model)}}}")

        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")

        for metric in ["MAE", "Alpha", "Kappa", "Mean-Diff"]:
            row = [rf"\textbf{{{metric}}}"]

            for model in MODEL_ORDER:
                match = year_df[year_df["model"] == model]

                if match.empty:
                    row.append(r"\textemdash")
                else:
                    row.append(latex_float(match.iloc[0][metric]))

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        if year != TREC_DL_YEARS[-1]:
            lines.append("")
            lines.append(r"\vspace{1em}")
            lines.append("")

    lines.append(r"\end{wraptable}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    bump_field_limit()

    rows = []

    for year in TREC_DL_YEARS:
        print(f"\n========== TREC-DL {year} ==========")

        for model in MODEL_ORDER:
            file_path = find_raw_label_file(year, model)

            if file_path is None:
                print(f"[SKIP] {year} | {model} | raw file not found")
                continue

            try:
                df = pd.read_csv(file_path)
                metrics = compute_metrics(df)

                rows.append({
                    "year": year,
                    "model": model,
                    "language": "raw",
                    **metrics,
                })

                print(
                    f"[OK] {year} | {model} | raw | "
                    f"MAE={metrics['MAE']:.4f} | "
                    f"Alpha={metrics['Alpha']:.4f} | "
                    f"Kappa={metrics['Kappa']:.4f} | "
                    f"Mean-Diff={metrics['Mean-Diff']:.4f}"
                )

            except Exception as e:
                print(f"[SKIP] {year} | {model} | {e}")

    if not rows:
        raise RuntimeError("No rows produced. Check raw label files.")

    out_df = pd.DataFrame(rows)

    out_df = out_df.sort_values(["year", "model"])

    OUTPUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(
        OUTPUT_CSV_FILE,
        index=False
    )

    write_latex_table(
        out_df,
        OUTPUT_TEX_FILE
    )

    print("\n===================================")
    print(f"Saved CSV table to:\n{OUTPUT_CSV_FILE}")
    print(f"Saved TEX table to:\n{OUTPUT_TEX_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()