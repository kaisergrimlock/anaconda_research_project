#!/usr/bin/env python3
from __future__ import annotations

"""
Create CSV and LaTeX tables with TOTAL FP/FN rates across
both TREC-DL 2021 and TREC-DL 2022.

This version is configured for comparing:
    - Phrase injections: language names such as ar, eng, fr, ...
    - Word injections:   language names such as ar_word, eng_word, fr_word, ...

Main fixes:
    1. Empty suffixes no longer produce empty base_language values.
    2. Word suffixes remove the leading underscore correctly, so ar_word -> AR.
    3. Longer suffixes are matched before shorter suffixes.
    4. LaTeX cells use the same red/orange heatmap styling as the 3-category table.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -------------------------------------------------------------------
# Project-root setup
# -------------------------------------------------------------------
cwd = Path.cwd().resolve()

if cwd.name == "statistics_tests":
    os.chdir(cwd.parents[3])

PROJECT_ROOT = Path.cwd().resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEABORN_ROOT = PROJECT_ROOT / "scripts" / "report" / "seaborn_script"

if str(SEABORN_ROOT) not in sys.path:
    sys.path.insert(0, str(SEABORN_ROOT))

from helpers.lang_profiles import get_langs
from scripts.csv_helpers import bump_field_limit


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
TREC_DL_YEARS = ["2021", "2022"]

# This should match the profile defined in helpers.lang_profiles.
LANG_PROFILE = "word"

# Format: (suffix_in_filename, display_name_in_table)
# Phrase files usually have no suffix: ar, eng, fr, ...
# Word files usually have _word suffix: ar_word, eng_word, fr_word, ...
SUFFIX_PHRASE = ("", "Phrase")
SUFFIX_WORD = ("_word", "Word")

SUFFIX_CONFIGS: List[Tuple[str, str]] = [
    SUFFIX_WORD,
    SUFFIX_PHRASE,
]

# Always match longer suffixes first.
# This prevents an empty suffix from matching everything too early.
SORTED_SUFFIX_CONFIGS: List[Tuple[str, str]] = sorted(
    SUFFIX_CONFIGS,
    key=lambda item: len(item[0]),
    reverse=True,
)

# Controls row order in the LaTeX table.
SETTING_ORDER = [
    SUFFIX_PHRASE[1],
    SUFFIX_WORD[1],
]

LANGS = [
    lang for lang in get_langs(LANG_PROFILE)
    if not lang.startswith("raw")
]

ALLOWED_MODELS = {
    "gpt-oss-20b",
    "qwen3-32b-v1",
    "llama3-8b-instruct",
}

MODEL_ORDER = [
    "gpt-oss-20b",
    "qwen3-32b-v1",
    "llama3-8b-instruct",
]

THRESHOLD = 2

OUTPUT_CSV_FILE = (
    Path("outputs")
    / "tables"
    / f"fp_fn_total_{LANG_PROFILE}.csv"
)

OUTPUT_TEX_FILE = (
    Path("outputs")
    / "tables"
    / f"fp_fn_total_{LANG_PROFILE}.tex"
)

GOLD_COL = "relevance"
LLM_COL = "llm_relevance"

LABELS = [0, 1, 2, 3]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def find_llm_files(year: str) -> Dict[str, List[Path]]:
    label_root = Path("outputs/llm_label") / f"trec_dl_{year}"

    if not label_root.exists():
        raise FileNotFoundError(f"Missing label root: {label_root}")

    model_files: Dict[str, List[Path]] = {}

    for model_dir in label_root.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        if model_name not in ALLOWED_MODELS:
            continue

        csv_files = list(
            model_dir.glob(
                f"{model_name}_trecdl_{year}_*_labels.csv"
            )
        )

        if csv_files:
            model_files[model_name] = csv_files

    return model_files


def get_lang_from_filename(
    file_path: Path,
    model: str,
    year: str,
) -> Optional[str]:
    prefix = f"{model}_trecdl_{year}_"
    suffix = "_labels.csv"

    name = file_path.name

    if not name.startswith(prefix):
        return None

    if not name.endswith(suffix):
        return None

    return name[len(prefix):-len(suffix)]


def compute_counts(df: pd.DataFrame) -> Tuple[int, int, int]:
    missing = {GOLD_COL, LLM_COL} - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gold = pd.to_numeric(df[GOLD_COL], errors="coerce")
    llm = pd.to_numeric(df[LLM_COL], errors="coerce")

    valid = gold.notna() & llm.notna()

    gold = gold[valid].astype(int)
    llm = llm[valid].astype(int)

    valid_labels = gold.isin(LABELS) & llm.isin(LABELS)

    gold = gold[valid_labels]
    llm = llm[valid_labels]

    gold_bin = (gold >= THRESHOLD).astype(int)
    llm_bin = (llm >= THRESHOLD).astype(int)

    fp = ((llm_bin == 1) & (gold_bin == 0)).sum()
    fn = ((llm_bin == 0) & (gold_bin == 1)).sum()

    total = len(gold)

    return int(fp), int(fn), int(total)


def model_display_name(model: str) -> str:
    names = {
        "gpt-oss-20b": "GPT-OSS",
        "qwen3-32b-v1": "QWEN",
        "llama3-8b-instruct": "LLAMA",
    }

    return names.get(model, model)


def setting_display_name(lang: str) -> str:
    """Return Phrase or Word based on the language filename suffix."""
    for suffix, display_name in SORTED_SUFFIX_CONFIGS:
        if suffix == "":
            # Empty suffix should be treated as the fallback/default setting.
            # It must not run before the non-empty suffixes.
            continue

        if lang.endswith(suffix):
            return display_name

    # If no explicit suffix matched, this is the phrase/default version.
    for suffix, display_name in SUFFIX_CONFIGS:
        if suffix == "":
            return display_name

    return "Unknown"


def base_lang(lang: str) -> str:
    """
    Extract the base language code used for table columns.

    Examples:
        ar      -> AR
        ar_word -> AR
        eng     -> ENG
        eng_word -> ENG
    """
    for suffix, _ in SORTED_SUFFIX_CONFIGS:
        if suffix == "":
            continue

        if lang.endswith(suffix):
            return lang[:-len(suffix)].rstrip("_").upper()

    return lang.rstrip("_").upper()


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("{", r"\{")
            .replace("}", r"\}")
    )


def percentage_cell(value: int, colour: str) -> str:
    """
    Colour the cell background based on percentage.

    Higher value = stronger colour.
    The range is capped to avoid unreadably dark cells.
    """
    value = int(value)

    if value <= 0:
        return rf"${value}\%$"

    intensity = min(max(value, 5), 80)

    return rf"\cellcolor{{{colour}!{intensity}}}${value}\%$"


def fp_cell(value: int) -> str:
    return percentage_cell(value, "red")


def fn_cell(value: int) -> str:
    return percentage_cell(value, "orange")


# -------------------------------------------------------------------
# LaTeX table writer
# -------------------------------------------------------------------
def write_latex_table(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    df = df.copy()

    # Recompute these here to avoid trusting stale CSV values.
    df["base_language"] = df["language"].apply(base_lang)
    df["setting"] = df["language"].apply(setting_display_name)

    df = df[df["setting"] != "Unknown"]
    df = df[~df["base_language"].str.lower().eq("raw")]

    languages = sorted(df["base_language"].unique())

    column_format = "ll|" + "|".join(["cc"] * len(languages))

    lines: List[str] = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Total False Positive (FP) and False Negative (FN) rates of QP (Phrase) and QW (Word) across TREC-DL 2021 and 2022. Darker cells indicate higher percentages.}"
    )
    lines.append(r"\label{tab:fp_fn_total_word}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")
    lines.append(r"\hline")

    header = [
        r"\textbf{Model}",
        r"\textbf{Setting}",
    ]

    for i, lang in enumerate(languages):
        safe_lang = latex_escape(lang)

        if i == len(languages) - 1:
            header.append(
                rf"\multicolumn{{2}}{{c}}{{\textbf{{{safe_lang}}}}}"
            )
        else:
            header.append(
                rf"\multicolumn{{2}}{{c|}}{{\textbf{{{safe_lang}}}}}"
            )

    lines.append(" & ".join(header) + r" \\")

    subheader = ["", ""]

    for _ in languages:
        subheader.extend([
            r"\textbf{FP}",
            r"\textbf{FN}",
        ])

    lines.append(" & ".join(subheader) + r" \\")
    lines.append(r"\hline")

    for model in MODEL_ORDER:
        model_df = df[df["model"] == model]

        if model_df.empty:
            continue

        model_name = model_display_name(model)

        available_settings = [
            setting for setting in SETTING_ORDER
            if not model_df[model_df["setting"] == setting].empty
        ]

        for idx, setting in enumerate(available_settings):
            row_df = model_df[model_df["setting"] == setting]

            if idx == 0:
                row = [
                    rf"\multirow{{{len(available_settings)}}}{{*}}{{\textbf{{{model_name}}}}}",
                    latex_escape(setting),
                ]
            else:
                row = [
                    "",
                    latex_escape(setting),
                ]

            for lang in languages:
                match = row_df[row_df["base_language"] == lang]

                if match.empty:
                    row.extend([
                        r"\textemdash",
                        r"\textemdash",
                    ])
                    continue

                fp = int(match.iloc[0]["FP"])
                fn = int(match.iloc[0]["FN"])

                row.extend([
                    fp_cell(fp),
                    fn_cell(fn),
                ])

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\hline")

    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    bump_field_limit()

    aggregate: Dict[Tuple[str, str], Dict[str, int]] = {}

    for year in TREC_DL_YEARS:
        print(f"\n========== TREC-DL {year} ==========")

        model_files = find_llm_files(year)

        for model, files in model_files.items():
            for file_path in files:
                lang = get_lang_from_filename(file_path, model, year)

                if lang is None:
                    continue

                if lang.startswith("raw"):
                    continue

                if lang not in LANGS:
                    continue

                setting = setting_display_name(lang)

                if setting == "Unknown":
                    continue

                try:
                    df = pd.read_csv(file_path)
                    fp, fn, total = compute_counts(df)

                    key = (model, lang)

                    if key not in aggregate:
                        aggregate[key] = {
                            "FP": 0,
                            "FN": 0,
                            "TOTAL": 0,
                        }

                    aggregate[key]["FP"] += fp
                    aggregate[key]["FN"] += fn
                    aggregate[key]["TOTAL"] += total

                    print(
                        f"[OK] "
                        f"{year} | "
                        f"{model} | "
                        f"{lang} | "
                        f"{setting} | "
                        f"base={base_lang(lang)} | "
                        f"FP={fp} "
                        f"FN={fn} "
                        f"TOTAL={total}"
                    )

                except Exception as e:
                    print(
                        f"[SKIP] "
                        f"{year} | "
                        f"{model} | "
                        f"{lang} | "
                        f"{e}"
                    )

    rows = []

    for (model, lang), vals in aggregate.items():
        total = vals["TOTAL"]

        if total == 0:
            continue

        fp_rate = int((vals["FP"] / total) * 100)
        fn_rate = int((vals["FN"] / total) * 100)

        rows.append({
            "model": model,
            "language": lang,
            "setting": setting_display_name(lang),
            "base_language": base_lang(lang),
            "FP": fp_rate,
            "FN": fn_rate,
        })

    if not rows:
        raise RuntimeError(
            "No rows produced. Check LANG_PROFILE, LANGS, suffixes, and input folders."
        )

    out_df = pd.DataFrame(rows)

    out_df = out_df.sort_values(
        ["model", "base_language", "setting"]
    )

    OUTPUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV_FILE, index=False)

    write_latex_table(out_df, OUTPUT_TEX_FILE)

    print("\n===================================")
    print(f"Saved CSV table to:\n{OUTPUT_CSV_FILE}")
    print(f"Saved TEX table to:\n{OUTPUT_TEX_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()
