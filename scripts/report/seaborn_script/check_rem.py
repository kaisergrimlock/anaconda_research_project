#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# =========================
# Path setup
# =========================
THIS_FILE = Path(__file__).resolve()
SEABORN_SCRIPT_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[3]
print("ROOT:", PROJECT_ROOT)

if str(SEABORN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SEABORN_SCRIPT_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit

# =========================
# Config
# =========================
TREC_DL_YEARS = ["2021", "2022"]

MODEL = "gpt-oss-20b"
# MODEL = "qwen3-32b-v1"
# MODEL = "llama3-8b-instruct"

CHANGE_THRESHOLD = 0.10  # 10%

# LANGUAGES = [
#     "eng_qp_rem",
#     "ar_qp_rem",
#     "vi_qp_rem",
#     "th_qp_rem",
#     "fr_qp_rem",
#     "ru_qp_rem",
#     "he_qp_rem",
#     "sw_qp_rem",
#     "ga_qp_rem",
#     "hi_qp_rem",
#     "zh_qp_rem",
# ]

LANGUAGES = [
        "eng_instruct_instruct_rem", "ar_instruct_instruct_rem", "vi_instruct_instruct_rem", "th_instruct_instruct_rem", "fr_instruct_instruct_rem",
        "ru_instruct_instruct_rem", "he_instruct_instruct_rem", "sw_instruct_instruct_rem", "ga_instruct_instruct_rem", "hi_instruct_instruct_rem", "zh_instruct_instruct_rem",
]


def get_input_file(year: str, lang: str) -> Path:
    return (
        Path("outputs/llm_label")
        / f"trec_dl_{year}"
        / MODEL
        / f"{MODEL}_trecdl_{year}_{lang}_labels.csv"
    )


def get_output_file(year: str, lang: str) -> Path:
    return (
        Path("outputs/oversanitization")
        / f"trec_dl_{year}"
        / MODEL
        / f"{MODEL}_trecdl_{year}_{lang}_oversanitization.csv"
    )


def normalize_words(text: str) -> list[str]:
    text = "" if pd.isna(text) else str(text).lower()
    return re.findall(r"\b\w+\b", text)


def get_erroneous_removal_and_ratio(
    original: str,
    removed: str,
) -> tuple[str, float]:
    original_words = normalize_words(original)
    removed_words = normalize_words(removed)

    if not original_words:
        return "", 0.0

    removed_word_counts = {}

    for word in removed_words:
        removed_word_counts[word] = removed_word_counts.get(word, 0) + 1

    missing_words = []

    for word in original_words:
        if removed_word_counts.get(word, 0) > 0:
            removed_word_counts[word] -= 1
        else:
            missing_words.append(word)

    change_ratio = len(missing_words) / len(original_words)
    erroneous_removal = " ".join(missing_words)

    return erroneous_removal, change_ratio


def is_parse_fail(text: str) -> bool:
    if pd.isna(text):
        return True

    text = str(text).lower().strip()

    parse_fail_patterns = [
        "sorry i can't",
        "sorry, i can't",
        "i cannot",
        "i can't assist",
        "i can’t assist",
        "i'm sorry",
        "i’m sorry",
    ]

    return any(pattern in text for pattern in parse_fail_patterns)


def check_oversanitization(year: str, lang: str) -> pd.DataFrame:
    bump_field_limit()

    input_file = get_input_file(year, lang)
    output_file = get_output_file(year, lang)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"[WARN] Missing input file for {year} {lang}: {input_file}")
        return pd.DataFrame()

    df = pd.read_csv(input_file)

    required_cols = [
        "qid",
        "query",
        "pid",
        "passage",
        "passage_removed",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing columns in {input_file}: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    result_df = df[
        [
            "qid",
            "query",
            "pid",
            "passage",
            "passage_removed",
        ]
    ].copy()

    result_df["parse_fail"] = result_df["passage_removed"].apply(
        lambda text: "yes" if is_parse_fail(text) else "no"
    )

    result_df[["erroneous_removal", "change_ratio"]] = result_df.apply(
        lambda row: pd.Series(
            get_erroneous_removal_and_ratio(
                row["passage"],
                row["passage_removed"],
            )
        ),
        axis=1,
    )

    result_df["oversanitization"] = result_df.apply(
        lambda row: (
            "yes"
            if row["change_ratio"] >= CHANGE_THRESHOLD
            and row["parse_fail"] == "no"
            else "no"
        ),
        axis=1,
    )

    result_df = result_df[
        [
            "qid",
            "query",
            "pid",
            "passage",
            "passage_removed",
            "parse_fail",
            "oversanitization",
            "erroneous_removal",
        ]
    ]

    result_df.to_csv(output_file, index=False, encoding="utf-8")

    total_rows = len(result_df)
    parse_fail_count = (result_df["parse_fail"] == "yes").sum()
    oversanitized_count = (result_df["oversanitization"] == "yes").sum()

    parse_fail_pct = (parse_fail_count / total_rows * 100) if total_rows else 0
    oversanitized_pct = (oversanitized_count / total_rows * 100) if total_rows else 0

    print(f"[{year} {lang}] Total rows: {total_rows}")
    print(f"[{year} {lang}] Parse failures: {parse_fail_count} ({parse_fail_pct:.2f}%)")
    print(f"[{year} {lang}] Oversanitized rows: {oversanitized_count} ({oversanitized_pct:.2f}%)")
    print(f"[{year} {lang}] Saved to: {output_file}")

    return result_df


def main() -> None:
    summary = []

    for year in TREC_DL_YEARS:
        for lang in LANGUAGES:
            try:
                result_df = check_oversanitization(year, lang)

                if result_df.empty:
                    summary.append((year, lang, 0, 0.0, 0, 0.0, 0))
                    continue

                total_rows = len(result_df)

                parse_fail_count = (result_df["parse_fail"] == "yes").sum()
                oversanitized_count = (result_df["oversanitization"] == "yes").sum()

                parse_fail_pct = parse_fail_count / total_rows * 100
                oversanitized_pct = oversanitized_count / total_rows * 100

                summary.append(
                    (
                        year,
                        lang,
                        parse_fail_count,
                        parse_fail_pct,
                        oversanitized_count,
                        oversanitized_pct,
                        total_rows,
                    )
                )

            except Exception as e:
                print(f"[ERROR] Failed for {year} {lang}: {e}")
                summary.append((year, lang, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"))

    print("\n=== Summary ===")
    print("year, language, parse_fail, parse_fail_pct, oversanitization, oversanitization_pct, total_rows")

    for (
        year,
        lang,
        parse_fail_count,
        parse_fail_pct,
        oversanitized_count,
        oversanitized_pct,
        total_rows,
    ) in summary:
        if parse_fail_count == "ERROR":
            print(f"{year}, {lang}: ERROR")
        else:
            print(
                f"{year}, {lang}: "
                f"{parse_fail_count} ({parse_fail_pct:.2f}%), "
                f"{oversanitized_count} ({oversanitized_pct:.2f}%), "
                f"total={total_rows}"
            )


if __name__ == "__main__":
    main()