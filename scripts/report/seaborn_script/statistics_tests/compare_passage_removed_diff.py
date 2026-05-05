"""
Compare original passages with passage_removed for each language profile.

This script reads CSV files by language, compares the `passage` and
`passage_removed` columns, and writes a new CSV with extra difference columns.

Expected default structure:
    retrieved/trec_dl_2021/{lang}_instruct_rem/all_topics_trecdl_2021_part1.csv
    retrieved/trec_dl_2021/{lang}_instruct_rem/all_topics_trecdl_2021_part2.csv
    ...

You can change INPUT_ROOT, YEAR, LANGUAGES, and PROFILE below.
"""

from pathlib import Path
import argparse
import difflib
import pandas as pd


# -----------------------------
# Default configuration
# -----------------------------
INPUT_ROOT = Path("retrieved")
OUTPUT_ROOT = Path("outputs/passage_removed_diff")
YEAR = "2021"
PROFILE = "instruct_rem"   # examples: instruct_rem, qp_rem

LANGUAGES = [
    "eng", "vi", "ru", "th", "sw", "ga", "he", "zh", "fr", "hi", "ar"
]

PARTS = [1, 2, 3, 4, 5, 6]


# -----------------------------
# Difference helpers
# -----------------------------
def safe_text(value):
    """Convert NaN/None to empty string and everything else to string."""
    if pd.isna(value):
        return ""
    return str(value)


def get_removed_text(original, removed):
    """
    Return text that appears in original passage but not in passage_removed.

    This works well when passage_removed is produced by deleting injected text
    from the original passage. It uses SequenceMatcher to identify deleted or
    replaced spans from the original.
    """
    original = safe_text(original)
    removed = safe_text(removed)

    matcher = difflib.SequenceMatcher(None, original, removed)
    deleted_chunks = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            chunk = original[i1:i2].strip()
            if chunk:
                deleted_chunks.append(chunk)

    return " ".join(deleted_chunks).strip()


def get_diff_summary(original, removed):
    """Return a short human-readable summary of whether the passage changed."""
    original = safe_text(original)
    removed = safe_text(removed)

    if original == removed:
        return "unchanged"
    if not removed:
        return "passage_removed_empty"
    if not original:
        return "passage_empty"
    return "changed"


def process_file(input_path, output_path):
    """Read one CSV, add difference columns, and write it to output_path."""
    df = pd.read_csv(input_path)

    required_cols = {"passage", "passage_removed"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[SKIP] {input_path} missing columns: {sorted(missing)}")
        return None

    df["removed_difference"] = df.apply(
        lambda row: get_removed_text(row["passage"], row["passage_removed"]),
        axis=1,
    )

    df["diff_status"] = df.apply(
        lambda row: get_diff_summary(row["passage"], row["passage_removed"]),
        axis=1,
    )

    df["passage_char_len"] = df["passage"].apply(lambda x: len(safe_text(x)))
    df["passage_removed_char_len"] = df["passage_removed"].apply(lambda x: len(safe_text(x)))
    df["removed_difference_char_len"] = df["removed_difference"].apply(len)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    changed = (df["diff_status"] == "changed").sum()
    unchanged = (df["diff_status"] == "unchanged").sum()
    print(f"[OK] {input_path} -> {output_path} | changed={changed}, unchanged={unchanged}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare passage and passage_removed by language profile."
    )
    parser.add_argument("--input-root", default=str(INPUT_ROOT), help="Root input folder")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Root output folder")
    parser.add_argument("--year", default=YEAR, help="TREC DL year, e.g. 2021 or 2022")
    parser.add_argument("--profile", default=PROFILE, help="Language folder suffix, e.g. instruct_rem or qp_rem")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGUAGES,
        help="Languages to process, e.g. eng vi ru th",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        type=int,
        default=PARTS,
        help="CSV part numbers to process",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Also write one merged CSV containing all processed rows",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    year = str(args.year)

    all_frames = []

    for lang in args.languages:
        lang_folder = f"{lang}_{args.profile}"

        for part in args.parts:
            input_path = (
                input_root
                / f"trec_dl_{year}"
                / lang_folder
                / f"all_topics_trecdl_{year}_part{part}.csv"
            )

            output_path = (
                output_root
                / f"trec_dl_{year}"
                / lang_folder
                / f"all_topics_trecdl_{year}_part{part}_with_removed_diff.csv"
            )

            if not input_path.exists():
                print(f"[MISSING] {input_path}")
                continue

            processed_df = process_file(input_path, output_path)
            if processed_df is not None:
                processed_df.insert(0, "language", lang)
                processed_df.insert(1, "profile", args.profile)
                processed_df.insert(2, "part", part)
                all_frames.append(processed_df)

    if args.merged and all_frames:
        merged_df = pd.concat(all_frames, ignore_index=True)
        merged_path = (
            output_root
            / f"trec_dl_{year}"
            / f"all_languages_{args.profile}_with_removed_diff.csv"
        )
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f"[MERGED] {merged_path}")


if __name__ == "__main__":
    main()
