from pathlib import Path
import csv
import json


BASE_DIR = Path("retrieved")

YEAR = "2021"
LANG = [
        "eng", "raw", "vi", "th", "fr", "ru", "he", "sw", "ga", "hi", "zh",
         "eng_instruct", "vi_instruct", "ar_instruct", "fr_instruct", "th_instruct", "ru_instruct", "he_instruct", "sw_instruct", "ga_instruct", "hi_instruct", "zh_instruct",
        ]

KEEP_ALL_COLUMNS = False
PARTS = range(7)  # 0 to 6


def build_input_paths(year: str, language: str) -> list[Path]:
    """
    Build the list of input CSV paths for one language.

    Expected structure:
    retrieved/trec_dl_{year}/{language}/all_topics_trecdl_{year}_part0.csv
    ...
    retrieved/trec_dl_{year}/{language}/all_topics_trecdl_{year}_part6.csv
    """
    base_lang_dir = BASE_DIR / f"trec_dl_{year}" / language
    return [
        base_lang_dir / f"all_topics_trecdl_{year}_part{i}.csv"
        for i in PARTS
    ]


def build_output_path(year: str) -> Path:
    """
    Build the combined JSONL output path.

    Expected structure:
    retrieved/jsonl/{year}/trec_dl_{year}.jsonl
    """
    return BASE_DIR / "jsonl" / year / f"trec_dl_{year}.jsonl"


def safe_int(value):
    """
    Convert value to int if possible, otherwise return None/original value.
    """
    try:
        return int(value) if value is not None and value != "" else None
    except (ValueError, TypeError):
        return value


def convert_csv_to_jsonl(
    input_file: Path,
    f_out,
    language: str,
    keep_all_columns: bool = False
) -> None:
    """
    Convert one CSV file and write its rows into an already-open JSONL file.

    If keep_all_columns is False:
        - keep selected columns only
        - add language to pid

    If keep_all_columns is True:
        - keep all columns
        - rename pid -> id
        - add language to id
        - remove original pid
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)

        for row in reader:
            new_id = f"{row['pid']}_{language}"

            if keep_all_columns:
                row["id"] = new_id
                row.pop("pid", None)
                json_obj = row
            else:
                json_obj = {
                    "pid": new_id,
                    "qid": row.get("qid"),
                    "query": row.get("query"),
                    "passage": row.get("passage"),
                    "relevance": safe_int(row.get("relevance")),
                    "passage_injected": row.get("passage_injected"),
                }

            f_out.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


def main():
    output_file = build_output_path(YEAR)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Combined output: {output_file}")

    with output_file.open("w", encoding="utf-8") as f_out:
        for language in LANG:
            print(f"\nProcessing language: {language}")

            input_files = build_input_paths(YEAR, language)

            for input_file in input_files:
                print(f"Input: {input_file}")

                try:
                    convert_csv_to_jsonl(
                        input_file=input_file,
                        f_out=f_out,
                        language=language,
                        keep_all_columns=KEEP_ALL_COLUMNS,
                    )
                    print(f"Done: {input_file.name}")
                except FileNotFoundError as e:
                    print(f"Skipped: {e}")

    print("\nFinished writing combined JSONL file.")


if __name__ == "__main__":
    main()