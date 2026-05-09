"""
Create a CSV table with FP and FN for each model/language in a LANG_PROFILE.

Expected input layout:
  outputs/llm_label/trec_dl_<YEAR>/<MODEL>/<MODEL>_trecdl_<YEAR>_<LANG>_labels.csv

Expected columns:
  relevance      = NIST/gold relevance label
  llm_relevance  = LLM predicted relevance label

Binary conversion:
  relevant     = label >= THRESHOLD
  not relevant = label < THRESHOLD
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# -------------------------------------------------------------------
# Project-root setup, so this script can be run from statistics_tests/
# or from the project root.
# -------------------------------------------------------------------
cwd = Path.cwd().resolve()

if cwd.name == "statistics_tests":
    os.chdir(cwd.parents[3])  # project root: anaconda_research_project

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
TREC_DL_YEAR = "2022"
LANG_PROFILE = "qp_rem"  # change to the profile you want
LANGS = get_langs(LANG_PROFILE)

LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"

ALLOWED_MODELS = {
    "gpt-oss-20b",
    "qwen3-32b-v1",
    "llama3_8b"
}

THRESHOLD = 2
OUTPUT_FILE = Path("outputs") / "tables" / f"fp_fn_{LANG_PROFILE}_{TREC_DL_YEAR}.csv"

GOLD_COL = "relevance"
LLM_COL = "llm_relevance"
LABELS = [0, 1, 2, 3]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def find_llm_files() -> Dict[str, List[Path]]:
    """
    Return model -> list of CSV files under LABEL_ROOT.
    The language itself is filtered later using LANG_PROFILE.
    """
    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"LABEL_ROOT not found: {LABEL_ROOT}")

    model_files: Dict[str, List[Path]] = {}

    for model_dir in LABEL_ROOT.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        if model_name not in ALLOWED_MODELS:
            continue

        csv_files = list(
            model_dir.glob(f"{model_name}_trecdl_{TREC_DL_YEAR}_*_labels.csv")
        )

        if csv_files:
            model_files[model_name] = csv_files
        else:
            print(f"Warning: no label CSVs found for model {model_name} in {model_dir}")

    return model_files


def get_lang_from_filename(file_path: Path, model: str) -> Optional[str]:
    """
    Extract <LANG> from:
      <MODEL>_trecdl_<YEAR>_<LANG>_labels.csv
    """
    prefix = f"{model}_trecdl_{TREC_DL_YEAR}_"
    suffix = "_labels.csv"
    name = file_path.name

    if not name.startswith(prefix) or not name.endswith(suffix):
        return None

    return name[len(prefix):-len(suffix)]


def compute_fp_fn(df: pd.DataFrame) -> tuple[int, int]:
    missing = {GOLD_COL, LLM_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

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

    total = len(df)

    FP = ((llm_bin == 1) & (gold_bin == 0)).sum()
    FN = ((llm_bin == 0) & (gold_bin == 1)).sum()

    FP = (FP / total) * 100
    FN = (FN / total) * 100

    return int(FP), int(FN)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    bump_field_limit()

    model_files = find_llm_files()
    rows = []

    print(f"Using language profile: {LANG_PROFILE}")
    print(f"Languages in profile: {LANGS}")
    print(f"Reading from: {LABEL_ROOT}")

    for model, files in model_files.items():
        for file_path in files:
            lang = get_lang_from_filename(file_path, model)

            if lang not in LANGS:
                continue

            try:
                df = pd.read_csv(file_path)
                fp, fn = compute_fp_fn(df)

                rows.append({
                    "language": lang,
                    "model": model,
                    "FP": fp,
                    "FN": fn,
                })

                print(f"[OK] {model} {lang}: FP={fp}, FN={fn}")

            except Exception as e:
                print(f"[SKIP] {model} {lang} ({file_path.name}): {e}")

    if not rows:
        raise RuntimeError(
            f"No rows produced. Check LANG_PROFILE={LANG_PROFILE}, "
            f"LABEL_ROOT={LABEL_ROOT}, and allowed models={sorted(ALLOWED_MODELS)}"
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["model", "language"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved FP/FN table to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
