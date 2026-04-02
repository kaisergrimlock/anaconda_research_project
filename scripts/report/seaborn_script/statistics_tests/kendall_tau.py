import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import kendalltau, spearmanr

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TREC_DL_YEAR = "2022"

SIMILARITY_ROOT = Path("similarity_outputs_qwen")
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"

OUT_DIR = Path("figures") / TREC_DL_YEAR / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add suffixes if needed, for example: ["", "_instruct"]
LANG_SUFFIXES = [""]

# Examples:
# MODEL_FILTER = "gpt"
# MODEL_FILTER = "llama"
# MODEL_FILTER = "qwen"
MODEL_FILTER = "qwen"


def find_similarity_files() -> Dict[str, Path]:
    files = {}
    for f in SIMILARITY_ROOT.glob("*_detailed_similarity.csv"):
        lang = f.name.replace("_detailed_similarity.csv", "")
        files[lang] = f
    return files


def find_label_files(model_filter: str) -> Dict[str, List[Path]]:
    model_files: Dict[str, List[Path]] = {}
    model_filter = model_filter.lower()

    for model_dir in LABEL_ROOT.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        if model_filter not in model_name.lower():
            continue

        csv_files = list(model_dir.glob(f"{model_name}_trecdl_{TREC_DL_YEAR}_*_labels.csv"))
        if csv_files:
            model_files[model_name] = csv_files

    return model_files


def get_lang_from_filename(file_path: Path, model: str) -> Optional[str]:
    fname = file_path.name
    pattern = rf"^{re.escape(model)}_trecdl_\d{{4}}_(.+?)_labels\.csv$"
    m = re.search(pattern, fname)
    return m.group(1) if m else None


def expand_lang_candidates(lang: str, suffixes: List[str]) -> List[str]:
    candidates = [lang]
    for suffix in suffixes:
        candidate = f"{lang}{suffix}" if suffix else lang
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def build_label_lookup(label_files: Dict[str, List[Path]]) -> Dict[str, Dict[str, Path]]:
    lookup: Dict[str, Dict[str, Path]] = {}

    for model, files in label_files.items():
        lookup[model] = {}
        for f in files:
            lang_name = get_lang_from_filename(f, model)
            if lang_name:
                lookup[model][lang_name] = f

    return lookup


def compute_avg_bleu(sim_file: Path) -> float:
    df = pd.read_csv(sim_file)
    df["bleu"] = pd.to_numeric(df["bleu"], errors="coerce")
    df = df.dropna(subset=["bleu"])
    return df["bleu"].mean()


def compute_avg_rouge_l_f1(sim_file: Path) -> float:
    df = pd.read_csv(sim_file)

    if "rouge_l_f1" not in df.columns:
        raise KeyError(
            f"'rouge_l_f1' column not found in {sim_file}. "
            f"Available columns: {list(df.columns)}"
        )

    df["rouge_l_f1"] = pd.to_numeric(df["rouge_l_f1"], errors="coerce")
    df = df.dropna(subset=["rouge_l_f1"])
    return df["rouge_l_f1"].mean()


def compute_mean_diff(label_file: Path) -> float:
    df = pd.read_csv(label_file)

    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["llm_relevance"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    df = df.dropna(subset=["relevance", "llm_relevance"]).copy()
    df["mean_diff"] = df["llm_relevance"] - df["relevance"]

    return df["mean_diff"].mean()


def main() -> None:
    similarity_files = find_similarity_files()
    label_files = find_label_files(MODEL_FILTER)
    label_lookup = build_label_lookup(label_files)

    if not similarity_files:
        raise RuntimeError(f"No similarity files found in: {SIMILARITY_ROOT}")

    if not label_lookup:
        raise RuntimeError(f"No label files found for MODEL_FILTER='{MODEL_FILTER}'")

    out_csv = OUT_DIR / f"language_similarity_vs_meandiff_{MODEL_FILTER}.csv"
    rows = []

    investigated_similarity_files = set()
    investigated_label_files = set()

    for model, model_lang_map in label_lookup.items():
        for base_lang, sim_file in similarity_files.items():
            candidates = expand_lang_candidates(base_lang, LANG_SUFFIXES)

            matched_lang = None
            matched_label_file = None

            for candidate in candidates:
                if candidate in model_lang_map:
                    matched_lang = candidate
                    matched_label_file = model_lang_map[candidate]
                    break

            if matched_label_file is None:
                print(f"[SKIP] {model} | {base_lang}: no label file for candidates {candidates}")
                continue

            investigated_similarity_files.add(str(sim_file))
            investigated_label_files.add(str(matched_label_file))

            avg_bleu = compute_avg_bleu(sim_file)
            avg_rouge_l_f1 = compute_avg_rouge_l_f1(sim_file)
            avg_mean_diff = compute_mean_diff(matched_label_file)

            rows.append({
                "model_filter": MODEL_FILTER,
                "model": model,
                "base_lang": base_lang,
                "label_lang": matched_lang,
                "similarity_file": str(sim_file),
                "label_file": str(matched_label_file),
                "avg_bleu": avg_bleu,
                "avg_rouge_l_f1": avg_rouge_l_f1,
                "avg_mean_diff": avg_mean_diff,
            })

            print(
                f"[INFO] {model} | base={base_lang} | label={matched_lang} "
                f"| bleu={avg_bleu:.6f} | rouge_l_f1={avg_rouge_l_f1:.6f} "
                f"| mean_diff={avg_mean_diff:.6f}"
            )
            print(f"       similarity_file: {sim_file}")
            print(f"       label_file:      {matched_label_file}")

    if not rows:
        raise RuntimeError(f"No rows produced for MODEL_FILTER='{MODEL_FILTER}'")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # BLEU vs mean-diff
    tau_bleu, p_tau_bleu = kendalltau(df["avg_bleu"], df["avg_mean_diff"])
    rho_bleu, p_rho_bleu = spearmanr(df["avg_bleu"], df["avg_mean_diff"])

    # ROUGE-L F1 vs mean-diff
    tau_rouge, p_tau_rouge = kendalltau(df["avg_rouge_l_f1"], df["avg_mean_diff"])
    rho_rouge, p_rho_rouge = spearmanr(df["avg_rouge_l_f1"], df["avg_mean_diff"])

    print(f"\nModel filter: {MODEL_FILTER}")

    print("\nCorrelation between BLEU and mean-diff:")
    print(f"Kendall tau: {tau_bleu}")
    print(f"p-value: {p_tau_bleu}")
    print(f"Spearman rho: {rho_bleu}")
    print(f"p-value: {p_rho_bleu}")

    print("\nCorrelation between ROUGE-L F1 and mean-diff:")
    print(f"Kendall tau: {tau_rouge}")
    print(f"p-value: {p_tau_rouge}")
    print(f"Spearman rho: {rho_rouge}")
    print(f"p-value: {p_rho_rouge}")

    print("\nInvestigated similarity files:")
    for f in sorted(investigated_similarity_files):
        print(f"  - {f}")

    print("\nInvestigated label files:")
    for f in sorted(investigated_label_files):
        print(f"  - {f}")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()