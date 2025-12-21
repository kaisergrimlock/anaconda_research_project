#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import pandas as pd


# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
MODEL = "qwen3-32b-v1"  # e.g. "gpt-oss-20b", "qwen3-32b-v1", ...

# Where the LLM label CSVs live
LABEL_DIR = Path("outputs") / "llm_label" / f"trec_dl_{TREC_DL_YEAR}" / MODEL

# Restrict which variants are required to be "higher than raw".
# Set to [] or None to include all non-raw files found.
TARGET_LANGS: List[str] = ["eng", "eng_word", "vi", "vi_word", "ru", "ru_word"]

# Output directory
BASELINE_DIR = Path("outputs") / "baseline" / TREC_DL_YEAR / MODEL
PROJECT_ROOT = BASELINE_DIR.parents[3]  # .../<project_root>/outputs/...
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / f"trec_dl_{TREC_DL_YEAR}" / MODEL / "higher_than_raw_all_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers (mirrors your existing style)
# =========================
def parse_lang_from_filename(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 5 and parts[1] == "trecdl":
        return "_".join(parts[3:-1])
    return parts[-2] if len(parts) >= 2 else "unknown"


def pick_id_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["pid_qrels", "pid", "docid", "pid_resolved", "passage_id"]:
        if col in df.columns:
            return col
    return None


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_key_series(df: pd.DataFrame, id_col: str) -> pd.Series:
    return df["qid"].astype(str) + "|" + df[id_col].astype(str)


# =========================
# Main
# =========================
def main() -> None:
    raw_path = LABEL_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_raw_labels.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw labels file: {raw_path}")

    df_raw = load_df(raw_path)

    if "qid" not in df_raw.columns or "llm_relevance" not in df_raw.columns:
        raise ValueError(f"{raw_path.name}: must contain 'qid' and 'llm_relevance' columns")

    raw_id_col = pick_id_col(df_raw)
    if raw_id_col is None:
        raise ValueError(f"{raw_path.name}: could not find any pid/docid column")

    # Build raw llm lookup
    df_raw = df_raw.copy()
    df_raw["llm_relevance"] = pd.to_numeric(df_raw["llm_relevance"], errors="coerce")
    raw_keys = build_key_series(df_raw, raw_id_col)
    raw_llm_map = pd.Series(df_raw["llm_relevance"].values, index=raw_keys.values).to_dict()

    # Discover variant files
    pattern = f"{MODEL}_trecdl_{TREC_DL_YEAR}_*_labels.csv"
    variant_paths: List[Path] = []
    for p in sorted(LABEL_DIR.glob(pattern)):
        if p.name.endswith("_raw_labels.csv"):
            continue
        lang = parse_lang_from_filename(p)
        if TARGET_LANGS and (lang not in TARGET_LANGS):
            continue
        variant_paths.append(p)

    if not variant_paths:
        print("[ERROR] No variant files found (check LABEL_DIR / TARGET_LANGS).")
        return

    # For each variant: compute the set of keys where variant_llm > raw_llm
    qualifying_sets: List[Set[str]] = []
    lang_by_path = {}

    for vp in variant_paths:
        lang = parse_lang_from_filename(vp)
        lang_by_path[vp] = lang

        df_v = load_df(vp)
        if "qid" not in df_v.columns or "llm_relevance" not in df_v.columns:
            print(f"[SKIP] {vp.name}: missing 'qid' or 'llm_relevance'")
            continue

        v_id_col = pick_id_col(df_v)
        if v_id_col is None:
            print(f"[SKIP] {vp.name}: missing pid/docid column")
            continue

        df_v = df_v.copy()
        df_v["llm_relevance"] = pd.to_numeric(df_v["llm_relevance"], errors="coerce")
        v_keys = build_key_series(df_v, v_id_col)

        # vectorized compare to raw
        raw_llm_for_v = v_keys.map(lambda k: raw_llm_map.get(k, pd.NA))
        mask = (
            df_v["llm_relevance"].notna()
            & raw_llm_for_v.notna()
            & (df_v["llm_relevance"] > raw_llm_for_v.astype(float))
        )

        qset = set(v_keys[mask].astype(str).tolist())
        qualifying_sets.append(qset)
        print(f"[INFO] {lang}: {len(qset):,} keys where variant LLM > raw LLM")

    if not qualifying_sets:
        print("[ERROR] No qualifying sets produced (all variants were skipped or empty).")
        return

    # Intersection across all variants => must be higher-than-raw in EVERY variant
    common_keys = set.intersection(*qualifying_sets) if qualifying_sets else set()
    print(f"[INFO] Keys higher-than-raw across ALL variants: {len(common_keys):,}")

    if not common_keys:
        print("[WARN] Intersection is empty; nothing to write.")
        return

    # Write filtered RAW (original lines only)
    df_raw_out = df_raw.copy()
    raw_keys2 = build_key_series(df_raw_out, raw_id_col)
    raw_mask2 = raw_keys2.isin(common_keys)
    raw_filtered = df_raw_out.loc[raw_mask2].copy()
    raw_filtered.to_csv(OUT_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_raw_higher_all_variants.csv", index=False)
    print(f"[OK] Wrote RAW filtered rows: {len(raw_filtered):,}")

    # Write filtered rows for each variant (original lines only; no new columns)
    for vp in variant_paths:
        lang = lang_by_path[vp]
        df_v = load_df(vp)
        v_id_col = pick_id_col(df_v)
        if v_id_col is None or "qid" not in df_v.columns:
            print(f"[SKIP] {vp.name}: cannot filter (missing qid or pid/docid)")
            continue

        v_keys = build_key_series(df_v, v_id_col)
        v_filtered = df_v.loc[v_keys.isin(common_keys)].copy()

        out_path = OUT_DIR / vp.name.replace("_labels.csv", "_higher_all_variants_labels.csv")
        v_filtered.to_csv(out_path, index=False)
        print(f"[OK] Wrote {lang} filtered rows: {len(v_filtered):,} -> {out_path}")

    print(f"[DONE] Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
