# output_writers.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from scripts.csv_helpers import write_chunked_csv

def _ensure_parent(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

def write_df(df: pd.DataFrame, out_path: Path, *, index=False, encoding="utf-8"):
    _ensure_parent(out_path); df.to_csv(out_path, index=index, encoding=encoding)

def write_df_and_chunks(df: pd.DataFrame, out_path: Path, *, chunk_dir: Path, prefix: str, size: int = 500):
    write_df(df, out_path); chunk_dir.mkdir(parents=True, exist_ok=True); write_chunked_csv(df, chunk_dir, prefix, size)

def write_confusion_outputs(cm, cm_pct, counts_path: Path, pct_path: Path) -> None:
    counts_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) counts as-is
    cm.to_csv(counts_path)

    # 2) pct with strict schema:
    #    header: NIST,0,1,2,3
    #    rows:   index as first col (NIST), values rounded to 2 decimals
    ordered_cols = [0, 1, 2, 3]

    pct_df = cm_pct.copy()

    # ensure column order and that all 0..3 columns exist
    for c in ordered_cols:
        if c not in pct_df.columns:
            pct_df[c] = 0.0
    pct_df = pct_df.reindex(columns=ordered_cols)

    # set first column name
    pct_df.index.name = "NIST"

    # round to 2 decimals (values already multiplied by 100 in your pipeline)
    pct_df = pct_df.round(2)

    # write with 2-decimal formatting
    pct_df.to_csv(pct_path, float_format="%.2f")

def write_metrics(metrics_df: pd.DataFrame, out_path: Path):
    write_df(metrics_df, out_path)

def write_unparseable_rows(df: pd.DataFrame, out_csv_path: Path, *, chunk_dir: Path, chunk_prefix="unparseable"):
    write_df_and_chunks(df, out_csv_path, chunk_dir=chunk_dir, prefix=chunk_prefix)

def write_missing_nist(df: pd.DataFrame, out_csv_path: Path, *, chunk_dir: Path, chunk_prefix="nist_not_joined"):
    write_df_and_chunks(df, out_csv_path, chunk_dir=chunk_dir, prefix=chunk_prefix)

def write_llm_extra(df: pd.DataFrame, out_csv_path: Path, *, chunk_dir: Path, chunk_prefix="llm_not_in_nist"):
    write_df_and_chunks(df, out_csv_path, chunk_dir=chunk_dir, prefix=chunk_prefix)

def save_heatmap(plt_module, out_svg_path: Path, *, dpi=200, tight=True, show=True):
    _ensure_parent(out_svg_path)
    if tight: plt_module.tight_layout()
    plt_module.savefig(out_svg_path, dpi=dpi)
    if show:  plt_module.show()
