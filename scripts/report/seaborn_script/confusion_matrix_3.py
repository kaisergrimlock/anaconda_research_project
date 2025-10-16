#!/usr/bin/env python3
from __future__ import annotations
import csv, sys, re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ====== EDIT THESE IF NEEDED ======
BASELINE_CSV = Path("outputs/queries/non_relevant/first_nonrelevant_with_llm_relevance.csv")
LLM_CSV      = Path("outputs/llm_label/trec_dl_2023_verbose_injected.csv")
OUT_DIR      = Path("outputs/baseline/2023/non_rel")
OUT_COUNTS   = OUT_DIR / "verbose_injection.csv"   # counts table
OUT_PCT      = OUT_DIR / "verbose_injection.csv"   # row-% table (kept as-is per your paths)
OUT_IMG      = OUT_DIR / "verbose_injection.svg"
LABELS       = [0, 1, 2, 3]
# ===================================

# Column names / aliases
COL_QUERY   = "query"
COL_PASSAGE = "passage"
COL_REL     = "relevance"
COL_LABEL   = "label"

def _bump_field_limit():
    try:
        limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
        while limit >= 131_072:
            try:
                csv.field_size_limit(limit); return
            except OverflowError:
                limit //= 2
    except Exception:
        pass
_bump_field_limit()

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", dtype=str, on_bad_lines="skip")

_digit_0_3 = re.compile(r"\b([0-3])\b")
def parse_label_any(x) -> int | None:
    if pd.isna(x): return None
    s = str(x).strip()
    if s in {"0","1","2","3"}: return int(s)
    m = _digit_0_3.search(s)
    return int(m.group(1)) if m else None

def extract_label(df: pd.DataFrame, outcol: str) -> pd.DataFrame:
    src = COL_REL if COL_REL in df.columns else (COL_LABEL if COL_LABEL in df.columns else None)
    if not src:
        raise KeyError(f"Neither '{COL_REL}' nor '{COL_LABEL}' found in columns: {list(df.columns)}")
    out = df.copy()
    out[outcol] = out[src].apply(parse_label_any)
    out = out[out[outcol].notna()].copy()
    out[outcol] = out[outcol].astype(int)
    return out

def norm_query(df: pd.DataFrame) -> pd.Series:
    if COL_QUERY not in df.columns:
        raise KeyError(f"Missing '{COL_QUERY}' in columns: {list(df.columns)}")
    # normalize whitespace for safer joins
    return df[COL_QUERY].astype(str).str.strip()

def main():
    if not BASELINE_CSV.exists():
        sys.exit(f"[FATAL] Baseline file not found: {BASELINE_CSV}")
    if not LLM_CSV.exists():
        sys.exit(f"[FATAL] LLM file not found: {LLM_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load baseline (first_nonrelevant_with_llm_relevance.csv) ---
    base_raw = read_csv(BASELINE_CSV)  # expected: qid,query,pid,passage,relevance
    base = base_raw.copy()
    base[COL_QUERY] = norm_query(base_raw)
    base = extract_label(base, "BASE")
    base = base[[COL_QUERY, "BASE"]].dropna().drop_duplicates([COL_QUERY])
    print(f"[BASE] rows={len(base):,} | unique queries={base[COL_QUERY].nunique():,}")

    # --- Load LLM predictions (trec_dl_2023_verbose_injected.csv) ---
    llm_raw = read_csv(LLM_CSV)
    llm = llm_raw.copy()
    llm[COL_QUERY] = norm_query(llm_raw)
    llm = extract_label(llm, "LLM")
    llm = llm[[COL_QUERY, "LLM"]].dropna().drop_duplicates([COL_QUERY])
    print(f"[LLM ] rows={len(llm):,} | unique queries={llm[COL_QUERY].nunique():,}")

    # --- Join on query ---
    paired = base.merge(llm, on=COL_QUERY, how="inner")
    print(f"[JOIN] pairs (by query)={len(paired):,}")

    if paired.empty:
        # write zero matrices and skip plot
        cm = pd.DataFrame(0, index=pd.Index(LABELS, name="Baseline"), columns=pd.Index(LABELS, name="LLM"))
        cm.to_csv(OUT_COUNTS)
        cm.to_csv(OUT_PCT)
        print("[WARN] Empty join. Wrote zero matrices; skipping heatmap.")
        return

    # Confusion matrix: rows=baseline, cols=LLM
    cm = pd.crosstab(
        index=pd.Categorical(paired["BASE"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM"],  categories=LABELS, ordered=True),
        dropna=False
    )
    cm.index.name = "Baseline"; cm.columns.name = "LLM"

    # Row-normalized (%)
    cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0

    # Save
    cm.to_csv(OUT_COUNTS)
    cm_pct.round(2).to_csv(OUT_PCT)

    # Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title("Confusion Matrix: Baseline vs LLM (match by query)")
    plt.ylabel("Baseline (first_nonrelevant_with_llm_relevance)")
    plt.xlabel("LLM (trec_dl_2023_verbose_injected)")
    plt.tight_layout(); plt.savefig(OUT_IMG, dpi=200); plt.show()

    print(f"[DONE] counts  → {OUT_COUNTS}")
    print(f"[DONE] row-%   → {OUT_PCT}")
    print(f"[DONE] heatmap → {OUT_IMG}")

if __name__ == "__main__":
    main()
