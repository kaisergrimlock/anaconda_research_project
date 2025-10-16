#!/usr/bin/env python3
from __future__ import annotations
import re, sys, csv
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Config ----------
TREC_DL_YEAR = "2023"
LANG = "eng"

# Ground truth judged file (single file)
NIST_FILE = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/all_topics_trecdl_{TREC_DL_YEAR}_part40.csv")

# LLM predictions (this file already has qid)
LLM_FILE  = Path("outputs/llm_label/trec_dl_2023_fake_link_injected.csv")

OUT_DIR   = Path(f"outputs/baseline/{TREC_DL_YEAR}/{LANG}")
OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT    = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_PNG    = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"

LABELS = [0, 1, 2, 3]
LABEL_COL_CHOICES = ["relevance", "label"]
MAP_INVALID_TO_ZERO = False  # if False, drop rows where label can't be parsed
# ---------------------------

def _bump_field_limit():
    limit = getattr(sys, "maxsize", 2_000_000_000)
    while limit >= 131_072:
        try:
            csv.field_size_limit(limit); return
        except OverflowError:
            limit //= 2
    csv.field_size_limit(1_000_000)
_bump_field_limit()

def read_csv_smart(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", dtype=str, on_bad_lines="skip")

def first_nonempty(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row-wise coalesce: first non-empty among given columns."""
    vals = pd.Series([""] * len(df), index=df.index, dtype="object")
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            vals = vals.mask(vals.ne("") | s.eq(""), other=vals)  # keep existing non-empty
            vals = vals.where(vals.ne(""), s, axis=0)             # fill empties with s
    return vals

def unify_qid(df: pd.DataFrame) -> pd.Series:
    for c in ["qid", "topic", "topic_id"]:
        if c in df.columns:
            return df[c].astype(str).str.strip()
    # last-resort fallbacks (not recommended)
    for c in ["query", "question", "topic_text"]:
        if c in df.columns:
            return df[c].astype(str).str.strip()
    raise KeyError(f"No qid/topic column in {list(df.columns)}")

def unify_pid_rowwise(df: pd.DataFrame) -> pd.Series:
    # IMPORTANT: same order on both sides
    cols = ["pid_resolved", "pid_qrels", "docid", "doc_id", "pid"]
    pid = first_nonempty(df, cols).astype(str).str.strip()
    # normalize trivial quirks if any (no-op here, but place for custom rules)
    return pid

_digit_0_3 = re.compile(r"\b([0-3])\b")
def parse_label(value):
    if pd.isna(value): return None
    s = str(value).strip()
    if s in {"0","1","2","3"}: return int(s)
    m = _digit_0_3.search(s)
    return int(m.group(1)) if m else None

def extract_label(df: pd.DataFrame, newcol: str) -> pd.DataFrame:
    lcol = next((c for c in LABEL_COL_CHOICES if c in df.columns), None)
    if not lcol:
        raise KeyError(f"Neither 'relevance' nor 'label' in {list(df.columns)}")
    out = df.copy()
    out[newcol] = out[lcol].apply(parse_label)
    if MAP_INVALID_TO_ZERO:
        out[newcol] = out[newcol].fillna(0).astype(int)
    else:
        out = out[out[newcol].notna()].copy()
        out[newcol] = out[newcol].astype(int)
    return out

def main():
    # 1) NIST
    if not NIST_FILE.exists():
        raise FileNotFoundError(f"NIST/judged file not found: {NIST_FILE}")
    nist_raw = read_csv_smart(NIST_FILE)
    nist = nist_raw.copy()
    nist["qid"] = unify_qid(nist_raw)
    nist["pid"] = unify_pid_rowwise(nist_raw)         # <<< row-wise coalesced pid
    nist = extract_label(nist, "NIST")
    nist = nist[["qid","pid","NIST"]].query("qid != '' and pid != ''").drop_duplicates(["qid","pid"])
    print(f"[NIST] rows={len(nist):,} | unique (qid,pid)={nist[['qid','pid']].drop_duplicates().shape[0]:,}")

    # 2) LLM (your file already has qid)
    if not LLM_FILE.exists():
        raise FileNotFoundError(f"LLM file not found: {LLM_FILE}")
    llm_raw = read_csv_smart(LLM_FILE)
    llm = llm_raw.copy()
    llm["qid"] = unify_qid(llm_raw)
    llm["pid"] = unify_pid_rowwise(llm_raw)           # <<< row-wise coalesced pid
    llm = extract_label(llm, "LLM")
    llm = llm[["qid","pid","LLM"]].query("qid != '' and pid != ''").drop_duplicates(["qid","pid"])
    print(f"[LLM ] rows={len(llm):,} | unique (qid,pid)={llm[['qid','pid']].drop_duplicates().shape[0]:,}")

    # 3) Join
    paired = nist.merge(llm, on=["qid","pid"], how="inner")
    print(f"[JOIN] pairs (qid,pid)={len(paired):,}")

    # Optional: debug when join is smaller than expected
    expected = len(llm)
    if len(paired) != expected:
        nkeys = set(map(tuple, nist[["qid","pid"]].itertuples(index=False, name=None)))
        lkeys = set(map(tuple, llm[["qid","pid"]].itertuples(index=False, name=None)))
        missing = sorted(lkeys - nkeys)[:10]
        print(f"[DIAG] LLM keys not in NIST: {len(lkeys - nkeys):,} (showing up to 10)")
        for qid, pid in missing:
            print("   -", qid, pid)

    # 4) Confusion matrices
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if paired.empty:
        cm = pd.DataFrame(0, index=pd.Index(LABELS, name="NIST"), columns=pd.Index(LABELS, name="LLM"))
        cm.to_csv(OUT_COUNTS); cm.to_csv(OUT_PCT)
        print("[WARN] Empty join. Zero matrices written; skipping plot.")
        return

    cm = pd.crosstab(
        index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM"],  categories=LABELS, ordered=True),
        dropna=False
    )
    cm.index.name = "NIST"; cm.columns.name = "LLM"

    cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0
    cm.to_csv(OUT_COUNTS)
    cm_pct.round(2).to_csv(OUT_PCT)

    # 5) Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title("Confusion Matrix: NIST vs LLM (counts)")
    plt.ylabel("NIST label"); plt.xlabel("LLM label")
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=200); plt.show()

    print(f"[DONE] Wrote counts: {OUT_COUNTS}")
    print(f"[DONE] Wrote row-% : {OUT_PCT}")
    print(f"[DONE] Saved plot  : {OUT_PNG}")

if __name__ == "__main__":
    main()
