#!/usr/bin/env python3
import re
import sys
import csv
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Config ----------
TREC_DL_YEAR = "2023"
MODEL = "gpt-oss-20b"
LANG = "eng"
NIST_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
if LANG != "raw":
    LLM_FILE   = Path("outputs/llm_label/" + MODEL + "/" + MODEL + "_trec_dl_" + TREC_DL_YEAR + "_" + LANG  + "_raw.csv")
    OUT_DIR    = Path("outputs/baseline/" + TREC_DL_YEAR + "/" + LANG)
else:
    LLM_FILE   = Path("outputs/llm_label/" + MODEL + "/" + MODEL + "_trec_dl_" + TREC_DL_YEAR + "_raw.csv")
    OUT_DIR    = Path("outputs/baseline/" + TREC_DL_YEAR + "/raw")


OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT    = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_PNG    = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"

LABEL_COL_CHOICES = ["relevance", "label"]
LABELS = [0, 1, 2, 3]
MAP_INVALID_TO_ZERO = False
# ---------------------------

# ---- Bump CSV field size limit (fixes "field larger than field limit (131072)") ----
def _bump_field_limit():
    limit = getattr(sys, "maxsize", 2_000_000_000)
    while limit >= 131_072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2
    csv.field_size_limit(1_000_000)

_bump_field_limit()
# -----------------------------------------------------------------------------------

def read_csv_smart(path: Path) -> pd.DataFrame:
    # Robust loader: tolerate huge cells + odd lines
    return pd.read_csv(
        path,
        engine="python",   # uses stdlib csv (honors field_size_limit)
        dtype=str,         # keep everything as string
        on_bad_lines="skip"
    )

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.strip().lower(): c for c in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in cols:
            return cols[key]
    return None

def pick_qid_col(df: pd.DataFrame) -> str:
    c = pick_col(df, ["qid", "topic", "topic_id"])
    if not c:
        c = pick_col(df, ["query", "question", "topic_text"])  # last resort
        if not c:
            raise KeyError(f"No qid/topic/query column in {list(df.columns)}")
    return c

def pick_pid_col(df: pd.DataFrame) -> str:
    c = pick_col(df, ["pid", "pid_resolved", "pid_qrels", "docid", "doc_id"])
    if not c:
        raise KeyError(f"No pid/pid_resolved/pid_qrels/docid column in {list(df.columns)}")
    return c

def pick_label_col(df: pd.DataFrame) -> str:
    c = pick_col(df, LABEL_COL_CHOICES)
    if not c:
        raise KeyError(f"Neither 'relevance' nor 'label' in {list(df.columns)}")
    return c

_digit_0_3 = re.compile(r"\b([0-3])\b")

def parse_label(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s in {"0", "1", "2", "3"}:
        return int(s)
    m = _digit_0_3.search(s)
    return int(m.group(1)) if m else None

# 1) Load NIST (all CSVs)
nist_files = sorted(NIST_DIR.rglob("*.csv"))
if not nist_files:
    raise FileNotFoundError(f"No CSV files under {NIST_DIR}")

print(f"[NIST] Found {len(nist_files)} files under {NIST_DIR}")
nist_parts = []
for i, fp in enumerate(nist_files, start=1):
    df = read_csv_smart(fp)
    qcol  = pick_qid_col(df)
    pcol  = pick_pid_col(df)
    lcol  = pick_label_col(df)

    part = df[[qcol, pcol, lcol]].rename(columns={qcol: "qid", pcol: "pid", lcol: "NIST"})
    part["qid"]  = part["qid"].astype(str).str.strip()
    part["pid"]  = part["pid"].astype(str).str.strip()
    part["NIST"] = part["NIST"].apply(parse_label).fillna(0).astype(int)

    nist_parts.append(part)
    if i % 10 == 0 or i == len(nist_files):
        print(f"[NIST] Parsed {i}/{len(nist_files)} files… rows so far={sum(len(x) for x in nist_parts):,}")

nist = pd.concat(nist_parts, ignore_index=True)\
         .drop_duplicates(subset=["qid", "pid"], keep="first")

print(f"[NIST] Total rows={len(nist):,} (after de-dup on qid,pid)")

# 2) Load LLM (with ids)
# 2) Load LLM (with or without qid)
llm = read_csv_smart(LLM_FILE)

# choose pid + label cols first (always required)
pcol = pick_pid_col(llm)
lcol = pick_label_col(llm)

# Try to find a qid-like column in the LLM file
try:
    qcol_llm = pick_qid_col(llm)  # may raise if not present
    have_llm_qid = True
except KeyError:
    have_llm_qid = False

# Parse/restrict to needed cols
if have_llm_qid:
    llm = llm[[qcol_llm, pcol, lcol]].rename(columns={qcol_llm: "qid", pcol: "pid", lcol: "LLM_raw"})
else:
    # No qid in LLM — keep pid + label for now
    llm = llm[[pcol, lcol]].rename(columns={pcol: "pid", lcol: "LLM_raw"})

llm["pid"] = llm["pid"].astype(str).str.strip()
llm["LLM_parsed"] = llm["LLM_raw"].apply(parse_label)

total_llm = len(llm)
parsed_ok = llm["LLM_parsed"].notna().sum()
parsed_bad = total_llm - parsed_ok
print(f"[LLM ] rows={total_llm:,} | parsed={parsed_ok:,} | unparseable={parsed_bad:,}")

# Drop invalid labels (or map to 0 if you set MAP_INVALID_TO_ZERO)
if MAP_INVALID_TO_ZERO:
    llm["LLM"] = llm["LLM_parsed"].fillna(0).astype(int)
else:
    llm = llm[llm["LLM_parsed"].notna()].copy()
    llm["LLM"] = llm["LLM_parsed"].astype(int)

# If LLM lacks qid, infer qids from NIST by pid (replicate across all qids for that pid)
if not have_llm_qid:
    # Build (pid -> qid) mapping from NIST
    nist_pid_qids = nist[["pid", "qid"]].drop_duplicates()
    before = len(llm)
    llm = llm.merge(nist_pid_qids, on="pid", how="inner")
    after = len(llm)
    covered_pids = nist_pid_qids["pid"].nunique()
    print(f"[LLM ] No qid column found; expanded by NIST pid→qid map: rows {before:,} -> {after:,} "
          f"(unique NIST pids={covered_pids:,})")
else:
    # Normalize qid text if present
    llm["qid"] = llm["qid"].astype(str).str.strip()

total_llm = len(llm)
parsed_ok = llm["LLM_parsed"].notna().sum()
parsed_bad = total_llm - parsed_ok
print(f"[LLM ] rows={total_llm:,} | parsed={parsed_ok:,} | unparseable={parsed_bad:,}")

if MAP_INVALID_TO_ZERO:
    llm["LLM"] = llm["LLM_parsed"].fillna(0).astype(int)
else:
    llm = llm[llm["LLM_parsed"].notna()].copy()
    llm["LLM"] = llm["LLM_parsed"].astype(int)

# 3) Join & sanity info (on qid, pid)
paired = nist.merge(llm[["qid", "pid", "LLM"]], on=["qid", "pid"], how="inner")
print(f"[JOIN] Pairs after join (qid,pid): {len(paired):,}")

# 4) Confusion matrix (counts)
cm = pd.crosstab(
    index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
    columns=pd.Categorical(paired["LLM"],  categories=LABELS, ordered=True),
    dropna=False
)
cm.index.name = "NIST"; cm.columns.name = "LLM"

# 5) Row-normalized (%)
cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0

# 6) Save + plot
OUT_DIR.mkdir(parents=True, exist_ok=True)
cm.to_csv(OUT_COUNTS)
cm_pct.round(2).to_csv(OUT_PCT)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
plt.title("Confusion Matrix: NIST vs LLM (counts)")
plt.ylabel("NIST label"); plt.xlabel("LLM label")
plt.tight_layout(); plt.savefig(OUT_PNG, dpi=200); plt.show()

print(f"[DONE] Wrote counts to: {OUT_COUNTS}")
print(f"[DONE] Wrote row-% to:  {OUT_PCT}")
print(f"[DONE] Saved heatmap to: {OUT_PNG}")
