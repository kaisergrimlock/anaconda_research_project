#!/usr/bin/env python3
from __future__ import annotations
import re, sys, csv
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============== Config ===============
TREC_DL_YEAR = "2023"
MODEL        = "gpt-oss-20b"
LANG         = "ru"            # "eng", "vi", "raw"

# Inputs/outputs
NIST_DIR  = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
LLM_FILE  = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_raw.csv"
TOPICS_DIR  = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / LANG
TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

OUT_DIR    = Path("outputs/baseline") / TREC_DL_YEAR / LANG
OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT    = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_SVG    = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"

# NEW: diagnostics
OUT_UNPARSEABLE = OUT_DIR / "llm_unparseable_labels.csv"
OUT_UNRESOLVED  = OUT_DIR / "llm_unresolved_qid.csv"

# Label handling
LABEL_COL_CHOICES   = ["relevance", "label"]
LABELS              = [0, 1, 2, 3]
MAP_INVALID_TO_ZERO = False

# Matching behavior
ALLOW_PID_ONLY_FALLBACK = True   # try pid→qid when (pid, passage) pair not found
# =====================================

# ---- Allow huge CSV cells ----
def _bump_field_limit():
    limit = getattr(sys, "maxsize", 2_000_000_000)
    while limit >= 131_072:
        try:
            csv.field_size_limit(limit); return
        except OverflowError:
            limit //= 2
    csv.field_size_limit(1_000_000)
_bump_field_limit()

# ---- Utils ----
def read_csv_smart(path: Path) -> pd.DataFrame:
    # NOTE: on_bad_lines="skip" means truly malformed lines are dropped by pandas
    # and cannot be written out here. This file will capture logical “unparseable label”
    # rows, not physically malformed lines.
    return pd.read_csv(path, engine="python", dtype=str, on_bad_lines="skip")

def _write_chunked_csv(df: pd.DataFrame, out_dir: Path, base_name: str, chunk_size: int = 500) -> list[Path]:
    """
    Write df into multiple CSV files with at most `chunk_size` rows each,
    stored under `out_dir`. Filenames: {base_name}_part_0001.csv, ...
    Returns the list of written paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    if n == 0:
        return []
    paths = []
    num_parts = (n + chunk_size - 1) // chunk_size
    pad = max(4, len(str(num_parts)))
    for i in range(num_parts):
        start = i * chunk_size
        end   = min(start + chunk_size, n)
        part  = df.iloc[start:end]
        fp = out_dir / f"{base_name}_part_{(i+1):0{pad}d}.csv"
        part.to_csv(fp, index=False, encoding="utf-8")
        paths.append(fp)
    return paths


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.strip().lower(): c for c in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in cols:
            return cols[key]
    return None

def pick_qid_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, ["qid", "topic", "topic_id"])

def pick_pid_col(df: pd.DataFrame) -> str:
    c = pick_col(df, ["pid", "pid_resolved", "pid_qrels", "docid", "doc_id", "docno"])
    if not c: raise KeyError(f"No pid-like column in {list(df.columns)}")
    return c

def pick_label_col(df: pd.DataFrame) -> str:
    c = pick_col(df, LABEL_COL_CHOICES)
    if not c: raise KeyError(f"Neither 'relevance' nor 'label' in {list(df.columns)}")
    return c

def norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

_digit_0_3 = re.compile(r"\b([0-3])\b")
def parse_label(value):
    if value is None or (isinstance(value, float) and pd.isna(value)): return None
    s = str(value).strip()
    if s in {"0","1","2","3"}: return int(s)
    m = _digit_0_3.search(s)
    return int(m.group(1)) if m else None

# =========================
# 1) Load NIST (judged)
# =========================
nist_files = sorted(NIST_DIR.rglob("*.csv"))
if not nist_files:
    raise FileNotFoundError(f"No CSV files under {NIST_DIR}")

print(f"[NIST] Found {len(nist_files)} files under {NIST_DIR}")
parts, seen = [], 0
for i, fp in enumerate(nist_files, 1):
    df = read_csv_smart(fp)
    qcol = pick_qid_col(df) or "qid"
    pcol = pick_pid_col(df)
    lcol = pick_label_col(df)
    part = df[[qcol, pcol, lcol]].rename(columns={qcol:"qid", pcol:"pid", lcol:"NIST"})
    part["qid"]  = part["qid"].astype(str).str.strip()
    part["pid"]  = part["pid"].astype(str).str.strip()
    part["NIST"] = part["NIST"].apply(parse_label).fillna(0).astype(int)
    parts.append(part); seen += len(part)
    if i % 10 == 0 or i == len(nist_files):
        print(f"[NIST] Parsed {i}/{len(nist_files)}… rows so far={seen:,}")

nist = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["qid","pid"], keep="first")
print(f"[NIST] Total rows={len(nist):,} (after de-dup on qid,pid)")

# Build NIST pid→qid table (many-to-many via merge)
nist_pid_qids = nist[["pid","qid"]].drop_duplicates()
unique_nist_pids = nist_pid_qids["pid"].nunique()

# =========================
# 2) Build topics pair map: (pid, norm(passage_injected)) → qid
# =========================
topic_files = sorted(TOPICS_DIR.glob(TOPICS_GLOB))
if not topic_files:
    raise FileNotFoundError(f"No topic files matching {TOPICS_GLOB!r} in {TOPICS_DIR}")

pair_map = {}  # (pid, norm_passage_injected) -> qid
rows_seen = 0
for fp in topic_files:
    df = read_csv_smart(fp)
    pid_res   = pick_col(df, ["pid_resolved"]) or pick_col(df, ["pid_qrels"])
    inj_col   = pick_col(df, ["passage_injected","passage_eng"])
    qid_col   = pick_col(df, ["qid","topic_id"])
    if not pid_res or not inj_col or not qid_col:
        continue
    tmp = df[[pid_res, inj_col, qid_col]].rename(columns={pid_res:"pid", inj_col:"passage_inj", qid_col:"qid"})
    tmp["pid"] = tmp["pid"].astype(str).str.strip()
    tmp["key_pass"] = tmp["passage_inj"].map(norm_text)
    rows_seen += len(tmp)
    for pid, key_pass, qid in tmp[["pid","key_pass","qid"]].itertuples(index=False):
        if pid and key_pass and qid and (pid, key_pass) not in pair_map:
            pair_map[(pid, key_pass)] = str(qid).strip()
print(f"[TOPICS] files={len(topic_files)}; rows scanned={rows_seen:,}; unique pairs in map={len(pair_map):,}")

# =========================
# 3) Load LLM file, parse labels, map qid
# =========================
if not LLM_FILE.exists():
    raise FileNotFoundError(f"LLM file not found: {LLM_FILE}")
print(f"[LLM ] USING FILE: {LLM_FILE}")

llm_raw = read_csv_smart(LLM_FILE)
pcol    = pick_pid_col(llm_raw)
lcol    = pick_label_col(llm_raw)
p_eng   = pick_col(llm_raw, ["passage_eng","passage_injected","passage_en","passage"])
if not p_eng:
    raise KeyError("LLM file must contain a passage_eng/passsage_injected/passsage_en/passsage column")

# Keep the original columns for diagnostics, but create a normalized view to work with
llm_work = llm_raw[[pcol, lcol, p_eng]].rename(columns={pcol:"pid", lcol:"LLM_raw", p_eng:"passage_eng"})
llm_work["pid"] = llm_work["pid"].astype(str).str.strip()
llm_work["LLM_parsed"] = llm_work["LLM_raw"].apply(parse_label)

total_rows = len(llm_work)
parsed_ok  = llm_work["LLM_parsed"].notna().sum()
unparsable = total_rows - parsed_ok
print(f"[LLM ] rows={total_rows:,} | parsed={parsed_ok:,} | unparseable={unparsable:,}")

# --- NEW: write out unparseable label rows (before we drop/map) ---
# --- NEW: write out unparseable label rows (before we drop/map) ---
if unparsable > 0:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bad_mask = llm_work["LLM_parsed"].isna()
    # include the original columns for user inspection
    bad_rows = llm_raw.loc[bad_mask.index[bad_mask]].copy()

    # 2a) Write the single combined CSV (kept for convenience/backward-compat)
    bad_rows.to_csv(OUT_UNPARSEABLE, index=False, encoding="utf-8")
    print(f"[LLM ] wrote unparseable labels to: {OUT_UNPARSEABLE}")

    # 2b) Also write chunked copies into OUT_DIR / 'unparseable'  (500 rows per file)
    UNPARSEABLE_DIR = OUT_DIR / "unparseable"
    written_parts = _write_chunked_csv(
        bad_rows,
        out_dir=UNPARSEABLE_DIR,
        base_name="unparseable",
        chunk_size=500
    )
    if written_parts:
        print(f"[LLM ] also split unparseable rows into {len(written_parts)} file(s) under: {UNPARSEABLE_DIR}")


# Handle invalid labels
if MAP_INVALID_TO_ZERO:
    llm_work["LLM"] = llm_work["LLM_parsed"].fillna(0).astype(int)
else:
    llm_work = llm_work[llm_work["LLM_parsed"].notna()].copy()
    llm_work["LLM"] = llm_work["LLM_parsed"].astype(int)

# Map qid via pair match on (pid, passage_eng)
llm_work["key_pass"] = llm_work["passage_eng"].map(norm_text)
llm_work["qid"] = llm_work.apply(lambda r: pair_map.get((r["pid"], r["key_pass"]), ""), axis=1)
matched_pairs = (llm_work["qid"] != "").sum()
print(f"[LLM ] qid matched by (pid,passage_eng): {matched_pairs:,} / {len(llm_work):,}")

# Optional fallback: expand remaining rows by pid→all NIST qids
if ALLOW_PID_ONLY_FALLBACK:
    need = llm_work["qid"] == ""
    if need.any():
        fallback = llm_work.loc[need, ["pid","LLM"]].merge(nist_pid_qids, on="pid", how="inner")
        fallback = fallback.rename(columns={"qid":"qid_fb"})
        llm_work = llm_work.merge(fallback[["pid","qid_fb"]], on="pid", how="left")
        llm_work["qid"] = llm_work["qid"].where(llm_work["qid"] != "", llm_work["qid_fb"].fillna(""))
        llm_work.drop(columns=["qid_fb"], inplace=True)
        resolved_after_fb = (llm_work["qid"] != "").sum()
        print(f"[LLM ] after pid-only fallback, qid resolved: {resolved_after_fb:,}")

# --- NEW: write out rows that still have no qid after mapping (and fallback) ---
no_qid_mask = llm_work["qid"] == ""
unresolved = llm_work.loc[no_qid_mask].copy()
if len(unresolved):
    # Join back to the original raw rows for full visibility (best-effort on pid+passage)
    unresolved_keys = set(zip(unresolved["pid"], unresolved["key_pass"]))
    # Build a quick key on llm_raw
    raw_copy = llm_raw.copy()
    raw_copy["__pid__"] = raw_copy[pcol].astype(str).str.strip()
    raw_copy["__key__"] = raw_copy[p_eng].map(norm_text)
    to_write = raw_copy[(raw_copy["__pid__"], raw_copy["__key__"])\
                        .apply(lambda _: False)]  # just to create structure
    # faster approach: boolean mask via merge
    stub = pd.DataFrame(list(unresolved_keys), columns=["__pid__","__key__"])
    to_write = stub.merge(raw_copy, on=["__pid__","__key__"], how="left").drop(columns=["__pid__","__key__"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    to_write.to_csv(OUT_UNRESOLVED, index=False, encoding="utf-8")
    print(f"[LLM ] wrote rows with unresolved qid to: {OUT_UNRESOLVED}  (rows={len(unresolved):,})")

# Drop rows still lacking qid
before_drop = len(llm_work)
llm_work = llm_work[llm_work["qid"] != ""].copy()
dropped = before_drop - len(llm_work)
if dropped:
    print(f"[LLM ] dropped rows with no qid after mapping: {dropped:,}")

# De-dup AFTER mapping/expansion
before_dedup = len(llm_work)
llm_work = llm_work.drop_duplicates(subset=["qid","pid"], keep="first")
after_dedup = len(llm_work)
if after_dedup != before_dedup:
    print(f"[LLM ] de-duplicated (qid,pid): {before_dedup:,} -> {after_dedup:,}")

# =========================
# 4) Join & build confusion matrix
# =========================
paired = nist.merge(llm_work[["qid","pid","LLM"]], on=["qid","pid"], how="inner")
print(f"[JOIN] Pairs after join (qid,pid): {len(paired):,}")

cm = pd.crosstab(
    index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
    columns=pd.Categorical(paired["LLM"],  categories=LABELS, ordered=True),
    dropna=False
)
cm.index.name = "NIST"; cm.columns.name = "LLM"
cm_pct = cm.div(cm.sum(axis=1).replace(0,1), axis=0) * 100.0

# =========================
# 5) Save + plot
# =========================
OUT_DIR.mkdir(parents=True, exist_ok=True)
cm.to_csv(OUT_COUNTS)
cm_pct.round(2).to_csv(OUT_PCT)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
plt.title(f"Confusion Matrix: NIST vs LLM — {MODEL} {TREC_DL_YEAR} {LANG}")
plt.ylabel("NIST label"); plt.xlabel("LLM label")
plt.tight_layout(); plt.savefig(OUT_SVG, dpi=200); plt.show()

print(f"[DONE] Wrote counts to: {OUT_COUNTS}")
print(f"[DONE] Wrote row-% to:  {OUT_PCT}")
print(f"[DONE] Saved heatmap to: {OUT_SVG}")
if unparsable > 0:
    print(f"[DONE] Unparseable label rows saved to: {OUT_UNPARSEABLE}")
if len(unresolved):
    print(f"[DONE] Unresolved-qid rows saved to: {OUT_UNRESOLVED}")
