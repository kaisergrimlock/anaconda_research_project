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
LANG         = "eng"            # "eng", "vi", "raw"

# Inputs/outputs
NIST_DIR   = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
LLM_FILE   = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_raw.csv"
TOPICS_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / LANG
TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

OUT_DIR    = Path("outputs/baseline") / TREC_DL_YEAR / LANG
OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT    = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_SVG    = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"

# NEW: diagnostics — exact format requested
OUT_UNPARSEABLE = OUT_DIR / "llm_unparseable_labels_formatted.csv"
TOPICS_OUT_HEADERS = [
    "qid","query","pid_qrels","pid_resolved","passage","relevance","query_eng","passage_injected"
]

# Label handling
LABEL_COL_CHOICES   = ["relevance", "label"]
LABELS              = [0, 1, 2, 3]
MAP_INVALID_TO_ZERO = False

# Matching behavior
ALLOW_PID_ONLY_FALLBACK = True   # try pid→qid when (pid, passage) not found in topics
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
    return pd.read_csv(path, engine="python", dtype=str, on_bad_lines="skip")

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.strip().lower() in cols:
            return cols[name.strip().lower()]
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

# =========================
# 2) Build topics indices
# =========================
topic_files = sorted(TOPICS_DIR.glob(TOPICS_GLOB))
if not topic_files:
    raise FileNotFoundError(f"No topic files matching {TOPICS_GLOB!r} in {TOPICS_DIR}")

# (a) Pair map for qid lookup
pair_to_qid = {}  # (pid, norm_passage_injected) -> qid
# (b) Rich record map to emit your requested output format
pair_to_rec = {}  # (pid, norm_passage_injected) -> full dict in TOPICS_OUT_HEADERS
pid_to_rec  = {}  # pid -> first seen record (fallback)

rows_seen = 0
for fp in topic_files:
    df = read_csv_smart(fp)
    # Columns we want to pull
    qid_col   = pick_col(df, ["qid","topic_id"])
    query_col = pick_col(df, ["query"])
    pid_res   = pick_col(df, ["pid_resolved"])
    pid_q     = pick_col(df, ["pid_qrels"])
    inj_col   = pick_col(df, ["passage_injected","passage_eng"])
    pass_col  = pick_col(df, ["passage"])
    rel_col   = pick_col(df, ["relevance"])
    qeng_col  = pick_col(df, ["query_eng"])
    if not inj_col or not qid_col or not (pid_res or pid_q):
        continue

    tmp = df[[col for col in [qid_col, query_col, pid_q, pid_res, pass_col, rel_col, qeng_col, inj_col] if col]]\
            .copy()
    # normalize keys
    tmp["__pid__"] = tmp[(pid_res or pid_q)].astype(str).str.strip()
    tmp["__inj__"] = tmp[inj_col].map(norm_text)
    rows_seen += len(tmp)

    for _, row in tmp.iterrows():
        pid = row["__pid__"]; keyp = row["__inj__"]
        if not pid or not keyp: 
            continue
        if (pid, keyp) in pair_to_qid:
            continue
        # Build normalized record in requested order
        rec = {
            "qid":              str(row.get(qid_col, "")).strip(),
            "query":            str(row.get(query_col, "")).strip() if query_col else "",
            "pid_qrels":        str(row.get(pid_q, "")).strip() if pid_q else "",
            "pid_resolved":     str(row.get(pid_res, "")).strip() if pid_res else "",
            "passage":          str(row.get(pass_col, "")).strip() if pass_col else "",
            "relevance":        str(row.get(rel_col, "")).strip() if rel_col else "",
            "query_eng":        str(row.get(qeng_col, "")).strip() if qeng_col else "",
            "passage_injected": str(row.get(inj_col, "")).strip() if inj_col else "",
        }
        pair_to_qid[(pid, keyp)] = rec["qid"]
        pair_to_rec[(pid, keyp)] = rec
        pid_to_rec.setdefault(pid, rec)

print(f"[TOPICS] files={len(topic_files)}; rows scanned={rows_seen:,}; unique pairs indexed={len(pair_to_qid):,}")

# =========================
# 3) Load LLM & collect UNPARSEABLE in requested format
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

# minimal working view
llm = llm_raw[[pcol, lcol, p_eng]].rename(columns={pcol:"pid", lcol:"LLM_raw", p_eng:"passage_eng"})
llm["pid"] = llm["pid"].astype(str).str.strip()
llm["key_pass"] = llm["passage_eng"].map(norm_text)
llm["LLM_parsed"] = llm["LLM_raw"].apply(parse_label)

total_rows = len(llm)
unparsable_mask = llm["LLM_parsed"].isna()
print(f"[LLM ] rows={total_rows:,} | unparseable={int(unparsable_mask.sum()):,}")

# --- WRITE unparseable in required format ---
unparse_out = []
for pid, keyp, pe_text in llm.loc[unparsable_mask, ["pid","key_pass","passage_eng"]].itertuples(index=False):
    # prefer exact (pid, passage) match
    rec = pair_to_rec.get((pid, keyp))
    if not rec and ALLOW_PID_ONLY_FALLBACK:
        rec = pid_to_rec.get(pid)  # fallback: first seen topic row for that pid
    if rec:
        out = {k: rec.get(k, "") for k in TOPICS_OUT_HEADERS}
        # Use topics' own 'relevance' (if any); it’s already in rec
    else:
        # Last-resort: emit a skeleton row; keep LLM passage as 'passage_injected' so you can inspect
        out = {k: "" for k in TOPICS_OUT_HEADERS}
        out["passage_injected"] = pe_text or ""
    unparse_out.append(out)

OUT_DIR.mkdir(parents=True, exist_ok=True)
with OUT_UNPARSEABLE.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=TOPICS_OUT_HEADERS)
    w.writeheader()
    for r in unparse_out:
        w.writerow(r)
print(f"[LLM ] wrote UNPARSEABLE (requested format) to: {OUT_UNPARSEABLE}  rows={len(unparse_out):,}")

# =========================
# (Optional) The rest of your confusion-matrix pipeline…
# If you only needed the unparseable export, you can stop here.
# =========================
