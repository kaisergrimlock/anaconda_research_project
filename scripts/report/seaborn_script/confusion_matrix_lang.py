#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import csv
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# ======  Config  =========
# =========================
TREC_DL_YEAR = "2023"
MODEL        = "gpt-oss-20b"
LANG         = "raw"          # "eng", "vi", "raw"

# Inputs / outputs
NIST_DIR  = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
if LANG == "raw":
    TOPICS_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
    LLM_FILE   = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_raw.csv"
else:
    TOPICS_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / LANG
    LLM_FILE   = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_raw.csv"

TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

OUT_DIR    = Path("outputs/baseline") / TREC_DL_YEAR / LANG
OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT    = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_SVG    = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"

# Diagnostics
OUT_UNPARSEABLE = OUT_DIR / "llm_unparseable_labels.csv"
OUT_UNRESOLVED  = OUT_DIR / "llm_unresolved_qid.csv"

# Label handling
NIST_LABEL_COL_CHOICES = ["relevance", "label", "nist"]  # be liberal for NIST/judged files
LLM_LABEL_COL_CHOICES  = ["llm_relevance", "label"]      # labels produced by LLM runs
LABELS                 = [0, 1, 2, 3]
MAP_INVALID_TO_ZERO    = False

# Matching behavior
ALLOW_PID_ONLY_FALLBACK = True  # try pid→qid when (pid, passage) pair not found

# =========================
# ======  Helpers  ========
# =========================

def _bump_field_limit():
    """Allow huge CSV cells to avoid _csv.Error: field larger than field limit."""
    limit = getattr(sys, "maxsize", 2_000_000_000)
    while limit >= 131_072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2
    csv.field_size_limit(1_000_000)

_bump_field_limit()

def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Read CSV robustly: use python engine and skip physically malformed lines.
    Use utf-8-sig to swallow BOM if present (e.g., ï»¿qid).
    """
    return pd.read_csv(path, engine="python", dtype=str, on_bad_lines="skip", encoding="utf-8-sig")

def _write_chunked_csv(df: pd.DataFrame, out_dir: Path, base_name: str, chunk_size: int = 500) -> List[Path]:
    """
    Write df into multiple CSV files with at most `chunk_size` rows each.
    Filenames: {base_name}_part_0001.csv, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    if n == 0:
        return []
    paths: List[Path] = []
    num_parts = (n + chunk_size - 1) // chunk_size
    pad = max(4, len(str(num_parts)))
    for i in range(num_parts):
        start, end = i * chunk_size, min((i + 1) * chunk_size, n)
        part = df.iloc[start:end]
        fp = out_dir / f"{base_name}_part_{(i + 1):0{pad}d}.csv"
        part.to_csv(fp, index=False, encoding="utf-8")
        paths.append(fp)
    return paths

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.strip().lower(): c for c in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in cols:
            return cols[key]
    return None

def pick_qid_col(df: pd.DataFrame) -> Optional[str]:
    return pick_col(df, ["qid", "topic", "topic_id"])

def pick_pid_col(df: pd.DataFrame) -> str:
    c = pick_col(df, ["pid", "pid_resolved", "pid_qrels", "docid", "doc_id"])
    if not c:
        raise KeyError(f"No pid-like column in {list(df.columns)}")
    return c

def pick_label_col_generic(df: pd.DataFrame, candidates: List[str], who: str) -> str:
    c = pick_col(df, candidates)
    if not c:
        raise KeyError(f"{who}: none of {candidates} found in columns {list(df.columns)}")
    return c

def pick_nist_label_col(df: pd.DataFrame) -> str:
    return pick_label_col_generic(df, NIST_LABEL_COL_CHOICES, "NIST")

def pick_llm_label_col(df: pd.DataFrame) -> str:
    return pick_label_col_generic(df, LLM_LABEL_COL_CHOICES, "LLM")

def norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

_digit_0_3 = re.compile(r"\b([0-3])\b")
def parse_label(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s in {"0", "1", "2", "3"}:
        return int(s)
    m = _digit_0_3.search(s)
    return int(m.group(1)) if m else None

# =========================
# ======  Pipeline  =======
# =========================

def load_nist() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nist_files = sorted(NIST_DIR.rglob("*.csv"))
    if not nist_files:
        raise FileNotFoundError(f"No CSV files under {NIST_DIR}")

    print(f"[NIST] Found {len(nist_files)} files under {NIST_DIR}")
    parts, seen = [], 0
    for i, fp in enumerate(nist_files, 1):
        df = read_csv_smart(fp)
        try:
            qcol = pick_qid_col(df) or "qid"
            pcol = pick_pid_col(df)
            lcol = pick_nist_label_col(df)
        except KeyError as e:
            print(f"[NIST] Label column not found in: {fp}  columns={list(df.columns)}")
            raise

        part = df[[qcol, pcol, lcol]].rename(columns={qcol: "qid", pcol: "pid", lcol: "NIST"})
        part["qid"]  = part["qid"].astype(str).str.strip()
        part["pid"]  = part["pid"].astype(str).str.strip()
        part["NIST"] = part["NIST"].apply(parse_label).fillna(0).astype(int)
        parts.append(part); seen += len(part)
        if i % 10 == 0 or i == len(nist_files):
            print(f"[NIST] Parsed {i}/{len(nist_files)}… rows so far={seen:,}")

    nist = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["qid", "pid"], keep="first")
    print(f"[NIST] Total rows={len(nist):,} (after de-dup on qid,pid)")

    # pid→qid (for fallback expansion)
    nist_pid_qids = nist[["pid", "qid"]].drop_duplicates()
    return nist, nist_pid_qids

def build_pair_map() -> Dict[Tuple[str, str], str]:
    topic_files = sorted(TOPICS_DIR.glob(TOPICS_GLOB))
    if not topic_files:
        raise FileNotFoundError(f"No topic files matching {TOPICS_GLOB!r} in {TOPICS_DIR}")

    pair_map: Dict[Tuple[str, str], str] = {}
    rows_seen = 0
    for fp in topic_files:
        df = read_csv_smart(fp)
        pid_res = pick_col(df, ["pid_resolved"]) or pick_col(df, ["pid_qrels"])
        inj_col = pick_col(df, ["passage_injected", "passage_eng"])
        qid_col = pick_col(df, ["qid", "topic_id"])
        if not pid_res or not inj_col or not qid_col:
            continue
        tmp = df[[pid_res, inj_col, qid_col]].rename(columns={pid_res: "pid", inj_col: "passage_inj", qid_col: "qid"})
        tmp["pid"] = tmp["pid"].astype(str).str.strip()
        tmp["key_pass"] = tmp["passage_inj"].map(norm_text)
        rows_seen += len(tmp)
        for pid, key_pass, qid in tmp[["pid", "key_pass", "qid"]].itertuples(index=False):
            if pid and key_pass and qid and (pid, key_pass) not in pair_map:
                pair_map[(pid, key_pass)] = str(qid).strip()
    print(f"[TOPICS] files={len(topic_files)}; rows scanned={rows_seen:,}; unique pairs in map={len(pair_map):,}")
    return pair_map

def load_llm(nist_pid_qids: pd.DataFrame, pair_map: Dict[Tuple[str, str], str]) -> Tuple[pd.DataFrame, int, int, bool, pd.DataFrame]:
    """Return (llm_work, unparsable_count, total_rows, QID_FROM_LLM, llm_raw_full)"""
    if not LLM_FILE.exists():
        raise FileNotFoundError(f"LLM file not found: {LLM_FILE}")
    print(f"[LLM ] USING FILE: {LLM_FILE}")

    llm_raw = read_csv_smart(LLM_FILE)
    pcol      = pick_pid_col(llm_raw)
    lcol      = pick_llm_label_col(llm_raw)
    p_eng     = pick_col(llm_raw, ["passage_eng", "passage_injected", "passage_en", "passage"])
    llm_qid_c = pick_qid_col(llm_raw)   # detect qid in LLM file, if any
    if not p_eng:
        raise KeyError("LLM file must contain a passage_eng/passage_injected/passage_en/passage column")

    keep_cols = [pcol, lcol, p_eng] + ([llm_qid_c] if llm_qid_c else [])
    ren_cols  = {pcol: "pid", lcol: "LLM_raw", p_eng: "passage_eng"}
    if llm_qid_c:
        ren_cols[llm_qid_c] = "qid"

    llm_work = llm_raw[keep_cols].rename(columns=ren_cols)
    llm_work["pid"] = llm_work["pid"].astype(str).str.strip()
    llm_work["LLM_parsed"] = llm_work["LLM_raw"].apply(parse_label)

    total_rows = len(llm_work)
    parsed_ok  = llm_work["LLM_parsed"].notna().sum()
    unparsable = total_rows - parsed_ok
    print(f"[LLM ] rows={total_rows:,} | parsed={parsed_ok:,} | unparseable={unparsable:,}")

    # diagnostics: write unparseable rows (combined + chunked)
    if unparsable > 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        bad_mask = llm_work["LLM_parsed"].isna()
        bad_rows = llm_raw.loc[bad_mask.index[bad_mask]].copy()
        bad_rows.to_csv(OUT_UNPARSEABLE, index=False, encoding="utf-8")
        print(f"[LLM ] wrote unparseable labels to: {OUT_UNPARSEABLE}")

        UNPARSEABLE_DIR = OUT_DIR / "unparseable"
        written_parts = _write_chunked_csv(
            bad_rows, out_dir=UNPARSEABLE_DIR, base_name="unparseable", chunk_size=500
        )
        if written_parts:
            print(f"[LLM ] also split unparseable rows into {len(written_parts)} file(s) under: {UNPARSEABLE_DIR}")

    # normalize LLM label
    if MAP_INVALID_TO_ZERO:
        llm_work["LLM"] = llm_work["LLM_parsed"].fillna(0).astype(int)
    else:
        llm_work = llm_work[llm_work["LLM_parsed"].notna()].copy()
        llm_work["LLM"] = llm_work["LLM_parsed"].astype(int)

    # Determine path: use qid from LLM if available; otherwise map
    QID_FROM_LLM = "qid" in llm_work.columns
    if QID_FROM_LLM:
        llm_work["qid"] = llm_work["qid"].astype(str).str.strip()
        before = len(llm_work)
        llm_work = llm_work[llm_work["qid"] != ""].copy()
        removed = before - len(llm_work)
        if removed:
            print(f"[LLM ] using qid from LLM file; removed rows with empty qid: {removed:,}")
        print(f"[LLM ] qid source: LLM file column 'qid' (rows with qid={len(llm_work):,})")
    else:
        # Map via (pid, normalized passage) → qid
        llm_work["key_pass"] = llm_work["passage_eng"].map(norm_text)
        llm_work["qid"] = llm_work.apply(lambda r: pair_map.get((r["pid"], r["key_pass"]), ""), axis=1)
        matched_pairs = (llm_work["qid"] != "").sum()
        print(f"[LLM ] qid matched by (pid,passage_eng): {matched_pairs:,} / {len(llm_work):,}")

        # Optional pid-only fallback expansion
        if ALLOW_PID_ONLY_FALLBACK:
            need = llm_work["qid"] == ""
            if need.any():
                fallback = llm_work.loc[need, ["pid", "LLM"]].merge(nist_pid_qids, on="pid", how="inner")
                fallback = fallback.rename(columns={"qid": "qid_fb"})
                llm_work = llm_work.merge(fallback[["pid", "qid_fb"]], on="pid", how="left")
                llm_work["qid"] = llm_work["qid"].where(llm_work["qid"] != "", llm_work["qid_fb"].fillna(""))
                llm_work.drop(columns=["qid_fb"], inplace=True)
                resolved_after_fb = (llm_work["qid"] != "").sum()
                print(f"[LLM ] after pid-only fallback, qid resolved: {resolved_after_fb:,}")

    return llm_work, unparsable, total_rows, QID_FROM_LLM, llm_raw

def write_unresolved_if_needed(llm_work: pd.DataFrame, llm_raw: pd.DataFrame, pcol_original: str, ptext_col_original: str) -> pd.DataFrame:
    """
    For the mapping path (when qid is NOT from LLM), write rows that still have no qid.
    Returns the 'unresolved' dataframe (possibly empty).
    """
    if "key_pass" not in llm_work.columns:
        return pd.DataFrame()

    no_qid_mask = llm_work["qid"] == ""
    unresolved = llm_work.loc[no_qid_mask, ["pid", "key_pass"]].copy()
    if len(unresolved) == 0:
        return unresolved

    # Join back to the original LLM rows for full visibility (best-effort pid+passage match)
    raw_copy = llm_raw.copy()
    raw_copy["__pid__"] = raw_copy[pcol_original].astype(str).str.strip()
    raw_copy["__key__"] = raw_copy[ptext_col_original].map(norm_text)

    stub = unresolved.rename(columns={"pid": "__pid__", "key_pass": "__key__"})
    to_write = stub.merge(raw_copy, on=["__pid__", "__key__"], how="left").drop(columns=["__pid__", "__key__"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    to_write.to_csv(OUT_UNRESOLVED, index=False, encoding="utf-8")
    print(f"[LLM ] wrote rows with unresolved qid to: {OUT_UNRESOLVED}  (rows={len(unresolved):,})")
    return unresolved

def main():
    # 1) Load NIST data and pid→qid table
    nist, nist_pid_qids = load_nist()

    # 2) Build pair map (only used if later we don't have qid in LLM)
    try:
        pair_map = build_pair_map()
    except FileNotFoundError as e:
        # If topics are missing but the LLM file has qid, we can still proceed.
        print(f"[TOPICS] Warning: {e}")
        pair_map = {}

    # 3) Load LLM judgments; possibly map qid
    #    We also need the original pid/passage column names for unresolved writer if used.
    llm_raw_probe = read_csv_smart(LLM_FILE)
    pcol_original  = pick_pid_col(llm_raw_probe)
    ptext_original = pick_col(llm_raw_probe, ["passage_eng", "passage_injected", "passage_en", "passage"])
    llm_work, unparsable, total_rows, QID_FROM_LLM, llm_raw_full = load_llm(nist_pid_qids, pair_map)

    # 4) (Optional) write unresolved only if we followed the mapping path
    unresolved = pd.DataFrame()
    if not QID_FROM_LLM and ptext_original:
        unresolved = write_unresolved_if_needed(llm_work, llm_raw_full, pcol_original, ptext_original)

    # 5) Drop rows still lacking qid
    before_drop = len(llm_work)
    llm_work = llm_work[llm_work["qid"] != ""].copy()
    dropped = before_drop - len(llm_work)
    if dropped:
        print(f"[LLM ] dropped rows with no qid after mapping: {dropped:,}")

    # 6) De-dup AFTER mapping/expansion
    before_dedup = len(llm_work)
    llm_work = llm_work.drop_duplicates(subset=["qid", "pid"], keep="first")
    after_dedup = len(llm_work)
    if after_dedup != before_dedup:
        print(f"[LLM ] de-duplicated (qid,pid): {before_dedup:,} -> {after_dedup:,}")

    # 7) Join & build confusion matrix
    paired = nist.merge(llm_work[["qid", "pid", "LLM"]], on=["qid", "pid"], how="inner")
    print(f"[JOIN] Pairs after join (qid,pid): {len(paired):,}")

    cm = pd.crosstab(
        index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM"],  categories=LABELS, ordered=True),
        dropna=False
    )
    cm.index.name = "NIST"; cm.columns.name = "LLM"
    cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0

    # 8) Save + plot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cm.to_csv(OUT_COUNTS)
    cm_pct.round(2).to_csv(OUT_PCT)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title(f"Confusion Matrix: NIST vs LLM — {MODEL} {TREC_DL_YEAR} {LANG}")
    plt.ylabel("NIST label"); plt.xlabel("LLM label")
    plt.tight_layout(); plt.savefig(OUT_SVG, dpi=200); plt.show()

    print(f"[DONE] Wrote counts to: {OUT_COUNTS}")
    print(f"[DONE] Wrote row-% to:  {OUT_PCT}")
    print(f"[DONE] Saved heatmap to: {OUT_SVG}")
    if unparsable > 0:
        print(f"[DONE] Unparseable label rows saved to: {OUT_UNPARSEABLE}")
    if not QID_FROM_LLM and len(unresolved):
        print(f"[DONE] Unresolved-qid rows saved to: {OUT_UNRESOLVED}")

if __name__ == "__main__":
    main()
