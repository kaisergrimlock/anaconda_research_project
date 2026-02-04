# pairing.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd

from scripts.csv_helpers import (
    read_csv_smart, pick_col, pick_qid_col, pick_pid_col,
    pick_label_col_generic, norm_text, parse_label,
)

# NEW: import diagnostics helpers (pure, no I/O)
from .diagnostics import build_unresolved_rows, build_missing_and_extras

# ---- Public API --------------------------------------------------------------

@dataclass(frozen=True)
class PairingConfig:
    nist_dir: Path
    topics_dir: Path
    topics_glob: str
    llm_file: Path
    lang: str
    topic_pid_col: str              # e.g. "pid_resolved"
    topic_query_col: str            # e.g. "query_ru" or "query"
    topic_passage_col: str          # e.g. "passage_injected" or "passage"
    nist_label_choices: List[str]   # ["relevance","label","nist"]
    llm_label_choices: List[str]    # ["llm_relevance","label"]
    allow_pid_only_fallback: bool = True
    map_invalid_to_zero: bool = False

@dataclass
class PairingResult:
    nist: pd.DataFrame                # columns: qid,pid,NIST
    llm: pd.DataFrame                 # columns: qid,pid,LLM (deduped, qid-resolved)
    paired: pd.DataFrame              # inner join on (qid,pid): NIST, LLM
    unparseable_rows: pd.DataFrame    # raw LLM rows whose labels couldn't be parsed
    unresolved_qid_rows: pd.DataFrame # language-shaped unresolved (may be empty)
    nist_missing_df: pd.DataFrame     # NIST rows with no LLM (may be empty)
    llm_extra_df: pd.DataFrame        # LLM rows with no NIST (may be empty)

# ---- Entry point -------------------------------------------------------------

def pair_labels(cfg: PairingConfig) -> PairingResult:
    """Top-level single entrypoint. No file I/O. Prints progress to screen."""

    # 1) Load NIST
    nist, nist_pid_qids, nist_meta = _load_nist(cfg)
    print(f"[NIST] files={nist_meta['files']}; rows_seen={nist_meta['rows_seen']:,}; unique(qid,pid)={len(nist):,}")

    # 2) Build pair map from topics
    pair_map, topic_meta = _build_pair_map(cfg)
    print(f"[TOPICS] files={topic_meta['files']}; rows_scanned={topic_meta['rows_scanned']:,}; pair_map_size={len(pair_map):,}")

    # 3) Load LLM & parse labels
    llm_work, unparseable_rows, pcol_original, ptext_original, llm_raw_full, llm_meta = _load_llm(cfg)
    print(f"[LLM ] rows_raw={llm_meta['rows_raw']:,} | parsed={llm_meta['labels_parsed']:,} | unparseable={llm_meta['unparseable']:,}")

    # 4) Resolve QIDs (don't drop yet so we can report unresolved)
    llm_work, qid_meta = _resolve_qids(cfg, llm_work, nist_pid_qids, pair_map)
    print(f"[QID ] in_llm={qid_meta['had_in_llm']:,} | by_pair_map={qid_meta['by_pair_map']:,} | by_pid_fallback={qid_meta['by_pid_fallback']:,}")

    # (count unresolved before drop)
    unresolved_pre_drop = int((llm_work["qid"] == "").sum())
    if unresolved_pre_drop:
        print(f"[QID ] unresolved before drop: {unresolved_pre_drop:,}")

    # 5) Build unresolved report (before drop)
    topics_full = _load_topics_full(cfg)
    unresolved_qid_rows = build_unresolved_rows(
        lang=cfg.lang,
        topic_passage_col=cfg.topic_passage_col,
        llm_work=llm_work,
        llm_raw=llm_raw_full,
        pcol_original=pcol_original,
        ptext_col_original=ptext_original,
        topics_full=topics_full,
    )
    if not unresolved_qid_rows.empty:
        print(f"[DIAG] unresolved-qid rows (report): {len(unresolved_qid_rows):,}")

    # 6) Drop empty qid & dedup
    before_drop = len(llm_work)
    llm_work = llm_work[llm_work["qid"] != ""].copy()
    after_drop = len(llm_work)
    dropped_empty_qid = before_drop - after_drop
    if dropped_empty_qid:
        print(f"[LLM ] dropped rows with empty qid: {dropped_empty_qid:,}")

    before_dedup = len(llm_work)
    llm_work = llm_work.drop_duplicates(subset=["qid", "pid"], keep="first")
    after_dedup = len(llm_work)
    dropped_dupes = before_dedup - after_dedup
    if dropped_dupes:
        print(f"[LLM ] de-duplicated (qid,pid): {before_dedup:,} -> {after_dedup:,}")

    # 7) Missing/extra diagnostics (after cleanup)
    nist_missing_df, llm_extra_df = build_missing_and_extras(
        lang=cfg.lang, nist=nist, llm=llm_work, topics_full=topics_full
    )
    if not nist_missing_df.empty:
        print(f"[DIAG] NIST rows with NO matching LLM label: {len(nist_missing_df):,}")
    else:
        print("[DIAG] All NIST rows had a matching LLM label.")
    if not llm_extra_df.empty:
        print(f"[DIAG] LLM rows with NO matching NIST judgment: {len(llm_extra_df):,}")
    else:
        print("[DIAG] All LLM rows had a matching NIST judgment.")

    # 8) Join for evaluation
    paired = nist.merge(llm_work[["qid", "pid", "LLM"]], on=["qid", "pid"], how="inner")
    print(f"[JOIN] Pairs after join (qid,pid): {len(paired):,}")

    # Summary line (handy)
    print(
        "[SUMMARY] paired="
        f"{len(paired):,} | nist_only={len(nist_missing_df):,} | llm_only={len(llm_extra_df):,} | "
        f"unparseable={llm_meta['unparseable']:,} | unresolved_pre_drop={unresolved_pre_drop:,}"
    )

    return PairingResult(
        nist=nist,
        llm=llm_work,
        paired=paired,
        unparseable_rows=unparseable_rows,
        unresolved_qid_rows=unresolved_qid_rows,
        nist_missing_df=nist_missing_df,
        llm_extra_df=llm_extra_df,
    )

# ---- Internal helpers (logic only) ------------------------------------------

def _pick_nist_label_col(df: pd.DataFrame, choices: List[str]) -> str:
    return pick_label_col_generic(df, choices, "NIST")

def _pick_llm_label_col(df: pd.DataFrame, choices: List[str]) -> str:
    return pick_label_col_generic(df, choices, "LLM")

def _load_nist(cfg: PairingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    files = sorted(cfg.nist_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {cfg.nist_dir}")
    parts: List[pd.DataFrame] = []
    rows_seen = 0
    for fp in files:
        df = read_csv_smart(fp)
        qcol = pick_qid_col(df) or "qid"
        pcol = pick_pid_col(df)
        lcol = _pick_nist_label_col(df, cfg.nist_label_choices)

        # Base keep
        keep_cols = [qcol, pcol, lcol]

        # OPTIONAL: if NIST CSV already includes query/passage, keep them
        if "query" in df.columns:
            keep_cols.append("query")
        if "passage" in df.columns:
            keep_cols.append("passage")

        part = df[keep_cols].rename(columns={qcol: "qid", pcol: "pid", lcol: "NIST"})
        part["qid"] = part["qid"].astype(str).str.strip()
        part["pid"] = part["pid"].astype(str).str.strip()
        part["NIST"] = part["NIST"].apply(parse_label).fillna(0).astype(int)

        # Normalize types for optional text
        if "query" in part.columns:
            part["query"] = part["query"].astype(str)
        if "passage" in part.columns:
            part["passage"] = part["passage"].astype(str)

        rows_seen += len(part)
        parts.append(part)

    # Note: we keep potential 'query'/'passage' in this merged frame.
    nist = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["qid", "pid"], keep="first")
    nist_pid_qids = nist[["pid", "qid"]].drop_duplicates()
    meta = {"files": len(files), "rows_seen": rows_seen}
    return nist, nist_pid_qids, meta



def _build_pair_map(cfg: PairingConfig) -> Tuple[Dict[Tuple[str, str], str], Dict[str, int]]:
    topic_files = sorted(cfg.topics_dir.glob(cfg.topics_glob))
    if not topic_files:
        return {}, {"files": 0, "rows_scanned": 0}
    pair_map: Dict[Tuple[str, str], str] = {}
    rows_scanned = 0
    for fp in topic_files:
        df = read_csv_smart(fp)
        needed = {cfg.topic_pid_col, cfg.topic_passage_col, "qid"}
        if not needed.issubset(df.columns):
            continue
        tmp = df[[cfg.topic_pid_col, cfg.topic_passage_col, "qid"]].rename(
            columns={cfg.topic_pid_col: "pid", cfg.topic_passage_col: "passage_inj"}
        )
        tmp["pid"] = tmp["pid"].astype(str).str.strip()
        tmp["key_pass"] = tmp["passage_inj"].map(norm_text)
        rows_scanned += len(tmp)
        for pid, key_pass, qid in tmp[["pid", "key_pass", "qid"]].itertuples(index=False):
            if pid and key_pass and qid and (pid, key_pass) not in pair_map:
                pair_map[(pid, key_pass)] = str(qid).strip()
    meta = {"files": len(topic_files), "rows_scanned": rows_scanned}
    return pair_map, meta

def _load_llm(cfg: PairingConfig):
    if not cfg.llm_file.exists():
        raise FileNotFoundError(f"LLM file not found: {cfg.llm_file}")
    raw = read_csv_smart(cfg.llm_file)
    pcol = pick_pid_col(raw)
    lcol = _pick_llm_label_col(raw, cfg.llm_label_choices)
    ptext_col = pick_col(raw, ["passage_eng", "passage_injected", "passage_en", "passage"])
    qid_col = pick_qid_col(raw)

    keep = [pcol, lcol, ptext_col] + ([qid_col] if qid_col else [])
    ren = {pcol: "pid", lcol: "LLM_raw", ptext_col: "passage_eng"}
    if qid_col:
        ren[qid_col] = "qid"

    work = raw[keep].rename(columns=ren)
    work["pid"] = work["pid"].astype(str).str.strip()
    work["LLM_parsed"] = work["LLM_raw"].apply(parse_label)

    rows_raw = len(work)
    labels_parsed = int(work["LLM_parsed"].notna().sum())
    unparseable_count = rows_raw - labels_parsed

    # capture unparseable rows (pure; caller decides to write)
    bad_mask = work["LLM_parsed"].isna()
    unparseable = raw.loc[bad_mask.index[bad_mask]].copy()

    # normalize label column
    if cfg.map_invalid_to_zero:
        work["LLM"] = work["LLM_parsed"].fillna(0).astype(int)
    else:
        work = work[work["LLM_parsed"].notna()].copy()
        work["LLM"] = work["LLM_parsed"].astype(int)

    meta = {
        "rows_raw": int(rows_raw),
        "labels_parsed": int(labels_parsed),
        "unparseable": int(unparseable_count),
    }
    return work, unparseable, pcol, ptext_col, raw, meta

def _resolve_qids(
    cfg: PairingConfig,
    llm_work: pd.DataFrame,
    nist_pid_qids: pd.DataFrame,
    pair_map: Dict[Tuple[str, str], str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    had_in_llm = 0
    by_pair_map = 0
    by_pid_fallback = 0

    if "qid" in llm_work.columns:
        had_in_llm = int((llm_work["qid"].astype(str).str.strip() != "").sum())
        llm_work["qid"] = llm_work["qid"].astype(str).str.strip()
        # don't drop here; caller will compute unresolved & drop
    else:
        # map via (pid,passage)
        llm_work["key_pass"] = llm_work["passage_eng"].map(norm_text)
        mapped = llm_work.apply(lambda r: pair_map.get((r["pid"], r["key_pass"]), ""), axis=1)
        by_pair_map = int((mapped != "").sum())
        llm_work["qid"] = mapped

        if cfg.allow_pid_only_fallback:
            need = llm_work["qid"] == ""
            if need.any():
                fb = llm_work.loc[need, ["pid"]].merge(nist_pid_qids, on="pid", how="inner")
                fb = fb.rename(columns={"qid": "qid_fb"})
                llm_work = llm_work.merge(fb[["pid", "qid_fb"]], on="pid", how="left")
                filled = (llm_work["qid"] == "") & llm_work["qid_fb"].notna() & (llm_work["qid_fb"] != "")
                by_pid_fallback = int(filled.sum())
                llm_work["qid"] = llm_work["qid"].where(llm_work["qid"] != "", llm_work["qid_fb"].fillna(""))
                llm_work.drop(columns=["qid_fb"], inplace=True)

    meta = {
        "had_in_llm": had_in_llm,
        "by_pair_map": by_pair_map,
        "by_pid_fallback": by_pid_fallback,
    }
    return llm_work, meta

def _load_topics_full(cfg: PairingConfig) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for fp in sorted(cfg.topics_dir.glob(cfg.topics_glob)):
        df_t = read_csv_smart(fp)
        needed = {"qid", cfg.topic_pid_col, cfg.topic_query_col, cfg.topic_passage_col}
        if not needed.issubset(df_t.columns):
            if not {"qid", cfg.topic_pid_col}.issubset(df_t.columns):
                continue
        keep = pd.DataFrame({
            "qid": df_t["qid"].astype(str).str.strip(),
            "pid_resolved": df_t[cfg.topic_pid_col].astype(str).str.strip(),
        })
        if cfg.topic_query_col in df_t.columns:
            keep[cfg.topic_query_col] = df_t[cfg.topic_query_col]
        if cfg.topic_passage_col in df_t.columns:
            keep[cfg.topic_passage_col] = df_t[cfg.topic_passage_col]
        parts.append(keep)
    if not parts:
        cols = ["qid", "pid_resolved", "pid_qrels"]
        if cfg.lang == "raw":
            cols += ["query", "passage"]
        else:
            cols += [f"query_{cfg.lang}", "passage_injected"]
        return pd.DataFrame(columns=cols)
    topics = pd.concat(parts, ignore_index=True)
    topics["pid_qrels"] = topics.get("pid_resolved", "").astype(str)
    return topics.drop_duplicates(subset=["qid", "pid_resolved"], keep="first")
