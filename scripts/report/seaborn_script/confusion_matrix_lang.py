#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

# calculate metrics
from metrics_llm import (
    compute_mae,
    compute_weighted_kappa_ordinal,
    compute_unweighted_kappa,
    binarize_labels,
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 👇 NEW: import helpers
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]  # seaborn_script -> report -> scripts -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import (
    bump_field_limit,
    read_csv_smart,
    write_chunked_csv,
    pick_col,
    pick_qid_col,
    pick_pid_col,
    pick_label_col_generic,
    norm_text,
    parse_label,
)

# =========================
# ======  Config  =========
# =========================
TREC_DL_YEAR = "2023"
MODEL        = "gpt-oss-20b"
LANG         = "ru"          # "eng", "vi", "fr", "raw"

# ---------- language-driven topic schema ----------
if LANG == "raw":
    TOPIC_QUERY_COL   = "query"
    TOPIC_PASSAGE_COL = "passage"
else:
    TOPIC_QUERY_COL   = f"query_{LANG}"  # e.g. query_eng, query_vi, query_fr
    TOPIC_PASSAGE_COL = "passage_injected"

TOPIC_PID_COL     = "pid_resolved"   # raw judged files in your repo keep this name

# Inputs / outputs
NIST_DIR  = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
if LANG == "raw":
    TOPICS_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
    LLM_FILE   = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_raw.csv"
else:
    TOPICS_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / LANG
    LLM_FILE   = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}.csv"

TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

OUT_DIR    = Path("outputs/baseline") / TREC_DL_YEAR / LANG
OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT    = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_SVG    = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"

# Diagnostics
OUT_UNPARSEABLE   = OUT_DIR / "llm_unparseable_labels.csv"
OUT_UNRESOLVED    = OUT_DIR / "llm_unresolved_qid.csv"
OUT_NIST_MISSING  = OUT_DIR / "nist_not_joined_by_llm.csv"
OUT_LLM_EXTRA     = OUT_DIR / "llm_not_in_nist.csv"

# Label handling
NIST_LABEL_COL_CHOICES = ["relevance", "label", "nist"]
LLM_LABEL_COL_CHOICES  = ["llm_relevance", "label"]
LABELS                 = [0, 1, 2, 3]
MAP_INVALID_TO_ZERO    = False

# Matching behavior
ALLOW_PID_ONLY_FALLBACK = True  # try pid→qid when (pid, passage) pair not found

# ============ init CSV limit ============
bump_field_limit()

# =========================
# ======  Pipeline  =======
# =========================

def pick_nist_label_col(df: pd.DataFrame) -> str:
    return pick_label_col_generic(df, NIST_LABEL_COL_CHOICES, "NIST")


def pick_llm_label_col(df: pd.DataFrame) -> str:
    return pick_label_col_generic(df, LLM_LABEL_COL_CHOICES, "LLM")


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
        except KeyError:
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

    # pid→qid (for fallback)
    nist_pid_qids = nist[["pid", "qid"]].drop_duplicates()
    return nist, nist_pid_qids


def build_pair_map() -> Dict[Tuple[str, str], str]:
    """
    Create (pid, normalized_passage) -> qid map from the topic files.
    Now purely language-driven: we know the expected column names from LANG.
    """
    topic_files = sorted(TOPICS_DIR.glob(TOPICS_GLOB))
    if not topic_files:
        raise FileNotFoundError(f"No topic files matching {TOPICS_GLOB!r} in {TOPICS_DIR}")

    pair_map: Dict[Tuple[str, str], str] = {}
    rows_seen = 0

    for fp in topic_files:
        df = read_csv_smart(fp)

        # if the file does not have the language-specific columns, we just skip it
        if not {TOPIC_PID_COL, TOPIC_PASSAGE_COL, "qid"}.issubset(df.columns):
            continue

        tmp = df[[TOPIC_PID_COL, TOPIC_PASSAGE_COL, "qid"]].rename(
            columns={
                TOPIC_PID_COL: "pid",
                TOPIC_PASSAGE_COL: "passage_inj",
                "qid": "qid",
            }
        )

        tmp["pid"] = tmp["pid"].astype(str).str.strip()
        tmp["key_pass"] = tmp["passage_inj"].map(norm_text)
        rows_seen += len(tmp)

        for pid, key_pass, qid in tmp[["pid", "key_pass", "qid"]].itertuples(index=False):
            if pid and key_pass and qid and (pid, key_pass) not in pair_map:
                pair_map[(pid, key_pass)] = str(qid).strip()

    print(f"[TOPICS] files={len(topic_files)}; rows scanned={rows_seen:,}; unique pairs in map={len(pair_map):,}")
    return pair_map


def load_llm(
    nist_pid_qids: pd.DataFrame,
    pair_map: Dict[Tuple[str, str], str],
) -> Tuple[pd.DataFrame, int, int, bool, pd.DataFrame]:
    """Return (llm_work, unparsable_count, total_rows, QID_FROM_LLM, llm_raw_full)."""
    if not LLM_FILE.exists():
        raise FileNotFoundError(f"LLM file not found: {LLM_FILE}")
    print(f"[LLM ] USING FILE: {LLM_FILE}")

    llm_raw = read_csv_smart(LLM_FILE)
    pcol      = pick_pid_col(llm_raw)
    lcol      = pick_llm_label_col(llm_raw)
    # LLM files are still messy → keep flexible here
    p_eng     = pick_col(llm_raw, ["passage_eng", "passage_injected", "passage_en", "passage"])
    llm_qid_c = pick_qid_col(llm_raw)
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

    # write unparseable like before
    if unparsable > 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        bad_mask = llm_work["LLM_parsed"].isna()
        bad_rows = llm_raw.loc[bad_mask.index[bad_mask]].copy()
        bad_rows.to_csv(OUT_UNPARSEABLE, index=False, encoding="utf-8")
        print(f"[LLM ] wrote unparseable labels to: {OUT_UNPARSEABLE}")
        write_chunked_csv(bad_rows, OUT_DIR / "unparseable", "unparseable", 500)

    # normalize LLM label
    if MAP_INVALID_TO_ZERO:
        llm_work["LLM"] = llm_work["LLM_parsed"].fillna(0).astype(int)
    else:
        llm_work = llm_work[llm_work["LLM_parsed"].notna()].copy()
        llm_work["LLM"] = llm_work["LLM_parsed"].astype(int)

    # qid path
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
        # map via (pid, passage)
        llm_work["key_pass"] = llm_work["passage_eng"].map(norm_text)
        llm_work["qid"] = llm_work.apply(
            lambda r: pair_map.get((r["pid"], r["key_pass"]), ""),
            axis=1,
        )
        matched_pairs = (llm_work["qid"] != "").sum()
        print(f"[LLM ] qid matched by (pid,passage_eng): {matched_pairs:,} / {len(llm_work):,}")
        if ALLOW_PID_ONLY_FALLBACK:
            need = llm_work["qid"] == ""
            if need.any():
                fb = llm_work.loc[need, ["pid", "LLM"]].merge(nist_pid_qids, on="pid", how="inner")
                fb = fb.rename(columns={"qid": "qid_fb"})
                llm_work = llm_work.merge(fb[["pid", "qid_fb"]], on="pid", how="left")
                llm_work["qid"] = llm_work["qid"].where(
                    llm_work["qid"] != "",
                    llm_work["qid_fb"].fillna(""),
                )
                llm_work.drop(columns=["qid_fb"], inplace=True)
                resolved_after_fb = (llm_work["qid"] != "").sum()
                print(f"[LLM ] after pid-only fallback, qid resolved: {resolved_after_fb:,}")

    return llm_work, unparsable, total_rows, QID_FROM_LLM, llm_raw


def write_unresolved_if_needed(
    llm_work: pd.DataFrame,
    llm_raw: pd.DataFrame,
    pcol_original: str,
    ptext_col_original: str,
) -> pd.DataFrame:
    """For the mapping path, write rows that still have no qid."""
    if "key_pass" not in llm_work.columns:
        return pd.DataFrame()

    no_qid_mask = llm_work["qid"] == ""
    unresolved = llm_work.loc[no_qid_mask, ["pid", "key_pass"]].copy()
    if len(unresolved) == 0:
        return unresolved

    raw_copy = llm_raw.copy()
    raw_copy["__pid__"] = raw_copy[pcol_original].astype(str).str.strip()
    raw_copy["__key__"] = raw_copy[ptext_col_original].map(norm_text)

    stub = unresolved.rename(columns={"pid": "__pid__", "key_pass": "__key__"})
    to_write = stub.merge(raw_copy, on=["__pid__", "__key__"], how="left").drop(columns=["__pid__", "__key__"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    to_write.to_csv(OUT_UNRESOLVED, index=False, encoding="utf-8")
    print(f"[LLM ] wrote rows with unresolved qid to: {OUT_UNRESOLVED}  (rows={len(unresolved):,})")
    return unresolved


def load_topics_full() -> pd.DataFrame:
    """
    Load all topics and normalise to language-driven fields.

    - LANG == "raw": qid, pid_resolved, query, passage
    - otherwise:     qid, pid_resolved, query_<LANG>, passage_injected
    """
    topic_files = sorted(TOPICS_DIR.glob(TOPICS_GLOB))
    parts = []

    for fp in topic_files:
        df_t = read_csv_smart(fp)

        # If the file doesn't match the expected schema for this LANG, skip it.
        needed = {"qid", TOPIC_PID_COL, TOPIC_QUERY_COL, TOPIC_PASSAGE_COL}
        if not needed.issubset(df_t.columns):
            # small tolerance: we at least need qid and pid
            if not {"qid", TOPIC_PID_COL}.issubset(df_t.columns):
                continue

        keep: Dict[str, pd.Series] = {}
        keep["qid"] = df_t["qid"].astype(str).str.strip()
        keep["pid_resolved"] = df_t[TOPIC_PID_COL].astype(str).str.strip()

        if TOPIC_QUERY_COL in df_t.columns:
            keep[TOPIC_QUERY_COL] = df_t[TOPIC_QUERY_COL]
        if TOPIC_PASSAGE_COL in df_t.columns:
            keep[TOPIC_PASSAGE_COL] = df_t[TOPIC_PASSAGE_COL]

        parts.append(pd.DataFrame(keep))

    if not parts:
        # empty frame with expected columns for this LANG
        if LANG == "raw":
            return pd.DataFrame(
                columns=["qid", "pid_resolved", "pid_qrels", "query", "passage"]
            )
        else:
            return pd.DataFrame(
                columns=["qid", "pid_resolved", "pid_qrels", f"query_{LANG}", "passage_injected"]
            )

    topics_full = pd.concat(parts, ignore_index=True)
    topics_full["pid_qrels"] = topics_full.get("pid_resolved", "").astype(str)
    topics_full = topics_full.drop_duplicates(subset=["qid", "pid_resolved"], keep="first")
    return topics_full

def main():
    # 1) Load NIST
    nist, nist_pid_qids = load_nist()

    # 2) Pair map
    try:
        pair_map = build_pair_map()
    except FileNotFoundError as e:
        print(f"[TOPICS] Warning: {e}")
        pair_map = {}

    # 3) LLM
    llm_raw_probe = read_csv_smart(LLM_FILE)
    pcol_original  = pick_pid_col(llm_raw_probe)
    ptext_original = pick_col(llm_raw_probe, ["passage_eng", "passage_injected", "passage_en", "passage"])
    llm_work, unparsable, total_rows, QID_FROM_LLM, llm_raw_full = load_llm(nist_pid_qids, pair_map)

    # 4) unresolved-QID writer (only if we had to map)
    unresolved = pd.DataFrame()
    if not QID_FROM_LLM and ptext_original:
        unresolved = write_unresolved_if_needed(llm_work, llm_raw_full, pcol_original, ptext_original)

    # 5) drop qid-empty
    before_drop = len(llm_work)
    llm_work = llm_work[llm_work["qid"] != ""].copy()
    dropped = before_drop - len(llm_work)
    if dropped:
        print(f"[LLM ] dropped rows with no qid after mapping: {dropped:,}")

    # 6) dedup on (qid,pid)
    before_dedup = len(llm_work)
    llm_work = llm_work.drop_duplicates(subset=["qid", "pid"], keep="first")
    after_dedup = len(llm_work)
    if after_dedup != before_dedup:
        print(f"[LLM ] de-duplicated (qid,pid): {before_dedup:,} -> {after_dedup:,}")

    # 7) inner join for confusion matrix
    paired = nist.merge(llm_work[["qid", "pid", "LLM"]], on=["qid", "pid"], how="inner")
    print(f"[JOIN] Pairs after join (qid,pid): {len(paired):,}")

    # 7a) load topics for reconstructing missing rows in full format
    topics_full = load_topics_full()

    # 7b) NIST rows that did NOT get LLM → output in language-specific format
    nist_with_llm = nist.merge(
        llm_work[["qid", "pid", "LLM"]],
        on=["qid", "pid"],
        how="left",
        indicator=True,
    )
    nist_missing = nist_with_llm[nist_with_llm["_merge"] == "left_only"].drop(columns=["_merge"])

    if len(nist_missing):
        # join to topics on (qid, pid == pid_resolved)
        miss_joined = nist_missing.merge(
            topics_full,
            left_on=["qid", "pid"],
            right_on=["qid", "pid_resolved"],
            how="left",
        )

        if LANG == "raw":
            out_df = pd.DataFrame()
            out_df["qid"] = miss_joined["qid"]
            out_df["query"] = miss_joined.get("query", "")
            out_df["pid_qrels"] = miss_joined.get("pid_qrels", miss_joined["pid"]).fillna(miss_joined["pid"])
            out_df["pid_resolved"] = miss_joined.get("pid_resolved", miss_joined["pid"]).fillna(miss_joined["pid"])
            out_df["passage"] = miss_joined.get("passage", "")
            out_df["relevance"] = miss_joined["NIST"]
            # enforce column order
            out_df = out_df[
                ["qid", "query", "pid_qrels", "pid_resolved", "passage", "relevance"]
            ]
        else:
            lang_qcol = f"query_{LANG}"
            out_df = pd.DataFrame()
            out_df["qid"] = miss_joined["qid"]
            out_df[lang_qcol] = miss_joined.get(lang_qcol, "")
            out_df["pid_qrels"] = miss_joined.get("pid_qrels", miss_joined["pid"]).fillna(miss_joined["pid"])
            out_df["pid_resolved"] = miss_joined.get("pid_resolved", miss_joined["pid"]).fillna(miss_joined["pid"])
            out_df["passage_injected"] = miss_joined.get("passage_injected", "")
            out_df["relevance"] = miss_joined["NIST"]
            out_df = out_df[
                ["qid", lang_qcol, "pid_qrels", "pid_resolved", "passage_injected", "relevance"]
            ]

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(OUT_NIST_MISSING, index=False, encoding="utf-8")
        print(f"[DIAG] NIST rows with NO matching LLM label: {len(out_df):,}")
        print(f"[DIAG] Saved to: {OUT_NIST_MISSING}")

        # chunked
        write_chunked_csv(out_df, OUT_DIR / "missing_nist", "nist_not_joined", 500)
    else:
        print("[DIAG] All NIST rows had a matching LLM label.")

    # 7c) LLM rows that are not in NIST (keep simple)
    llm_with_nist = llm_work.merge(
        nist[["qid", "pid"]],
        on=["qid", "pid"],
        how="left",
        indicator=True,
    )
    llm_extra = llm_with_nist[llm_with_nist["_merge"] == "left_only"].drop(columns=["_merge"])
    if len(llm_extra):
        llm_extra.to_csv(OUT_LLM_EXTRA, index=False, encoding="utf-8")
        print(f"[DIAG] LLM rows with NO matching NIST judgment: {len(llm_extra):,}")
        write_chunked_csv(llm_extra, OUT_DIR / "missing_llm", "llm_not_in_nist", 500)
    else:
        print("[DIAG] All LLM rows had a matching NIST judgment.")

    # 8) confusion matrix
    cm = pd.crosstab(
        index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM"],  categories=LABELS, ordered=True),
        dropna=False,
    )
    cm.index.name = "NIST"
    cm.columns.name = "LLM"
    cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0

    # ===== metrics (imported) =====
    mae = compute_mae(paired["NIST"], paired["LLM"])
    kappa_weighted = compute_weighted_kappa_ordinal(cm)

    # binary version
    paired["NIST_bin"] = binarize_labels(paired["NIST"])
    paired["LLM_bin"]  = binarize_labels(paired["LLM"])

    cm_bin = pd.crosstab(
        index=pd.Categorical(paired["NIST_bin"], categories=[0, 1], ordered=True),
        columns=pd.Categorical(paired["LLM_bin"],  categories=[0, 1], ordered=True),
        dropna=False,
    )
    cm_bin.index.name = "NIST_bin"
    cm_bin.columns.name = "LLM_bin"

    kappa_binary = compute_unweighted_kappa(cm_bin)

    # 9) outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cm.to_csv(OUT_COUNTS)
    cm_pct.round(2).to_csv(OUT_PCT)

    metrics_df = pd.DataFrame(
        [
            {"metric": "mae", "value": mae},
            {"metric": "kappa_weighted_4pt", "value": kappa_weighted},
            {"metric": "kappa_binary_2pt", "value": kappa_binary},
            {"metric": "pairs", "value": float(len(paired))},
        ]
    )
    metrics_df.to_csv(OUT_DIR / "metrics_llm_vs_nist.csv", index=False, encoding="utf-8")

    print(f"[METRIC] MAE:                    {mae:.4f}")
    print(f"[METRIC] κ (weighted, 4-pt):     {kappa_weighted:.4f}")
    print(f"[METRIC] κ (binary 0-1 vs 2-3):  {kappa_binary:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title(f"Confusion Matrix: NIST vs LLM — {MODEL} {TREC_DL_YEAR} {LANG}")
    plt.ylabel("NIST label")
    plt.xlabel("LLM label")
    plt.tight_layout()
    plt.savefig(OUT_SVG, dpi=200)
    plt.show()

    print(f"[DONE] Wrote counts to: {OUT_COUNTS}")
    print(f"[DONE] Wrote row-% to:  {OUT_PCT}")
    print(f"[DONE] Saved heatmap to: {OUT_SVG}")
    if unparsable > 0:
        print(f"[DONE] Unparseable label rows saved to: {OUT_UNPARSEABLE}")
    if not QID_FROM_LLM and len(unresolved):
        print(f"[DONE] Unresolved-qid rows saved to: {OUT_UNRESOLVED}")
    if Path(OUT_NIST_MISSING).exists():
        print(f"[DONE] NIST-missing rows saved to: {OUT_NIST_MISSING}")
    if Path(OUT_LLM_EXTRA).exists():
        print(f"[DONE] LLM-extra rows saved to: {OUT_LLM_EXTRA}")

if __name__ == "__main__":
    main()
