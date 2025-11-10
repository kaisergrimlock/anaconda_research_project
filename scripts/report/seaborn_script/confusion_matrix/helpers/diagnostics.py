# diagnostics.py
from __future__ import annotations
from typing import Tuple, List
import pandas as pd

from scripts.csv_helpers import norm_text

_COLUMNS = [
    "qid",
    "query",
    "pid_qrels",
    "pid_resolved",
    "passage",
    "relevance",
    "query_ru",
    "passage_injected",
]

def _empty_series_like(df: pd.DataFrame, dtype: str = "object") -> pd.Series:
    return pd.Series([""] * len(df), index=df.index, dtype=dtype)

def _series_if_present(df: pd.DataFrame, col: str, *, dtype: str = "object") -> pd.Series:
    """Return df[col] coerced to dtype if present, else empty Series aligned to df."""
    if col in df.columns:
        s = df[col]
        # allow 'relevance' numeric passthrough when dtype is 'keep'
        if dtype == "keep":
            return s
        return s.astype(dtype).fillna("" if dtype == "object" else 0)
    return _empty_series_like(df, dtype if dtype != "keep" else "object")

def _first_available(df: pd.DataFrame, cols: List[str], *, dtype: str = "object") -> pd.Series:
    """Return first existing column from list, coerced; else empty Series."""
    for c in cols:
        if c in df.columns:
            s = df[c]
            if dtype == "keep":
                return s
            return s.astype(dtype).fillna("" if dtype == "object" else 0)
    return _empty_series_like(df, dtype if dtype != "keep" else "object")

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with exactly the required columns in order, filling missing with ''."""
    out = pd.DataFrame({c: df[c] if c in df.columns else "" for c in _COLUMNS})
    # coerce textual columns to str
    for c in ["qid", "query", "pid_qrels", "pid_resolved", "passage", "query_ru", "passage_injected"]:
        out[c] = out[c].astype(str).fillna("")
    # leave relevance as-is if present; else set to ""
    if "relevance" in df.columns:
        out["relevance"] = df["relevance"]
    else:
        out["relevance"] = ""
    return out[_COLUMNS]

# --- Public, pure helpers ----------------------------------------------------

def build_unresolved_rows(
    *,
    lang: str,                # kept for signature compatibility
    topic_passage_col: str,
    llm_work: pd.DataFrame,
    llm_raw: pd.DataFrame,
    pcol_original: str,
    ptext_col_original: str,
    topics_full: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct a report of rows where qid could not be resolved, with the exact
    fixed schema:
    qid,query,pid_qrels,pid_resolved,passage,relevance,query_ru,passage_injected
    Pure: no file I/O, no prints.
    """
    if "key_pass" not in llm_work.columns:
        return pd.DataFrame(columns=_COLUMNS)

    unresolved = llm_work.loc[llm_work["qid"] == "", ["pid", "key_pass"]].copy()
    if unresolved.empty:
        return pd.DataFrame(columns=_COLUMNS)

    raw_copy = llm_raw.copy()
    raw_copy["__pid__"] = raw_copy[pcol_original].astype(str).str.strip()
    raw_copy["__key__"] = raw_copy[ptext_col_original].map(norm_text)

    stub = unresolved.rename(columns={"pid": "__pid__", "key_pass": "__key__"})
    merged_raw = stub.merge(raw_copy, on=["__pid__", "__key__"], how="left")

    if topics_full.empty:
        merged = merged_raw
    else:
        t = topics_full.copy()
        if "pid_resolved" not in t.columns and "pid" in t.columns:
            t = t.rename(columns={"pid": "pid_resolved"})
        t["pid_resolved"] = t["pid_resolved"].astype(str).str.strip()
        if topic_passage_col in t.columns:
            t["__key__"] = t[topic_passage_col].map(norm_text)
        else:
            t["__key__"] = ""
        t = t.rename(columns={"pid_resolved": "__pid__"})
        merged = merged_raw.merge(t, on=["__pid__", "__key__"], how="left", suffixes=("", "_topic"))

    out = pd.DataFrame(index=merged.index)
    out["qid"] = _series_if_present(merged, "qid")
    out["query"] = _first_available(merged, ["query", "query_topic"])
    out["pid_qrels"] = _first_available(merged, ["pid_qrels", pcol_original, "__pid__"])
    out["pid_resolved"] = _first_available(merged, ["pid_resolved", pcol_original, "__pid__"])
    out["passage"] = _first_available(merged, [ptext_col_original, "passage"])
    # relevance: keep numeric if present on any of these
    rel = _first_available(merged, ["relevance", "NIST", "label", "nist"], dtype="keep")
    out["relevance"] = rel if len(rel) == len(merged) else _empty_series_like(merged)
    out["query_ru"] = _first_available(merged, ["query_ru", "query_ru_topic"])
    # fall back to base passage if injected not available
    passage_inj = _series_if_present(merged, "passage_injected")
    out["passage_injected"] = passage_inj.where(passage_inj != "", out["passage"])

    return _ensure_schema(out)

def build_missing_and_extras(
    *,
    lang: str,  # kept for signature compatibility
    nist: pd.DataFrame,
    llm: pd.DataFrame,
    topics_full: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute (a) NIST rows with no LLM judgment (fixed schema),
            (b) LLM rows with no NIST judgment (raw columns, caller can filter/shape).
    Returns:
      missing_out: DataFrame with the exact fixed columns in order
      llm_extra:   DataFrame of LLM rows not present in NIST (unchanged shape)
    Pure: no file I/O, no prints.
    """
    nist_with_llm = nist.merge(llm[["qid", "pid", "LLM"]], on=["qid", "pid"], how="left", indicator=True)
    nist_missing = nist_with_llm[nist_with_llm["_merge"] == "left_only"].drop(columns=["_merge"])

    if nist_missing.empty:
        missing_out = pd.DataFrame(columns=_COLUMNS)
    else:
        # Bring in topic-side fields (qid, pid_resolved, query_ru, passage_injected, etc.)
        miss_joined = nist_missing.merge(
            topics_full, left_on=["qid", "pid"], right_on=["qid", "pid_resolved"], how="left"
        )

        out = pd.DataFrame(index=miss_joined.index)
        out["qid"] = miss_joined["qid"].astype(str).fillna("")

        # --- Prefer NIST's 'query' if present, else topic-side 'query'/'query_topic'
        # If nist had 'query', it is already in miss_joined (came from nist_with_llm).
        q_from_nist = _series_if_present(miss_joined, "query")
        q_from_topics = _first_available(miss_joined, ["query", "query_topic"])
        out["query"] = q_from_nist.where(q_from_nist != "", q_from_topics)

        # pid_qrels prefer explicit column; fallback to pid
        out["pid_qrels"] = _first_available(miss_joined, ["pid_qrels"], dtype="object")
        out["pid_qrels"] = out["pid_qrels"].where(out["pid_qrels"] != "", miss_joined["pid"].astype(str))

        # pid_resolved prefer explicit; fallback to pid
        out["pid_resolved"] = _first_available(miss_joined, ["pid_resolved"], dtype="object")
        out["pid_resolved"] = out["pid_resolved"].where(out["pid_resolved"] != "", miss_joined["pid"].astype(str))

        # --- Prefer NIST's 'passage' if present, else topic-side
        p_from_nist = _series_if_present(miss_joined, "passage")
        p_from_topics = _series_if_present(miss_joined, "passage")  # topic-side 'passage' if it exists there
        out["passage"] = p_from_nist.where(p_from_nist != "", p_from_topics)

        # relevance from NIST
        out["relevance"] = _series_if_present(miss_joined, "NIST", dtype="keep")

        # topic-side language/injected fields (if any)
        out["query_ru"] = _series_if_present(miss_joined, "query_ru")
        out["passage_injected"] = _series_if_present(miss_joined, "passage_injected")

        # enforce schema/order
        missing_out = _ensure_schema(out)

    llm_with_nist = llm.merge(nist[["qid", "pid"]], on=["qid", "pid"], how="left", indicator=True)
    llm_extra = llm_with_nist[llm_with_nist["_merge"] == "left_only"].drop(columns=["_merge"])

    return missing_out, llm_extra
