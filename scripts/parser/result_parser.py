#!/usr/bin/env python3
"""
Parse Bedrock *_llm_responses_*.json logs into:
  query,docid,passage,relevance

- Reads each item: {query, docid, prompt, response_text, full_response}
- Extracts `passage` from the prompt (after "Consider the following passage:")
- Extracts overall score `O` (as `relevance`) from the model output
"""

from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

# =========================
# CONFIG: EDIT THESE
# =========================
LOGS_DIR   = Path("outputs/trec_dl/logs")
OUT_CSV    = Path("outputs/trec_dl_llm_label/relevant/overall_from_logs.csv")
FILE_GLOB  = "20250918_102202_llm_responses_openai.gpt-oss-20b-1_0_top2.json"

# Passage extraction anchor & guard phrases (regex-friendly, case-insensitive)
PASSAGE_ANCHOR = r"Consider the following passage:"
PASSAGE_STOP_BOUNDARY = r"(?:\n\s*(?:Split this problem|For match|For trustworthiness|For overall|Also consider|Strictly produce)|$)"

# Optional: collapse whitespace in passage to single spaces for CSV friendliness
COLLAPSE_PASSAGE_WHITESPACE = True
# =========================


# --- Derived / compiled regexes ---
PASSAGE_RE = re.compile(
    rf"{PASSAGE_ANCHOR}\s*(?P<p>.*?){PASSAGE_STOP_BOUNDARY}",
    re.DOTALL | re.IGNORECASE,
)

# Forgiving regex to pull "O": <digit> from messy generations
O_REGEX = re.compile(r'"O"\s*:\s*(\d)', re.DOTALL)


def extract_passage_from_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    m = PASSAGE_RE.search(prompt)
    if m:
        passage = m.group("p").strip()
        if COLLAPSE_PASSAGE_WHITESPACE:
            passage = re.sub(r"\s+", " ", passage)
        return passage
    # Fallback: take everything after last "passage:" literal
    i = prompt.lower().rfind("passage:")
    if i != -1:
        passage = prompt[i + len("passage:"):].strip()
        if COLLAPSE_PASSAGE_WHITESPACE:
            passage = re.sub(r"\s+", " ", passage)
        return passage
    return ""


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def extract_text_from_full_response(full: dict) -> str:
    """Fallback: pull text from full_response if response_text wasn't stored."""
    try:
        contents = full["output"]["message"]["content"]
        # Prefer a chunk that contains "O":
        for item in contents:
            t = item.get("text", "")
            if t and O_REGEX.search(t):
                return t
        # Else, first text chunk
        for item in contents:
            t = item.get("text", "")
            if t:
                return t
    except Exception:
        pass
    return ""


def extract_overall(text: str) -> Optional[int]:
    """Try to find Overall (O) score from a variety of shapes."""
    if not text:
        return None

    # Strategy 1: direct JSON parse (dict or [dict])
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "O" in obj:
            return safe_int(obj["O"])
        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict) and "O" in first:
                return safe_int(first["O"])
    except Exception:
        pass

    # Strategy 2: locate a JSON array segment and parse the first object
    try:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            arr = json.loads(text[start:end+1])
            if isinstance(arr, list) and arr:
                first = arr[0]
                if isinstance(first, dict) and "O" in first:
                    return safe_int(first["O"])
    except Exception:
        pass

    # Strategy 3: regex fallback
    m = O_REGEX.search(text)
    if m:
        return safe_int(m.group(1))

    return None


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    log_files = sorted(LOGS_DIR.glob(FILE_GLOB))
    if not log_files:
        print(f"No log files found in: {LOGS_DIR}")
        return

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "docid", "passage", "relevance"])

        for path in log_files:
            try:
                items = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(items, list):
                    print(f"WARN: {path} is not a JSON array; skipping.")
                    continue
            except Exception as e:
                print(f"ERROR: failed to read {path}: {e}")
                continue

            for item in items:
                query = (item.get("query") or "").strip()
                docid = (item.get("docid") or "").strip()

                # passage from prompt
                prompt = item.get("prompt") or ""
                passage = extract_passage_from_prompt(prompt)

                # overall from response
                raw_text = (item.get("response_text") or "").strip()
                if not raw_text:
                    raw_text = extract_text_from_full_response(item.get("full_response", {}) or {})

                O = extract_overall(raw_text)
                relevance = "" if O is None else O

                writer.writerow([query, docid, passage, relevance])

    print(f"Wrote CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
