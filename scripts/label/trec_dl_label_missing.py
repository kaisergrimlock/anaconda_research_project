#!/usr/bin/env python3
"""
Single-file LLM labeller (replace-by-(query,docid)):

- Reads ONE input CSV (INPUT_CSV) with columns incl. query, docid, passage
- Calls the model row-by-row (serial)
- Updates OUTPUT_FILE so that rows with the same (query, docid) are REPLACED
  (no duplicates; identical docids under different queries are preserved)
- Rows with missing query/docid are kept DISTINCT by using a per-row unique key
- Saves a JSON log per run and appends token usage to TOKENS_CSV

Output schema (CSV): query,docid,passage,relevance
"""

from __future__ import annotations
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Tuple, List

import boto3
from botocore.config import Config

# ============================
# AWS / Bedrock client config
# ============================
cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)

# ----------------------------
# Paths (resolved relative to repo root)
# ----------------------------
# repo root = two levels up from this file: <repo>/scripts/label/this_file.py
REPO_ROOT  = Path(__file__).resolve().parents[2]

PROMPT_NAME = "utility"
PROMPT_FILE = REPO_ROOT / "prompts" / f"{PROMPT_NAME}.txt"

# <<< choose ONE input file (NO leading slash) >>>
INPUT_CSV   = REPO_ROOT / "outputs" / "queries" / "trec_dl_2023_expanded_queries.csv"

# Output will be UPDATED in-place; matching (query,docid) rows replaced
OUTPUT_FILE = REPO_ROOT / "outputs" / "llm_label" / "trec_dl_2023_irrelevant.csv"

LOG_DIR     = REPO_ROOT / "logs"
TOKENS_CSV  = REPO_ROOT / "token_usage.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Bedrock / model config
# ----------------------------
MODELS = [
    "openai.gpt-oss-20b-1:0",
]

INFERENCE_CONFIG = {
    "maxTokens": 1000,
    "temperature": 0.0,
    "topP": 1.0,
}

# ----------------------------
# Utilities
# ----------------------------
def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def timestamp_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def append_token_row(tokens_csv: Path, row: dict):
    file_exists = tokens_csv.exists()
    with tokens_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id","timestamp","model","num_examples",
                "input_tokens","output_tokens","total_tokens",
                "labels_csv","log_json",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def read_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r]

def parse_llm_text_to_score(text: str) -> str:
    """Expect model returns JSON like {"O": <label>} -> returns label as string; else ''. """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return str(parsed.get("O", ""))
    except Exception:
        pass
    return ""

def extract_text_from_resp(model_id: str, resp: dict) -> str:
    """
    Bedrock Converse response shape:
      resp["output"]["message"]["content"] is a list of blocks with {"text": "..."}.
    Some models put the 'answer' at index 1; keep prior compatibility.
    """
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        else:
            return resp["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""

def usage_from_resp(resp: dict) -> Tuple[int, int]:
    usage = resp.get("usage", {}) or {}
    return int(usage.get("inputTokens", 0) or 0), int(usage.get("outputTokens", 0) or 0)

def _norm(s: str) -> str:
    return (s or "").strip()

def _key_from_values(query: str, docid: str, idx_hint: int) -> Tuple[str, str]:
    """
    Make a stable composite key (query, docid). If either is blank,
    attach a per-row tag so "blank" entries don't collapse together.
    """
    q = _norm(query)
    d = _norm(docid)
    if q == "" or d == "":
        tag = f"__row_{idx_hint}"
        return (q or tag, d or tag)
    return (q, d)

def load_existing_output(path: Path) -> "OrderedDict[tuple[str,str], list]":
    """
    Load OUTPUT_FILE into an OrderedDict keyed by (query, docid).
    Value: [query, docid, passage, relevance]
    Preserves file order for non-updated rows.
    """
    out_map: "OrderedDict[tuple[str,str], list]" = OrderedDict()
    if not path.exists():
        return out_map
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # expect ["query","docid","passage","relevance"]
        for i, row in enumerate(reader):
            q, did, pas, rel = (row + ["", "", "", ""])[:4]
            k = _key_from_values(q, did, i)
            out_map[k] = [q, did, pas, rel]
    return out_map

def save_output(path: Path, rows_by_key: "OrderedDict[tuple[str,str], list]"):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "docid", "passage", "relevance"])
        for row in rows_by_key.values():
            w.writerow(row)

# ----------------------------
# Core runner for ONE file
# ----------------------------
def run_single_file(model_id: str):
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # Path sanity (helpful when running from IDE/terminal)
    print(f"[PATHS] INPUT_CSV={INPUT_CSV}")
    print(f"[PATHS] OUTPUT_FILE={OUTPUT_FILE}")
    print(f"[PATHS] PROMPT_FILE={PROMPT_FILE}")
    print(f"[PATHS] LOG_DIR={LOG_DIR}")

    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    rows = read_rows(INPUT_CSV)
    total_rows = len(rows)
    print(f"[{INPUT_CSV.name}] Loaded {total_rows} rows")

    # Load existing output -> we'll update/replace by (query,docid)
    output_map = load_existing_output(OUTPUT_FILE)

    # Prepare logs
    run_id = timestamp_id()
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    log_path   = LOG_DIR / f"{run_id}_llm_responses_{safe_model}_{INPUT_CSV.stem}.json"

    bedrock = boto3.client("bedrock-runtime", config=cfg)

    total_in = 0
    total_out = 0
    logs = []
    processed = 0

    for idx, row in enumerate(rows, start=1):
        query = row.get("query", "")
        docid = row.get("docid", "")
        passage_text = _norm(row.get("passage", ""))

        prompt = prompt_template.format(query=query, passage=passage_text)
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": INFERENCE_CONFIG,
        }

        try:
            resp = bedrock.converse(**kwargs)
            text = extract_text_from_resp(model_id, resp)
            score = parse_llm_text_to_score(text)
            in_tok, out_tok = usage_from_resp(resp)
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] Last docid={docid!r} (row {idx}). Stopping.")
            break
        except Exception as api_err:
            print(f"[ERROR] API failed on docid={docid!r} (row {idx}) :: {api_err}")
            text = ""
            score = ""
            in_tok = out_tok = 0
            resp = {"error": str(api_err)}

        # Replace (or insert) by composite key (query, docid), with safe unique key if blank
        k = _key_from_values(query, docid, idx)
        output_map[k] = [query, docid, passage_text, score]

        logs.append({
            "query": query, "docid": docid, "prompt": prompt,
            "response_text": text, "full_response": resp
        })

        total_in  += in_tok
        total_out += out_tok
        processed += 1
        print(f"[{INPUT_CSV.name}] [{idx}/{total_rows}]  tokens in/out={total_in}/{total_out}", end="\r", flush=True)

    # Save JSON log
    with log_path.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    # Write UPDATED output (all rows, with replacements applied)
    save_output(OUTPUT_FILE, output_map)

    print(f"\n[DONE] File: {INPUT_CSV.name} -> {OUTPUT_FILE.name} "
          f"(replaced by (query,docid)) | tokens in/out={total_in}/{total_out}")

    # Token usage row
    append_token_row(TOKENS_CSV, {
        "run_id": run_id,
        "timestamp": timestamp_iso(),
        "model": model_id,
        "num_examples": processed,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "labels_csv": str(OUTPUT_FILE),
        "log_json": str(log_path),
    })
    print(f"[DONE] Token usage appended to: {TOKENS_CSV}")

def main():
    # Run models one-by-one on the SAME single input file
    for model_id in MODELS:
        run_single_file(model_id)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Top-level stop.")
