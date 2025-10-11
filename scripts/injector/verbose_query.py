#!/usr/bin/env python3
from __future__ import annotations
import csv, sys, json
from pathlib import Path
import boto3

# ========= USER SETTINGS (edit these) =========
INPUT_CSV  = Path("outputs/queries/trec_dl_2023_all_queries.csv")          # your input CSV
OUTPUT_CSV = Path("outputs/queries/trec_dl_2023_expanded_queries.csv")   # where to write results
PROMPT_FILE = Path("prompts/query_prompt.txt")   # prompt .txt with {query}
# ==============================================

# Fixed config (change if needed)
MODEL_ID   = "openai.gpt-oss-20b-1:0"   # Bedrock modelId
REGION     = "us-west-2"           # AWS region
QUERY_COL  = "query"                    # column to read
MAX_TOKENS = 500
TEMPERATURE= 0.3

# Allow very large CSV cells
def _bump_field_limit():
    try:
        limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
        while limit >= 131072:
            try:
                csv.field_size_limit(limit); return
            except OverflowError:
                limit //= 2
    except Exception:
        pass
_bump_field_limit()

def _extract_text(resp: dict) -> str:
    try:
        for block in resp["output"]["message"]["content"]:
            if isinstance(block, dict) and "text" in block:
                return block["text"].strip()
    except Exception:
        pass
    return ""

def main():
    # Basic checks
    if not INPUT_CSV.exists():
        sys.exit(f"[FATAL] Input CSV not found: {INPUT_CSV}")
    if not PROMPT_FILE.exists():
        sys.exit(f"[FATAL] Prompt file not found: {PROMPT_FILE}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Bedrock client
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    # Load prompt template
    tmpl = PROMPT_FILE.read_text(encoding="utf-8")
    if "{query}" not in tmpl:
        print("[WARN] Prompt does not contain '{query}' placeholder.", file=sys.stderr)

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as fin, \
         OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, skipinitialspace=True)
        if QUERY_COL not in (reader.fieldnames or []):
            sys.exit(f"[FATAL] Missing '{QUERY_COL}' column. Found: {reader.fieldnames}")

        writer = csv.DictWriter(fout, fieldnames=[QUERY_COL, "expanded_question"])
        writer.writeheader()

        for row in reader:
            q = (row.get(QUERY_COL) or "").strip()
            if not q:
                continue

            prompt_text = tmpl.format(query=q)
            body = {
                "messages": [{"role": "user", "content": [{"text": prompt_text}]}],
                "inferenceConfig": {"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE, "topP": 1.0},
            }

            try:
                resp = bedrock.converse(modelId=MODEL_ID, **body)
                expanded = _extract_text(resp)
            except Exception as e:
                expanded = f"[ERROR] {type(e).__name__}: {e}"

            writer.writerow({QUERY_COL: q, "expanded_question": expanded})

    print(f"[DONE] Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
