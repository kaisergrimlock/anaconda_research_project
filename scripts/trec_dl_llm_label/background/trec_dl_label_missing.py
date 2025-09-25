#!/usr/bin/env python3
import math
from time import strftime
import boto3
import json
import csv
from pathlib import Path
from datetime import datetime

# ----------------------------
# Configurable Paths
# ----------------------------
now = str(datetime.now().strftime("_%Y%m%d_%H%M%S"))
PROMPT_NAME = "prompt"  # prompt template file, e.g., prompts/prompt.txt
PROMPT_FILE = Path("prompts") / f"{PROMPT_NAME}.txt"

# >>> Use the missing list we just created (must contain: query,docid,passage)
MISSING_CSV = Path("outputs/trec_dl_llm_label/processed/missing_labels.csv")

# >>> Append new judgments here
TARGET_ALL_DOCS = Path("outputs/trec_dl_llm_label/processed/all_docs_label_cleaned.csv")

# Logs / metrics (unchanged)
OUTPUT_DIR  = Path(f"outputs/trec_dl_llm_label/relevant/{PROMPT_NAME}/{now}")  # per-run JSON logs live here
LOG_DIR     = Path("outputs/trec_dl/logs")
TOKENS_CSV  = Path("outputs/trec_dl_llm_label/token_usage.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TARGET_ALL_DOCS.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Bedrock Config
# ----------------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

MODELS = [
    "openai.gpt-oss-20b-1:0",
]

INFERENCE_CONFIG = {
    "maxTokens": 1000,
    "temperature": 0.0,
    "topP": 1.0,
}

# ----------------------------
# Utility
# ----------------------------
def round_half_up(n):
    try:
        return math.floor(float(n) + 0.5)
    except (TypeError, ValueError):
        return ""

def timestamp_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def timestamp_iso():
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

def sniff_reader(path: Path):
    f = path.open("r", encoding="utf-8-sig", newline="")
    sample = f.read(4096); f.seek(0)
    # Try sniffer, fall back to comma
    try:
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(f, dialect=dialect)
    except csv.Error:
        reader = csv.DictReader(f, delimiter=",")
        return f, reader

    # Safeguard: if header wasn't split, force comma
    if reader.fieldnames and len(reader.fieldnames) == 1 and "," in reader.fieldnames[0]:
        f.seek(0)
        reader = csv.DictReader(f, delimiter=",")

    return f, reader

# ----------------------------
# Load prompt + inputs
# ----------------------------
prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

# Read missing list (must have: query,docid,passage)
fin, rdr = sniff_reader(MISSING_CSV)
try:
    need = {"docid", "query", "passage"}
    cols = set(rdr.fieldnames or [])
    if not need.issubset(cols):
        raise KeyError(f"{MISSING_CSV} missing columns {sorted(need - cols)}; has {sorted(cols)}")
    data_rows = [row for row in rdr]
finally:
    fin.close()

# Load already-judged docids to avoid duplicating rows
existing_docids = set()
if TARGET_ALL_DOCS.exists():
    fexist, rdr2 = sniff_reader(TARGET_ALL_DOCS)
    try:
        if rdr2.fieldnames and "docid" in rdr2.fieldnames:
            for r in rdr2:
                did = (r.get("docid") or "").strip()
                if did:
                    existing_docids.add(did)
    finally:
        fexist.close()

# ----------------------------
# Main Logic
# ----------------------------
for model_id in MODELS:
    print(f"\n--- Running inference for model: {model_id} ---")

    logs = []
    total_input_tokens = 0
    total_output_tokens = 0
    run_id = timestamp_id()

    # We will append directly to TARGET_ALL_DOCS (header only if file is new/empty)
    header_needed = (not TARGET_ALL_DOCS.exists()) or (TARGET_ALL_DOCS.stat().st_size == 0)
    out = TARGET_ALL_DOCS.open("a", encoding="utf-8", newline="")
    writer = csv.writer(out)
    try:
        if header_needed:
            writer.writerow(["query", "docid", "passage", "relevance"])

        judged_count = 0
        skipped_existing = 0

        for row in data_rows:
            query = (row["query"] or "").strip()
            docid = (row["docid"] or "").strip()
            passage_text = (row["passage"] or "").strip()

            if not docid or docid in existing_docids:
                skipped_existing += 1
                continue

            # Prepare prompt
            prompt = prompt_template.format(query=query, passage=passage_text)

            messages = [{"role": "user", "content": [{"text": prompt}]}]
            kwargs = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": INFERENCE_CONFIG,
            }

            # Call Bedrock API
            resp = bedrock.converse(**kwargs)

            # Extract response text
            try:
                if model_id.startswith("openai."):
                    text = resp["output"]["message"]["content"][1]["text"]
                else:
                    text = resp["output"]["message"]["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                text = ""

            # Parse for JSON score
            try:
                scores = json.loads(text)
                llm_score = scores.get("O", "")
            except Exception:
                llm_score = ""

            # Append a new judged row immediately
            writer.writerow([query, docid, passage_text, llm_score])
            judged_count += 1
            existing_docids.add(docid)  # prevent double-judging within same run

            # Collect logs
            logs.append({
                "query": query,
                "docid": docid,
                "prompt": prompt,
                "response_text": text,
                "full_response": resp
            })

            # Token usage
            usage = resp.get("usage", {})
            total_input_tokens  += int(usage.get("inputTokens", 0) or 0)
            total_output_tokens += int(usage.get("outputTokens", 0) or 0)

    finally:
        out.close()

    # Write JSON logs for this run/model
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    log_file = LOG_DIR / f"{run_id}_llm_responses_{safe_model}_missing.json"
    with log_file.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    # Token usage summary
    token_row = {
        "run_id": run_id,
        "timestamp": timestamp_iso(),
        "model": model_id,
        "num_examples": len(data_rows),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "labels_csv": str(TARGET_ALL_DOCS),
        "log_json": str(log_file),
    }
    append_token_row(TOKENS_CSV, token_row)

    print(f"Appended {judged_count} new rows to: {TARGET_ALL_DOCS} (skipped {skipped_existing} already-present docids)")
    print(f"Saved logs:       {log_file}")
    print(f"Appended tokens:  {TOKENS_CSV}")
    print(f"Total input tokens used by {model_id}: {total_input_tokens}")
    print(f"Total output tokens used by {model_id}: {total_output_tokens}")
