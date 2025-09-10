import math
import boto3
import json
import csv
from pathlib import Path
from datetime import datetime

# ----------------------------
# Configurable Paths
# ----------------------------
PROMPT_FILE = Path("prompt.txt")
INPUT_CSV   = Path("outputs/trec_dl/combined_result_injected_eng_20.csv")

OUTPUT_DIR  = Path("outputs/trec_dl_llm_label/translated")   # CSV outputs per run/model
LOG_DIR     = Path("outputs/trec_dl/logs")                    # JSON logs
TOKENS_CSV  = Path("outputs/trec_dl_llm_label/token_usage.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Bedrock (LLM scoring)
# ----------------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

MODELS = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "mistral.mixtral-8x7b-instruct-v0:1",
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

# ----------------------------
# Main Logic
# ----------------------------
prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

with INPUT_CSV.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    data_rows = [row for row in reader]

for model_id in MODELS:
    print(f"\n--- Running inference for model: {model_id} ---")

    results = []
    logs = []
    total_input_tokens = 0
    total_output_tokens = 0
    run_id = timestamp_id()

    for row in data_rows:
        query   = row.get("query", "")
        docid   = row.get("docid", "")
        # IMPORTANT: use the injected passage for judgment
        passage_injected = (row.get("passage_injected") or "").strip()

        # Build prompt with passage_injected instead of passage
        prompt = prompt_template.format(query=query, passage=passage_injected)
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": INFERENCE_CONFIG,
        }

        # Call Bedrock
        resp = bedrock.converse(**kwargs)

        # Extract response text (Bedrock schema variant)
        try:
            if model_id.startswith("openai."):
                text = resp["output"]["message"]["content"][1]["text"]
            else:
                text = resp["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = ""

        # Parse LLM label (expects JSON with key "O")
        try:
            scores = json.loads(text)
            llm_score = scores.get("O", "")
        except Exception:
            llm_score = ""

        # Keep original columns + new relevance
        results.append([
            row.get("query", ""),
            row.get("docid", ""),
            row.get("passage", ""),       # original passage preserved
            llm_score,                    # new relevance from judging passage_injected
            row.get("query_vi", ""),      # pass through if present
            row.get("passage_injected","")
        ])

        logs.append({
            "query": query,
            "docid": docid,
            "used_passage": "passage_injected",
            "prompt": prompt,
            "response_text": text,
            "full_response": resp
        })

        usage = resp.get("usage", {})
        total_input_tokens  += int(usage.get("inputTokens", 0) or 0)
        total_output_tokens += int(usage.get("outputTokens", 0) or 0)

    # Write combined CSV per model
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    out_csv = OUTPUT_DIR / f"{run_id}_llm_labels_{safe_model}_top2.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query","docid","passage","relevance","query_vi","passage_injected"])
        writer.writerows(results)

    # Logs
    log_file = LOG_DIR / f"{run_id}_llm_responses_{safe_model}_top2.json"
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
        "labels_csv": str(out_csv),
        "log_json": str(log_file),
    }
    append_token_row(TOKENS_CSV, token_row)

    print(f"Wrote combined CSV: {out_csv}")
    print(f"Saved detailed logs: {log_file}")
    print(f"Appended token usage to: {TOKENS_CSV}")
    print(f"Total input tokens used by {model_id}: {total_input_tokens}")
    print(f"Total output tokens used by {model_id}: {total_output_tokens}")
