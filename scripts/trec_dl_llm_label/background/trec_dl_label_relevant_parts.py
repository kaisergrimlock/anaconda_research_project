from time import strftime
import boto3
import json
import csv
from pathlib import Path
from datetime import datetime
from botocore.config import Config

cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,      # handshake/socket connect
    read_timeout=300,        # wait up to 5 min for a response
    retries={"max_attempts": 8, "mode": "standard"}
)


# ----------------------------
# Configurable Paths
# ----------------------------
PROMPT_NAME = "prompt"
PROMPT_FILE = Path(f"prompts/{PROMPT_NAME}.txt")

# >>> Choose which parts to process (inclusive) <<<
START_PART = 16
END_PART   = 19

# Where the part files live & their filename pattern
PART_DIR     = Path("outputs/trec_dl/retrieved/all_topics")
PART_PATTERN = "all_topics_trecdl_2019_part{n}.csv"

# Single shared labels file (appends)
OUTPUT_FILE = Path("outputs/trec_dl_llm_label/relevant/all_llm_labels.csv")

LOG_DIR    = Path("outputs/trec_dl/logs")
TOKENS_CSV = Path("outputs/trec_dl_llm_label/token_usage.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Bedrock Config
# ----------------------------
bedrock = boto3.client("bedrock-runtime", config=cfg)

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

def iter_part_files(start_part: int, end_part: int):
    """Yield Path objects for the requested part range if they exist."""
    for n in range(start_part, end_part + 1):
        p = PART_DIR / PART_PATTERN.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")

def load_rows_from_parts(start_part: int, end_part: int):
    """Load and concatenate rows from the requested range."""
    all_rows = []
    files_used = []
    for path in iter_part_files(start_part, end_part):
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            rows = [row for row in r]
            all_rows.extend(rows)
            files_used.append(path.name)
    print(f"[INFO] Loaded {len(all_rows)} rows from {len(files_used)} files: {files_used}")
    return all_rows

# Write header to shared CSV once if needed
if not OUTPUT_FILE.exists():
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query", "docid", "passage", "relevance"])

# ----------------------------
# Main
# ----------------------------
prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
data_rows = load_rows_from_parts(START_PART, END_PART)

for model_id in MODELS:
    print(f"\n--- Running inference for model: {model_id} ---")
    total_input_tokens = 0
    total_output_tokens = 0
    run_id = timestamp_id()
    logs = []

    for idx, row in enumerate(data_rows, start=1):
        query = row.get("query", "")
        docid = row.get("docid", f"<missing-docid-{idx}>")
        passage_text = (row.get("passage", "") or "").strip()

        prompt = prompt_template.format(query=query, passage=passage_text)
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": INFERENCE_CONFIG,
        }

        try:
            resp = bedrock.converse(**kwargs)
        except KeyboardInterrupt:
            print(f"[INTERRUPTED] Last doc: {docid} (row {idx}) — flushing partials.")
            break
        except Exception as api_err:
            print(f"[ERROR] API call failed on docid={docid} (row {idx}) :: {api_err}")
            # Append blank relevance so downstream joins still work
            with OUTPUT_FILE.open("a", encoding="utf-8", newline="") as csvfile:
                csv.writer(csvfile).writerow([query, docid, passage_text, ""])
            logs.append({"query": query, "docid": docid, "prompt": prompt,
                         "response_text": "", "full_response": {"error": str(api_err)}})
            continue

        # Extract model text robustly
        try:
            if model_id.startswith("openai."):
                text = resp["output"]["message"]["content"][1]["text"]
            else:
                text = resp["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = ""

        # Parse score (expect {"O": ...})
        try:
            parsed = json.loads(text)
            llm_score = parsed.get("O", "") if isinstance(parsed, dict) else ""
        except Exception:
            llm_score = ""

        # Append one labeled row immediately
        with OUTPUT_FILE.open("a", encoding="utf-8", newline="") as csvfile:
            csv.writer(csvfile).writerow([query, docid, passage_text, llm_score])

        logs.append({"query": query, "docid": docid, "prompt": prompt,
                     "response_text": text, "full_response": resp})

        usage = resp.get("usage", {})
        total_input_tokens  += int(usage.get("inputTokens", 0) or 0)
        total_output_tokens += int(usage.get("outputTokens", 0) or 0)

    # Save per-run JSON log
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    log_file = LOG_DIR / f"{run_id}_llm_responses_{safe_model}_parts_{START_PART}-{END_PART}.json"
    with log_file.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    # Append token usage summary
    append_token_row(TOKENS_CSV, {
        "run_id": run_id,
        "timestamp": timestamp_iso(),
        "model": model_id,
        "num_examples": len(data_rows),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "labels_csv": str(OUTPUT_FILE),
        "log_json": str(log_file),
    })

    print(f"Appended labels to: {OUTPUT_FILE}")
    print(f"Saved logs: {log_file}")
    print(f"Token usage appended to: {TOKENS_CSV}")
