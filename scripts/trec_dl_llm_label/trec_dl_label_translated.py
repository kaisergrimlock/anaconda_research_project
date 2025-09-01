import math
import boto3
import json
import csv
import random
from pathlib import Path
from datetime import datetime

# ----------------------------
# Configurable Paths
# ----------------------------
PROMPT_FILE = Path("prompt.txt")
INPUT_CSV   = Path("outputs/trec_dl/combined_result_translated_20.csv")

OUTPUT_DIR  = Path("outputs/trec_dl_llm_label/translated")  # CSV outputs per run/model
LOG_DIR     = Path("outputs/trec_dl/logs")       # JSON logs
TOKENS_CSV  = Path("outputs/trec_dl_llm_label/token_usage.csv")  # cumulative token usage log

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# AWS Clients
# ----------------------------
# Bedrock (LLM scoring)
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

# Amazon Translate (query -> Vietnamese)
translate = boto3.client("translate", region_name="ap-southeast-2")

# ----------------------------
# Models / Inference
# ----------------------------
MODELS = [
    #"anthropic.claude-3-haiku-20240307-v1:0",
    #"mistral.mixtral-8x7b-instruct-v0:1",
    "openai.gpt-oss-20b-1:0"
]

INFERENCE_CONFIG = {
    "maxTokens": 1000,
    "temperature": 0.0,
    "topP": 1.0,
}

# ----------------------------
# Injection helpers
# ----------------------------
RNG = random.Random(42)   # set to None for non-deterministic
INJECT_COUNT = 1          # how many times to inject query_vi into passage
INJECT_PROB  = 1.0        # probability per planned injection (0..1)

def find_between_word_positions(text: str):
    """Return indices where inserting text keeps us between words (across whitespace runs)."""
    positions = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if i > 0 and j < n and not text[i-1].isspace() and not text[j].isspace():
                positions.append(j)
            i = j
        else:
            i += 1
    return positions

def inject_once(text: str, snippet: str) -> str:
    spots = find_between_word_positions(text)
    if not spots:
        return text
    idx = RNG.choice(spots)
    return text[:idx] + snippet + " " + text[idx:]

def inject_n(text: str, snippet: str, n: int, prob: float) -> str:
    out = text
    for _ in range(max(0, n)):
        if RNG.random() <= prob:
            out = inject_once(out, snippet)
    return out

# ----------------------------
# Utility
# ----------------------------
def round_half_up(n):
    try:
        return math.floor(float(n) + 0.5)
    except (TypeError, ValueError):
        return ""

def timestamp_id():
    """Return current timestamp as YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def timestamp_iso():
    return datetime.now().isoformat(timespec="seconds")

def append_token_row(tokens_csv: Path, row: dict):
    """Append a row to the token usage CSV, writing header if file doesn't exist."""
    file_exists = tokens_csv.exists()
    with tokens_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "timestamp",
                "model",
                "num_examples",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "labels_csv",
                "log_json",
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
        query = row["query"]
        docid = row["docid"]
        passage_text = row["passage"].strip()

        # (A) Translate the query to Vietnamese
        tr_resp = translate.translate_text(
            Text=query,
            SourceLanguageCode="auto",
            TargetLanguageCode="vi"
        )
        query_vi = tr_resp["TranslatedText"]

        # (B) Prepare prompt for relevance judging
        prompt = prompt_template.format(query=query, passage=passage_text)
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": INFERENCE_CONFIG,
        }

        # (C) Call Bedrock API to get relevance label
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

        # (D) Inject translated query into the passage
        passage_injected = inject_n(passage_text, query_vi, INJECT_COUNT, INJECT_PROB)

        # Collect output row (now includes query_vi & passage_injected)
        results.append([query, docid, passage_text, llm_score, query_vi, passage_injected])

        # Detailed logs
        logs.append({
            "query": query,
            "docid": docid,
            "query_vi": query_vi,
            "passage_injected": passage_injected,
            "prompt": prompt,
            "response_text": text,
            "full_response": resp
        })

        usage = resp.get("usage", {})
        total_input_tokens  += int(usage.get("inputTokens", 0) or 0)
        total_output_tokens += int(usage.get("outputTokens", 0) or 0)

    # ----------------------------
    # Write combined CSV per model
    # ----------------------------
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    out_csv = OUTPUT_DIR / f"{run_id}_llm_labels_{safe_model}_top2.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query", "docid", "passage", "relevance", "query_vi", "passage_injected"])
        for r in results:
            writer.writerow(r)

    # Write logs
    log_file = LOG_DIR / f"{run_id}_llm_responses_{safe_model}_top2.json"
    with log_file.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    # ----------------------------
    # Append token usage summary row
    # ----------------------------
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
