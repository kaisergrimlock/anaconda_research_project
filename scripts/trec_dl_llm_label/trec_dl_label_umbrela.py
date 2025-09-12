import boto3
import json
import csv
from pathlib import Path
from datetime import datetime
import re

# ----------------------------
# Configurable Paths
# ----------------------------
PROMPT_FILE = Path("prompts/umbrela.txt")
INPUT_CSV   = Path("outputs/trec_dl/combined_result_translated_vi_20.csv")

OUTPUT_DIR  = Path("outputs/trec_dl_llm_label/translated/viet/umbrela")  # CSV outputs per run/model
LOG_DIR     = Path("outputs/trec_dl/logs")                  # JSON logs
TOKENS_CSV  = Path("outputs/trec_dl_llm_label/token_usage.csv")  # cumulative token usage log

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Bedrock Config
# ----------------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

MODELS = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "openai.gpt-oss-20b-1:0",
    # "anthropic.claude-3-5-sonnet-20240620-v1:0",
]

INFERENCE_CONFIG = {
    "maxTokens": 1000,
    "temperature": 0.0,
    "topP": 1.0,
}

# ----------------------------
# Utility
# ----------------------------
def timestamp_id() -> str:
    """Return current timestamp as YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def timestamp_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def append_token_row(tokens_csv: Path, row: dict) -> None:
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

def collect_text_from_bedrock(resp: dict) -> str:
    """
    Concatenate text-like payloads from Bedrock Converse response.
    Handles entries with 'text' and 'reasoningContent' (Anthropic-style).
    """
    content = resp.get("output", {}).get("message", {}).get("content", []) or []
    chunks = []

    for item in content:
        # Direct text chunk
        t = item.get("text")
        if t:
            chunks.append(t)

        # Reasoning block (provider-specific)
        rc = item.get("reasoningContent")
        if isinstance(rc, dict):
            rt = rc.get("reasoningText")
            if isinstance(rt, dict):
                rtxt = rt.get("text")
                if rtxt:
                    chunks.append(rtxt)

    return "\n".join(chunks).strip()

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_FINAL_SCORE_RE = re.compile(r"##\s*final\s*score\s*:\s*([0-3])\b", re.IGNORECASE)
_O_FIELD_RE = re.compile(r'"?O"?\s*[:=]\s*([0-3])\b')

def extract_o_score_from_text(text: str) -> str:
    """
    Extract the 'O' (overall) score as a string in {'0','1','2','3'}.
    Tries, in order:
      1) Parse whole text as JSON and read 'O'
      2) Find first JSON object substring and parse 'O'
      3) Regex for '##final score: N'
      4) Loose regex for O field like: "O": 3 or O=3
    Returns '' if not found.
    """
    if not text:
        return ""

    # 1) Whole text is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            val = obj.get("O")
            if val is not None:
                s = str(val).strip()
                if s in {"0", "1", "2", "3"}:
                    return s
    except Exception:
        pass

    # 2) Try to pull the first JSON object substring
    try:
        m = _JSON_OBJ_RE.search(text)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                val = obj.get("O")
                if val is not None:
                    s = str(val).strip()
                    if s in {"0", "1", "2", "3"}:
                        return s
    except Exception:
        pass

    # 3) Regex for "##final score: N"
    m2 = _FINAL_SCORE_RE.search(text)
    if m2:
        return m2.group(1)

    # 4) Loose "O": N, O=N, etc.
    m3 = _O_FIELD_RE.search(text)
    if m3:
        return m3.group(1)

    return ""

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
        query = row.get("query", "")
        docid = row.get("docid", "")
        passage_text = (row.get("passage") or "").strip()

        # Prepare prompt
        prompt = prompt_template.format(query=query, passage=passage_text)

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": INFERENCE_CONFIG,
        }

        # Call Bedrock API
        try:
            resp = bedrock.converse(**kwargs)
        except Exception as e:
            # Log the error and continue
            text = ""
            llm_score = ""
            logs.append({
                "query": query,
                "docid": docid,
                "prompt": prompt,
                "response_text": text,
                "full_response": {"error": str(e)},
            })
            results.append([query, docid, passage_text, llm_score])
            continue

        # Extract response text (robust)
        text = collect_text_from_bedrock(resp)

        # Extract the 'O' numeric score
        llm_score = extract_o_score_from_text(text)

        results.append([query, docid, passage_text, llm_score])

        logs.append({
            "query": query,
            "docid": docid,
            "prompt": prompt,
            "response_text": text,
            "full_response": resp,
        })

        usage = resp.get("usage", {}) or {}
        total_input_tokens  += int(usage.get("inputTokens", 0) or 0)
        total_output_tokens += int(usage.get("outputTokens", 0) or 0)

    # ----------------------------
    # Write combined CSV per model
    # ----------------------------
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    out_csv = OUTPUT_DIR / f"{run_id}_llm_labels_{safe_model}_top2.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query", "docid", "passage", "relevance"])
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
