from time import strftime
import asyncio
import json
import csv
from pathlib import Path
from datetime import datetime

import boto3
from botocore.config import Config

# ============================
# Async-per-file, serial-per-row
# ============================

cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,      # handshake/socket connect
    read_timeout=300,        # wait up to 5 min for a response
    retries={"max_attempts": 8, "mode": "standard"},
)

# ----------------------------
# Configurable Paths
# ----------------------------
PROMPT_NAME = "prompt"
PROMPT_FILE = Path(f"prompts/{PROMPT_NAME}.txt")

# >>> Choose which parts to process (inclusive) <<<
START_PART = 1
END_PART   = 19

# Where the part files live & their filename pattern
PART_DIR     = Path("retrieved/trec_dl_2019/judged")
PART_PATTERN = "all_topics_trecdl_2019_part{n}.csv"

# Shared combined labels file
OUTPUT_FILE = Path("outputs/llm_label/trec_dl_2019_raw.csv")

LOG_DIR    = Path("logs")
TOKENS_CSV = Path("token_usage.csv")

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

def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]

def ensure_combined_header(path: Path):
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["query", "docid", "passage", "relevance"])

def parse_llm_text_to_score(text: str) -> str:
    """Expect model returns JSON like {"O": <label>}."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return str(parsed.get("O", ""))
    except Exception:
        pass
    return ""

def extract_text_from_resp(model_id: str, resp: dict) -> str:
    # Bedrock Converse response shape:
    # resp["output"]["message"]["content"] is a list of blocks
    # Your original code kept a model-specific index. Keep that for compatibility.
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        else:
            return resp["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""

def usage_from_resp(resp: dict):
    usage = resp.get("usage", {}) or {}
    return int(usage.get("inputTokens", 0) or 0), int(usage.get("outputTokens", 0) or 0)

# ----------------------------
# Per-file (part) worker
# ----------------------------
async def label_single_part_file(
    part_csv: Path,
    model_id: str,
    prompt_template: str,
    run_id: str,
    per_file_out_dir: Path,
) -> dict:
    """
    Process one part file (serial per row). Returns dict with:
      - 'part': file name
      - 'rows': number of rows processed
      - 'input_tokens', 'output_tokens'
      - 'labels_csv' (path)
      - 'log_json' (path)
    """
    rows = await asyncio.to_thread(read_rows, part_csv)
    print(f"[{part_csv.name}] Loaded {len(rows)} rows")

    # Each file writes its own labels CSV and log
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = per_file_out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    log_path    = LOG_DIR / f"{run_id}_llm_responses_{safe_model}_{part_csv.stem}.json"

    # Write header for per-file labels
    if not labels_path.exists():
        with labels_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["query", "docid", "passage", "relevance"])

    # Create a bedrock client per task (safer across threads)
    bedrock = boto3.client("bedrock-runtime", config=cfg)

    total_in = 0
    total_out = 0
    logs = []

    # Process serially within this file
    for idx, row in enumerate(rows, start=1):
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

        # Call sync SDK on a worker thread to avoid blocking the loop
        try:
            resp = await asyncio.to_thread(bedrock.converse, **kwargs)
        except KeyboardInterrupt:
            print(f"[INTERRUPTED] {part_csv.name}: Last doc {docid} (row {idx}) — stopping file early.")
            break
        except Exception as api_err:
            print(f"[ERROR] {part_csv.name}: API failed on docid={docid} (row {idx}) :: {api_err}")
            # Append blank relevance so downstream joins still work
            with labels_path.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([query, docid, passage_text, ""])
            logs.append({
                "query": query, "docid": docid, "prompt": prompt,
                "response_text": "", "full_response": {"error": str(api_err)}
            })
            continue

        text = extract_text_from_resp(model_id, resp)
        score = parse_llm_text_to_score(text)

        # Append one labeled row immediately
        with labels_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([query, docid, passage_text, score])

        logs.append({"query": query, "docid": docid, "prompt": prompt,
                     "response_text": text, "full_response": resp})

        in_tok, out_tok = usage_from_resp(resp)
        total_in  += in_tok
        total_out += out_tok

    # Save per-file JSON log
    with log_path.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    print(f"[{part_csv.name}] Wrote labels: {labels_path.name} | Log: {log_path.name} | tokens in/out={total_in}/{total_out}")

    return {
        "part": part_csv.name,
        "rows": len(rows),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "labels_csv": str(labels_path),
        "log_json": str(log_path),
    }

# ----------------------------
# Combine per-file labels into the shared OUTPUT_FILE
# ----------------------------
def merge_labels(per_file_labels: list[str], combined_out: Path):
    ensure_combined_header(combined_out)
    appended = 0
    with combined_out.open("a", encoding="utf-8", newline="") as out_f:
        out_writer = csv.writer(out_f)
        for path_str in per_file_labels:
            p = Path(path_str)
            if not p.exists():
                print(f"[WARN] Missing per-file labels for merge: {p}")
                continue
            with p.open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)  # skip header
                for row in reader:
                    out_writer.writerow(row)
                    appended += 1
    print(f"[MERGE] Appended {appended} labeled rows into: {combined_out}")

# ----------------------------
# Orchestration
# ----------------------------
async def run_for_model(model_id: str):
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    # Build the list of part files to process
    part_files = list(iter_part_files(START_PART, END_PART))
    if not part_files:
        print("[INFO] No part files found in range.")
        return

    run_id = timestamp_id()
    print(f"\n--- Running inference for model: {model_id} (run_id={run_id}) ---")

    # Directory to hold per-file outputs before merge
    per_file_out_dir = OUTPUT_FILE.parent / f"_tmp_{run_id}_{model_id.replace(':','_')}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    # Limit parallelism if desired (e.g., avoid throttling). Tune this:
    max_concurrent_files = min(6, len(part_files))  # change as needed
    sem = asyncio.Semaphore(max_concurrent_files)

    results = []

    async def sem_task(part_csv: Path):
        async with sem:
            return await label_single_part_file(
                part_csv, model_id, prompt_template, run_id, per_file_out_dir
            )

    # Create tasks: one per file
    tasks = [asyncio.create_task(sem_task(p)) for p in part_files]
    for task in asyncio.as_completed(tasks):
        res = await task
        if res:
            results.append(res)

    # Merge per-file label CSVs into the shared OUTPUT_FILE
    per_file_labels = [r["labels_csv"] for r in results]
    await asyncio.to_thread(merge_labels, per_file_labels, OUTPUT_FILE)

    # Aggregate token usage and write a single row
    total_in  = sum(r["input_tokens"]  for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    num_rows  = sum(r["rows"] for r in results)

    # For reproducibility, also write a combined log index (list of per-file logs)
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    combined_log_index = LOG_DIR / f"{run_id}_llm_logs_index_{safe_model}.json"
    with combined_log_index.open("w", encoding="utf-8") as f:
        json.dump([{"part": r["part"], "log_json": r["log_json"]} for r in results],
                  f, indent=2, ensure_ascii=False)

    append_token_row(TOKENS_CSV, {
        "run_id": run_id,
        "timestamp": timestamp_iso(),
        "model": model_id,
        "num_examples": num_rows,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "labels_csv": str(OUTPUT_FILE),
        "log_json": str(combined_log_index),
    })

    print(f"[DONE] Model: {model_id} | Appended to: {OUTPUT_FILE}")
    print(f"[DONE] Token usage appended to: {TOKENS_CSV}")

async def main():
    # Run models one-by-one; within each, files run in parallel
    for model_id in MODELS:
        await run_for_model(model_id)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INTERRUPTED] Top-level stop.")
