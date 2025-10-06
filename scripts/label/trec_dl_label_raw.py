#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

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
PROMPT_NAME  = "utility"
PROMPT_FILE  = Path(f"prompts/{PROMPT_NAME}.txt")
LLM_COST_CSV = Path("scripts/report/llm_cost.csv")  # csv with columns: llm,input,output

# >>> Choose which parts to process (inclusive) <<<
START_PART = 1
END_PART   = 19
TREC_DL_YEAR = "2023"  # '2019', '2020', or '2023'

# Where the part files live & their filename pattern
PART_DIR     = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
PART_PATTERN = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

# Shared combined labels file
OUTPUT_FILE = Path(f"outputs/llm_label/trec_dl_{TREC_DL_YEAR}_raw.csv")

LOG_DIR    = Path("logs")
TOKENS_CSV = Path("token_usage.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Bedrock / model config
# ----------------------------
MODELS = [
    "anthropic.claude-3-5-haiku-20241022-v1:0",
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
    """
    Appends a run summary row. Adds an 'estimated_cost_usd' column.
    """
    file_exists = tokens_csv.exists()
    with tokens_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id","timestamp","model","num_examples",
                "input_tokens","output_tokens","total_tokens",
                "estimated_cost_usd",
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


def model_short_name(model_id: str) -> str:
    """
    'anthropic.claude-3-5-haiku-20241022-v1:0' -> 'claude-3-5'
    Rule: drop provider (before first '.'), strip version (after ':'), keep first 3 '-' parts.
    """
    s = model_id
    if "." in s:
        s = s.split(".", 1)[1]
    if ":" in s:
        s = s.split(":", 1)[0]
    parts = s.split("-")
    s = "-".join(parts[:3])
    # sanitize
    return "".join(ch if (ch.isalnum() or ch == "-") else "-" for ch in s).strip("-")
def iter_part_files_in(part_dir: Path, start_part: int, end_part: int, pattern: str):
    for n in range(start_part, end_part + 1):
        p = part_dir / pattern.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")


def extract_text_from_resp(model_id: str, resp: dict) -> str:
    """
    Bedrock Converse response shape:
      resp["output"]["message"]["content"] -> list of blocks (with "text")
    Your OpenAI-compat model had content at index 1; others at 0.
    """
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        else:
            return resp["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""

def usage_from_resp(resp: dict) -> tuple[int, int]:
    usage = resp.get("usage", {}) or {}
    # Bedrock uses inputTokens/outputTokens
    return int(usage.get("inputTokens", 0) or 0), int(usage.get("outputTokens", 0) or 0)

def load_model_prices(csv_path: Path) -> dict[str, tuple[float, float]]:
    """
    Read a CSV with header: llm,input,output
    Prices are per 1K tokens. Returns {model: (input_price, output_price)}.
    """
    prices: dict[str, tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row["llm"] or "").strip().strip('"').strip("'")
            pin  = float((row["input"] or "0").strip())
            pout = float((row["output"] or "0").strip())
            prices[name] = (pin, pout)
    return prices

def estimate_run_cost(model: str, tin: int, tout: int, csv_path: Path) -> float:
    """
    Cost = (tin * input_price + tout * output_price) / 1000
    """
    prices = load_model_prices(csv_path)
    if model not in prices:
        raise KeyError(f"Model '{model}' not found in {csv_path}")
    pin, pout = prices[model]
    return (tin * pin + tout * pout) / 1000.0

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
    total_rows = len(rows)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")

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
    logs: list[dict[str, Any]] = []

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

        in_tok, out_tok = usage_from_resp(resp)
        total_in  += in_tok
        total_out += out_tok

        logs.append({
            "query": query,
            "docid": docid,
            "prompt": prompt,
            "response_text": text,
            "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
            "full_response": resp
        })

        # Progress line
        print(f"[{part_csv.name}] [{idx}/{total_rows}]  tokens in/out += {in_tok}/{out_tok} (totals {total_in}/{total_out})", end="\r", flush=True)

    # Save per-file JSON log
    with log_path.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    print()  # newline after progress
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

    # Limit parallelism across files to reduce throttling
    max_concurrent_files = min(6, len(part_files))
    sem = asyncio.Semaphore(max_concurrent_files)

    results: list[dict[str, Any]] = []

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

    # Aggregate token usage and compute cost
    total_in  = sum(r["input_tokens"]  for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    num_rows  = sum(r["rows"] for r in results)

    # Cost from CSV (per 1K tokens)
    try:
        cost_usd = estimate_run_cost(model_id, total_in, total_out, LLM_COST_CSV)
    except Exception as e:
        cost_usd = 0.0
        print(f"[WARN] Could not compute cost from {LLM_COST_CSV}: {e}")

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
        "estimated_cost_usd": f"{cost_usd:.6f}",
        "labels_csv": str(OUTPUT_FILE),
        "log_json": str(combined_log_index),
    })

    print(f"[DONE] Model: {model_id} | Labeled rows: {num_rows}")
    print(f"[TOKENS] in={total_in:,}  out={total_out:,}  total={total_in + total_out:,}")
    print(f"[COST]   from {LLM_COST_CSV.name} -> ${cost_usd:,.6f} USD")
    print(f"[DONE] Token usage appended to: {TOKENS_CSV}")

    # --- Clean up temp per-file labels directory ---
    try:
        shutil.rmtree(per_file_out_dir, ignore_errors=False)
        print(f"[CLEANUP] Removed temp folder: {per_file_out_dir}")
    except Exception as e:
        print(f"[WARN] Failed to remove temp folder {per_file_out_dir}: {e}")

async def main():
    # Run models one-by-one; within each, files run in parallel
    for model_id in MODELS:
        await run_for_model(model_id)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INTERRUPTED] Top-level stop.")
