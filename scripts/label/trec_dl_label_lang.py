#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

# ============================
# Async-per-file, serial-per-row (in a background thread per file)
# ============================

cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)

# --- allow very large "passage" cells (fixes _csv.Error: field larger than field limit (131072)) ---
def _bump_field_limit():
    limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
    while limit >= 131072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2
_bump_field_limit()
# ---------------------------------------------------------------------------------------------------

# ----------------------------
# Configurable Paths
# ----------------------------
PROMPT_NAME   = "utility"
PROMPT_FILE   = Path(f"prompts/{PROMPT_NAME}.txt")
LLM_COST_CSV  = Path("scripts/report/llm_cost.csv")  # csv with columns: llm,input,output
LANG = "fr"  # use 'eng' to point to judged/original folder per logic below
# >>> Choose which parts to process (inclusive) <<<
START_PART    = 40
END_PART      = 45
TREC_DL_YEAR  = "2023"

# Where the part files live & their filename pattern
if LANG != "eng":
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{LANG}/")
else:
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
PART_PATTERN  = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

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
                "estimated_cost_usd",
                "labels_csv","log_json",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def iter_part_files(start_part: int, end_part: int):
    for n in range(start_part, end_part + 1):
        p = PART_DIR / PART_PATTERN.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")

def _header_cols(lang_out_name: str):
    # Output header uses pid and includes both passages
    # lang_out_name will be "passage_<LANG>"
    return ["pid", "docid", "passage", lang_out_name, "relevance"]

def ensure_combined_header(path: Path, lang_out_name: str):
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as csvfile:
            csv.writer(csvfile).writerow(_header_cols(lang_out_name))

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
    return "".join(ch if (ch.isalnum() or ch == "-") else "-" for ch in s).strip("-")

def extract_text_from_resp(model_id: str, resp: dict) -> str:
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        else:
            return resp["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""

def usage_from_resp(resp: dict) -> tuple[int, int]:
    usage = resp.get("usage", {}) or {}
    return int(usage.get("inputTokens", 0) or 0), int(usage.get("outputTokens", 0) or 0)

def load_model_prices(csv_path: Path) -> Dict[str, tuple[float, float]]:
    prices: Dict[str, tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row["llm"] or "").strip().strip('"').strip("'")
            pin  = float((row["input"] or "0").strip())
            pout = float((row["output"] or "0").strip())
            prices[name] = (pin, pout)
    return prices

def estimate_run_cost(model: str, tin: int, tout: int, csv_path: Path) -> float:
    prices = load_model_prices(csv_path)
    if model not in prices:
        raise KeyError(f"Model '{model}' not found in {csv_path}")
    pin, pout = prices[model]
    return (tin * pin + tout * pout) / 1000.0

# ----------------------------
# CSV & header helpers
# ----------------------------
def read_rows_stream(path: Path):
    f = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(f, skipinitialspace=True)
    try:
        for row in reader:
            yield row
    finally:
        f.close()

def count_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)  # minus header

def _clean_key(k: Optional[str]) -> str:
    return (k or "").lstrip("\ufeff").strip()

def _inspect_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return [_clean_key(k) for k in (reader.fieldnames or [])]

def _choose_key(header: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in header:
            return c
    return None

def _require_any_key(path: Path, header: List[str], label: str, candidates: List[str]) -> str:
    key = _choose_key(header, candidates)
    if not key:
        msg = (
            f"[FATAL] {path.name}: missing required column for {label}. "
            f"Looked for any of: {', '.join(candidates)}\n"
            f"Header columns: {header}"
        )
        print(msg)
        sys.exit(2)
    return key

# ----------------------------
# Stop key listener (press Q to stop)
# ----------------------------
def start_stop_key_listener(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> threading.Thread:
    def _listen():
        try:
            import msvcrt  # Windows
            print("[STOP] Press 'Q' to stop gracefully.")
            while not stop_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch and ch.lower() == 'q':
                        loop.call_soon_threadsafe(stop_event.set)
                        break
        except ImportError:
            print("[STOP] Type 'Q' + Enter to stop gracefully.")
            while not stop_event.is_set():
                try:
                    line = sys.stdin.readline()
                except Exception:
                    break
                if not line:
                    break
                if line.strip().lower() == 'q':
                    loop.call_soon_threadsafe(stop_event.set)
                    break

    t = threading.Thread(target=_listen, name="stop-key-listener", daemon=True)
    t.start()
    return t

# ----------------------------
# Blocking per-file worker (runs in a thread)
# ----------------------------
def _label_single_part_file_blocking(
    part_csv: Path,
    model_id: str,
    prompt_template: str,
    run_id: str,
    per_file_out_dir: Path,
    logs_dir: Path,
    stop_event: Optional[asyncio.Event] = None,
) -> dict:
    """
    Process one part file (serial per row) in a blocking manner.
    Returns dict with: part, rows, input_tokens, output_tokens, labels_csv, log_json
    """
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    per_file_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Validate header & concretize keys (fail fast)
    header = _inspect_header(part_csv)
    if "passage" not in header:
        print(f"[FATAL] {part_csv.name}: missing required column 'passage'. Header={header}")
        sys.exit(2)

    pid_candidates  = ["pid", "pid_resolved", "pid_qrels"]
    lang_candidates = [f"passage_{LANG}", f"passage_{LANG}_injected", "passage_injected"]

    pid_key          = _require_any_key(part_csv, header, "pid", pid_candidates)
    passage_lang_key = _require_any_key(part_csv, header, f"passage_{LANG}", lang_candidates)

    # Output column name for the localized passage is standardized as "passage_<LANG>"
    lang_out_col = f"passage_{LANG}"

    labels_path = per_file_out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    log_path    = logs_dir / f"{run_id}_llm_responses_{safe_model}_{part_csv.stem}.json"

    if not labels_path.exists():
        with labels_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(_header_cols(lang_out_col))

    bedrock = boto3.client("bedrock-runtime", config=cfg)

    total_in = 0
    total_out = 0
    logs: List[Dict[str, Any]] = []

    total_rows = count_data_rows(part_csv)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] Using pid='{pid_key}', localized passage='{passage_lang_key}' -> output column '{lang_out_col}'")

    for idx, row in enumerate(read_rows_stream(part_csv), start=1):
        # stop check
        if stop_event is not None and stop_event.is_set():
            print(f"\n[STOP] Requested. Halting file early: {part_csv.name}")
            break

        # --- Inputs from row (strict, fail if missing) ---
        pid = (row.get(pid_key, "") or "").strip()
        if not pid:
            print(f"[FATAL] {part_csv.name}: empty pid at row {idx} using column '{pid_key}'.")
            sys.exit(3)

        docid = (row.get("docid", "") or "").strip()
        if not docid:
            docid = pid  # fallback to pid if explicit docid not present

        query = (row.get("query", "") or "").strip()

        passage_orig = (row.get("passage", "") or "").strip()
        if not passage_orig:
            print(f"[FATAL] {part_csv.name}: empty 'passage' at row {idx}.")
            sys.exit(3)

        passage_lang = (row.get(passage_lang_key, "") or "").strip()
        if not passage_lang:
            print(f"[FATAL] {part_csv.name}: empty '{passage_lang_key}' at row {idx}.")
            sys.exit(3)

        # Prompt uses localized passage (required)
        prompt = prompt_template.format(query=query, passage=passage_lang)
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {"modelId": model_id, "messages": messages, "inferenceConfig": INFERENCE_CONFIG}

        try:
            resp = bedrock.converse(**kwargs)
        except KeyboardInterrupt:
            print(f"[INTERRUPTED] {part_csv.name}: Last pid {pid} (row {idx}) — stopping file early.")
            break
        except Exception as api_err:
            print(f"[ERROR] {part_csv.name}: API failed on pid={pid}, docid={docid} (row {idx}) :: {api_err}")
            with labels_path.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([pid, docid, passage_orig, passage_lang, ""])
            logs.append({
                "pid": pid, "docid": docid, "query": query,
                "prompt": prompt,
                "response_text": "",
                "full_response": {"error": str(api_err)},
                "passage": passage_orig,
                lang_out_col: passage_lang
            })
            continue

        # parse response
        try:
            text = extract_text_from_resp(model_id, resp)
        except Exception:
            text = ""
        score = parse_llm_text_to_score(text)

        # write labeled row (pid-first, and both passages recorded)
        with labels_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([pid, docid, passage_orig, passage_lang, score])

        in_tok, out_tok = usage_from_resp(resp)
        total_in  += in_tok
        total_out += out_tok

        logs.append({
            "pid": pid,
            "docid": docid,
            "query": query,
            "prompt": prompt,
            "response_text": text,
            "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
            "full_response": resp,
            "passage": passage_orig,
            lang_out_col: passage_lang,
        })

        print(f"[{part_csv.name}] [{idx}/{total_rows}]  tokens in/out += {in_tok}/{out_tok} "
              f"(totals {total_in}/{total_out})", end="\r", flush=True)

    # save log
    with log_path.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    print()  # newline after progress
    print(f"[{part_csv.name}] Wrote labels: {labels_path.name} | Log: {log_path.name} "
          f"| tokens in/out={total_in}/{total_out}")

    return {
        "part": part_csv.name,
        "rows": total_rows,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "labels_csv": str(labels_path),
        "log_json": str(log_path),
        "lang_out_col": lang_out_col,
    }

# Thin async wrapper that runs the blocking worker in a thread
async def label_single_part_file(
    part_csv: Path,
    model_id: str,
    prompt_template: str,
    run_id: str,
    per_file_out_dir: Path,
    logs_dir: Path,
    stop_event: asyncio.Event,
) -> dict:
    return await asyncio.to_thread(
        _label_single_part_file_blocking,
        part_csv, model_id, prompt_template, run_id, per_file_out_dir, logs_dir, stop_event
    )

# ----------------------------
# Combine per-file labels into the shared per-model OUTPUT_FILE
# ----------------------------
def merge_labels(per_file_labels: List[str], combined_out: Path, lang_out_name: str, stop_event: Optional[asyncio.Event] = None):
    ensure_combined_header(combined_out, lang_out_name)
    appended = 0
    with combined_out.open("a", encoding="utf-8", newline="") as out_f:
        out_writer = csv.writer(out_f)
        for path_str in per_file_labels:
            if stop_event is not None and stop_event.is_set():
                print("[STOP] Merge halted early by user.")
                break
            p = Path(path_str)
            if not p.exists():
                print(f"[WARN] Missing per-file labels for merge: {p}")
                continue
            with p.open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.reader(in_f)
                _ = next(reader, None)  # skip header
                for row in reader:
                    out_writer.writerow(row)
                    appended += 1
    print(f"[MERGE] Appended {appended} labeled rows into: {combined_out}")

# ----------------------------
# Orchestration
# ----------------------------
async def run_for_model(model_id: str, stop_event: asyncio.Event):
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    # --- Per-model directories / files ---
    short = model_short_name(model_id)            # e.g., "claude-3-5" or "gpt-oss-20b"
    MODEL_OUT_DIR  = Path("outputs/llm_label") / short
    MODEL_LOGS_DIR = Path("logs") / short
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Combined output CSV & token usage CSV for this model
    if LANG == "eng":
        output_file = MODEL_OUT_DIR / f"{short}_trec_dl_{TREC_DL_YEAR}_raw.csv"
    else:
        output_file = MODEL_OUT_DIR / f"{short}_trec_dl_{TREC_DL_YEAR}_{LANG}_raw.csv"

    tokens_csv  = MODEL_OUT_DIR / "token_usage.csv"

    # Build the list of part files to process
    part_files = list(iter_part_files(START_PART, END_PART))
    if not part_files:
        print("[INFO] No part files found in range.")
        return

    run_id = timestamp_id()
    print(f"\n--- Running inference for model: {model_id} (run_id={run_id}) ---")
    print("[STOP] Press 'Q' at any time to stop after the current in-flight items.")

    # Temp per-file labels live under the model's output folder
    per_file_out_dir = MODEL_OUT_DIR / f"_tmp_{run_id}_{model_id.replace(':','_')}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    # Limit parallelism across files to reduce throttling
    max_concurrent_files = min(6, len(part_files))
    sem = asyncio.Semaphore(max_concurrent_files)

    results: List[Dict[str, Any]] = []

    async def sem_task(part_csv: Path):
        async with sem:
            if stop_event.is_set():
                return None
            return await label_single_part_file(
                part_csv, model_id, prompt_template, run_id, per_file_out_dir, MODEL_LOGS_DIR, stop_event
            )

    # Create tasks: one per file
    tasks = [asyncio.create_task(sem_task(p)) for p in part_files]

    # Consume tasks, honoring stop_event (cancel remaining if requested)
    for task in asyncio.as_completed(tasks):
        if stop_event.is_set():
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print("[STOP] Cancelled remaining files.")
            break
        res = await task
        if res:
            results.append(res)

    if results and not stop_event.is_set():
        # Ensure all per-file outputs used the same lang column name (they should)
        lang_out_names = {r["lang_out_col"] for r in results}
        if len(lang_out_names) != 1:
            print(f"[FATAL] Inconsistent localized output column names across files: {lang_out_names}")
            sys.exit(4)
        lang_out_name = next(iter(lang_out_names))

        # Merge per-file label CSVs into the per-model combined OUTPUT_FILE
        per_file_labels = [r["labels_csv"] for r in results]
        await asyncio.to_thread(merge_labels, per_file_labels, output_file, lang_out_name, stop_event)

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
    combined_log_index = MODEL_LOGS_DIR / f"{run_id}_llm_logs_index_{safe_model}.json"
    with combined_log_index.open("w", encoding="utf-8") as f:
        json.dump([{"part": r["part"], "log_json": r["log_json"]} for r in results],
                  f, indent=2, ensure_ascii=False)

    append_token_row(tokens_csv, {
        "run_id": run_id,
        "timestamp": timestamp_iso(),
        "model": model_id,
        "num_examples": num_rows,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "estimated_cost_usd": f"{cost_usd:.6f}",
        "labels_csv": str(output_file),
        "log_json": str(combined_log_index),
    })

    print(f"[DONE] Model: {model_id} | Labeled rows: {num_rows}")
    print(f"[TOKENS] in={total_in:,}  out={total_out:,}  total={total_in + total_out:,}")
    print(f"[COST]   from {LLM_COST_CSV.name} -> ${cost_usd:,.6f} USD")

    # --- Clean up temp per-file labels directory ---
    try:
        shutil.rmtree(per_file_out_dir, ignore_errors=False)
        print(f"[CLEANUP] Removed temp folder: {per_file_out_dir}")
    except Exception as e:
        print(f"[WARN] Failed to remove temp folder {per_file_out_dir}: {e}")

async def main():
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    listener_thread = start_stop_key_listener(loop, stop_event)

    try:
        for model_id in MODELS:
            if stop_event.is_set():
                print("[STOP] Skipping remaining models.")
                break
            await run_for_model(model_id, stop_event)
    finally:
        stop_event.set()
        try:
            listener_thread.join(timeout=0.2)
        except Exception:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Top-level stop.")
