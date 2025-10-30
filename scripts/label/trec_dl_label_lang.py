#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import shutil
import sys
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

LANG          = "raw"   # "raw", "vi", "fr", ...
START_PART    = 0
END_PART      = 0
TREC_DL_YEAR  = "2023"

# Where the part files live & their filename pattern
if LANG == "raw":
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
else:
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{LANG}/")
PART_PATTERN  = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

# ----------------------------
# Bedrock / model config
# ----------------------------
MODELS = [
    "openai.gpt-oss-20b-1:0",
]

INFERENCE_CONFIG = {
    "maxTokens": 2000,
    "temperature": 0.0,
    "topP": 1.0,
}

# ----------------------------
# Output schema handling
# ----------------------------
def expected_extra_cols_for_lang(lang: str) -> List[str]:
    if lang == "raw":
        return []  # exactly the base six
    # language-specific columns (query_<lang>, passage_injected)
    return [f"query_{lang}", "passage_injected"]

def expected_base_cols() -> List[str]:
    return ["qid", "query", "pid_qrels", "pid_resolved", "passage", "relevance"]

def output_header_from_input(input_header: List[str]) -> List[str]:
    out = list(input_header)
    if "llm_relevance" not in out:
        out.append("llm_relevance")
    return out


# Upsert/Merge identity
KEY_COLS: Tuple[str, str, str] = ("pid_qrels", "pid_resolved", "passage")

def ensure_csv_with_header(path: Path, header: List[str]):
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

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

def parse_llm_text_to_score(text: str) -> str:
    """
    Accepts either {"O": <int>} or a list of dicts containing "O".
    Returns string score or "" on failure.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "O" in parsed:
            return str(parsed["O"])
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "O" in item:
                    return str(item["O"])
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
# CSV helpers
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

def _row_key_from_list(row: List[str], header: List[str], key_cols: Tuple[str, str, str]) -> Tuple[str, str, str]:
    idx = [header.index(c) for c in key_cols]
    return tuple(row[i] for i in idx)  # type: ignore[return-value]

def _read_csv_as_ordered_map(path: Path, header: List[str], key_cols: Tuple[str, str, str]) -> "OrderedDict[Tuple[str,str,str], List[str]]":
    """
    Load CSV (with known header) into an OrderedDict keyed by (pid_qrels, pid_resolved, passage).
    Preserves insertion order.
    """
    od: "OrderedDict[Tuple[str,str,str], List[str]]" = OrderedDict()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            if file_header is None:
                return od
            for row in reader:
                if not row:
                    continue
                k = _row_key_from_list(row, header, key_cols)
                od[k] = row
    return od

def _write_ordered_map_to_csv(path: Path, header: List[str], od: "OrderedDict[Tuple[str,str,str], List[str]]") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in od.values():
            w.writerow(row)
    tmp.replace(path)

def upsert_row_csv(path: Path, header: List[str], key_cols: Tuple[str, str, str], new_row: List[str]) -> None:
    """
    Insert/replace a row in CSV keyed by (pid_qrels, pid_resolved, passage).
    Creates the file with header if it doesn't exist.
    """
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)
    od = _read_csv_as_ordered_map(path, header, key_cols)
    k  = _row_key_from_list(new_row, header, key_cols)
    od[k] = new_row  # upsert
    _write_ordered_map_to_csv(path, header, od)

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
    Returns dict with: part, rows, input_tokens, output_tokens, labels_csv, log_json, header_out
    """
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    per_file_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Validate and capture input header
    header_in = _inspect_header(part_csv)
    base_needed = set(expected_base_cols())
    lang_needed = set(expected_extra_cols_for_lang(LANG))
    missing = [c for c in base_needed if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv.name}: missing required base columns {missing}. Header={header_in}")
        sys.exit(2)
    # extra columns for non-raw
    for c in lang_needed:
        if c not in header_in:
            print(f"[WARN] {part_csv.name}: expected language column '{c}' not found; will fall back if needed.")

    # Build output header: exact input order + llm_relevance
    header_out = output_header_from_input(header_in)

    labels_path = per_file_out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    log_path    = logs_dir / f"{run_id}_llm_responses_{safe_model}_{part_csv.stem}.json"

    ensure_csv_with_header(labels_path, header_out)

    bedrock = boto3.client("bedrock-runtime", config=cfg)

    total_in = 0
    total_out = 0
    logs: List[Dict[str, Any]] = []

    total_rows = count_data_rows(part_csv)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] LANG='{LANG}' | output columns = input columns + ['llm_relevance']")

    # Helpers to select prompt fields
    query_lang_col = f"query_{LANG}"
    def pick_query(row: Dict[str, str]) -> str:
        if LANG != "raw" and query_lang_col in row and row.get(query_lang_col):
            return (row.get(query_lang_col) or "").strip()
        return (row.get("query", "") or "").strip()

    def pick_passage_for_prompt(row: Dict[str, str]) -> str:
        if LANG != "raw" and "passage_injected" in row and row.get("passage_injected"):
            return (row.get("passage_injected") or "").strip()
        return (row.get("passage", "") or "").strip()

    for idx, row in enumerate(read_rows_stream(part_csv), start=1):
        # stop check
        if stop_event is not None and stop_event.is_set():
            print(f"\n[STOP] Requested. Halting file early: {part_csv.name}")
            break

        # Build a writable row map that mirrors input header exactly
        row_out_map: Dict[str, str] = {k: (row.get(k, "") or "") for k in header_in}

        # Stabilize pid_resolved with fallbacks if empty
        pr = (row_out_map.get("pid_resolved", "") or "").strip()
        if not pr:
            pr = (row.get("docid", "") or row.get("pid", "") or row.get("pid_qrels", "") or "").strip()
            row_out_map["pid_resolved"] = pr

        # Minimal sanity checks
        qid       = (row_out_map.get("qid", "") or "").strip()
        pid_qrels = (row_out_map.get("pid_qrels", "") or "").strip()
        passage   = (row_out_map.get("passage", "") or "").strip()
        if not (qid and pid_qrels and passage):
            print(f"[FATAL] {part_csv.name}: missing qid/pid_qrels/passage at row {idx}.")
            sys.exit(3)

        # Prompt compose
        q_for_prompt = pick_query(row_out_map)
        p_for_prompt = pick_passage_for_prompt(row_out_map)
        prompt = prompt_template.format(query=q_for_prompt, passage=p_for_prompt)
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs   = {"modelId": model_id, "messages": messages, "inferenceConfig": INFERENCE_CONFIG}

        # Call LLM
        text = ""
        score = ""
        in_tok = out_tok = 0
        try:
            resp  = bedrock.converse(**kwargs)
            text  = extract_text_from_resp(model_id, resp) or ""
            score = parse_llm_text_to_score(text)  # llm_relevance
            in_tok, out_tok = usage_from_resp(resp)
            total_in  += in_tok
            total_out += out_tok
        except KeyboardInterrupt:
            print(f"[INTERRUPTED] {part_csv.name}: Last qid {qid} (row {idx}) — stopping file early.")
            break
        except Exception as api_err:
            print(f"[ERROR] {part_csv.name}: API failed on qid={qid}, pid_resolved={pr} (row {idx}) :: {api_err}")
            # score remains blank

        # Construct output row in exact input order + llm_relevance
        row_out = [row_out_map.get(col, "") for col in header_in] + [score]

        # Upsert using identity (pid_qrels, pid_resolved, passage)
        upsert_row_csv(
            labels_path,
            header_out,
            KEY_COLS,
            row_out
        )

        # log
        logs.append({
            "qid": qid,
            "pid_qrels": pid_qrels,
            "pid_resolved": pr,
            "prompt": prompt,
            "response_text": text,
            "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
            "passage_prompt_used": "passage_injected" if (LANG != "raw" and row_out_map.get("passage_injected")) else "passage",
            "query_prompt_used": query_lang_col if (LANG != "raw" and row_out_map.get(query_lang_col)) else "query",
            "llm_relevance": score,
        })

        print(f"[{part_csv.name}] [{idx}/{total_rows}]  tokens in/out += {in_tok}/{out_tok} "
              f"(totals {total_in}/{total_out})", end="\r", flush=True)

    # save log
    with (logs_dir / f"{run_id}_llm_responses_{safe_model}_{part_csv.stem}.json").open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    print()  # newline after progress
    print(f"[{part_csv.name}] Wrote labels: {labels_path.name} | tokens in/out={total_in}/{total_out}")

    return {
        "part": part_csv.name,
        "rows": total_rows,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "labels_csv": str(labels_path),
        "log_json": str((logs_dir / f"{run_id}_llm_responses_{safe_model}_{part_csv.stem}.json")),
        "header_out": header_out,
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
def merge_labels(per_file_labels: List[str], combined_out: Path, header_out: List[str], stop_event: Optional[asyncio.Event] = None):
    ensure_csv_with_header(combined_out, header_out)

    # Load existing combined into ordered map (preserve existing order)
    od = _read_csv_as_ordered_map(combined_out, header_out, KEY_COLS)

    appended_or_updated = 0
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
            in_header = next(reader, None)  # skip header
            if in_header and in_header != header_out:
                print(f"[FATAL] Inconsistent header in {p.name}.\n  got: {in_header}\n  exp: {header_out}")
                sys.exit(4)
            for row in reader:
                if not row:
                    continue
                k = _row_key_from_list(row, header_out, KEY_COLS)
                od[k] = row  # upsert
                appended_or_updated += 1

    # Write back atomically
    _write_ordered_map_to_csv(combined_out, header_out, od)
    print(f"[MERGE] Upserted {appended_or_updated} rows into: {combined_out}")

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

    # Combined output CSV (one per model & LANG)
    if LANG == "raw":
        output_file = MODEL_OUT_DIR / f"{short}_trec_dl_{TREC_DL_YEAR}_raw.csv"
    else:
        output_file = MODEL_OUT_DIR / f"{short}_trec_dl_{TREC_DL_YEAR}_{LANG}.csv"

    tokens_csv  = MODEL_OUT_DIR / "token_usage.csv"

    # Build the list of part files to process
    part_files = list(iter_part_files(START_PART, END_PART))
    if not part_files:
        print("[INFO] No part files found in range.")
        return

    run_id = timestamp_id()
    print(f"\n--- Running inference for model: {model_id} (run_id={run_id}, LANG={LANG}) ---")
    print("[STOP] Press 'Q' at any time to stop after the current in-flight items.")

    # Temp per-file labels live under the model's output folder
    per_file_out_dir = MODEL_OUT_DIR / f"_tmp_{run_id}_{model_id.replace(':','_')}_{LANG}"
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
        # All per-file outputs should share identical header_out (since all parts share schema)
        header_out_set = {tuple(r["header_out"]) for r in results}
        if len(header_out_set) != 1:
            print(f"[FATAL] Inconsistent output headers across files: {header_out_set}")
            sys.exit(4)
        header_out = list(next(iter(header_out_set)))

        # Merge per-file label CSVs into the per-model combined OUTPUT_FILE
        per_file_labels = [r["labels_csv"] for r in results]
        await asyncio.to_thread(merge_labels, per_file_labels, output_file, header_out, stop_event)

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
    combined_log_index = MODEL_LOGS_DIR / f"{run_id}_llm_logs_index_{safe_model}_{LANG}.json"
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
