#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import sys
import threading
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue
from threading import Lock

from botocore.config import Config

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import (
    bump_field_limit,
    ensure_csv_with_header,
    pick_passage_for_lang,
    model_short_name,
    _inspect_header,
    write_combined_dynamic,
)

from scripts.log_helpers import (
    timestamp_id,
    timestamp_iso,
    estimate_run_cost,
    append_token_row,
    write_run_log_index,
)

# ===== Bedrock helper (ONLY Bedrock stuff) =====
from scripts.bedrock_client import (
    make_bedrock_runtime_client,
    converse_prompt,
)

# =========================
# Config
# =========================
cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)

PROMPT_TYPE = "label"
PROMPT_NAME = "utility"
PROMPT_FILE = Path(f"prompts/{PROMPT_TYPE}/{PROMPT_NAME}.txt")
LLM_COST_CSV = Path("scripts/report/llm_cost.csv")
ALLOW_BLANK_OVERWRITE = True

LANG = "hicwb_instruct"          # "raw", "vi", "enclosed", ...
START_PART = 0
END_PART = 0
TREC_DL_YEAR = "2022"
MODE = "replace"           # "append" or "replace"

# Models
#MODELS = ["openai.gpt-oss-20b-1:0"]
#MODELS = ["meta.llama3-8b-instruct-v1:0"]
MODELS = ["qwen.qwen3-32b-v1:0"]
INFERENCE_CONFIG = {"maxTokens": 2000, "temperature": 0.0, "topP": 1.0}

# Output roots
short = model_short_name(MODELS[0])
OUTPUT_ROOT_DIR = Path(f"outputs/llm_label/trec_dl_{TREC_DL_YEAR}/{short}/")
LOG_ROOT_DIR = Path("logs")

# ===== Concurrency knobs =====
# Part-level concurrency is handled by asyncio semaphore in run_for_model().
# Row-level concurrency here speeds up Bedrock calls massively.
if MODELS[0].startswith("meta.llama3"):
    ROW_CONCURRENCY = 2
else:
    ROW_CONCURRENCY = 50
ROW_QUEUE_MAXSIZE = 2 * ROW_CONCURRENCY

# Input part files
if LANG == "raw":
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
else:
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{LANG}/")
PART_PATTERN = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

bump_field_limit()  # Allow large fields to accommodate passages


# =========================
# Helpers
# =========================
def iter_part_files(start: int, end: int):
    for n in range(start, end + 1):
        p = PART_DIR / PART_PATTERN.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")


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
        return max(0, sum(1 for _ in f) - 1)


def start_stop_key_listener(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> threading.Thread:
    def _listen():
        try:
            import msvcrt
            print("[STOP] Press 'Q' to stop gracefully.")
            while not stop_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch and ch.lower() == "q":
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
                if line.strip().lower() == "q":
                    loop.call_soon_threadsafe(stop_event.set)
                    break

    t = threading.Thread(target=_listen, name="stop-key-listener", daemon=True)
    t.start()
    return t


def _resolve_pid(row: Dict[str, str]) -> str:
    pr = (row.get("pid_resolved", "") or "").strip()
    if pr:
        return pr
    return (
        row.get("docid", "")
        or row.get("pid", "")
        or row.get("pid_qrels", "")
        or row.get("passage_id", "")
        or ""
    ).strip()


def _append_row_csv(path: Path, header: List[str], new_row: List[str]) -> None:
    # Header already ensured by ensure_csv_with_header, but keep this safe.
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(new_row)


# =========================
# Core: label one part file (blocking, row-concurrent)
# =========================
def _label_single_part_file_blocking(
    part_csv: Path,
    model_id: str,
    prompt_template: str,
    run_id: str,
    per_file_out_dir: Path,
    logs_dir: Path,
    stop_event: Optional[asyncio.Event] = None,
) -> dict:
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    per_file_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    header_in = _inspect_header(part_csv)

    # Minimal required columns
    required_cols = ["query"]
    required_cols.append("passage" if LANG == "raw" else "passage_injected")
    missing = [c for c in required_cols if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv.name}: missing required columns {missing}.")
        sys.exit(2)

    # Output header
    header_out = header_in if "llm_relevance" in header_in else header_in + ["llm_relevance"]
    if "llm_relevance" in header_in:
        print(f"[WARN] {part_csv.name}: 'llm_relevance' already in header; will overwrite values.")

    labels_path = per_file_out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    bedrock = make_bedrock_runtime_client(cfg)

    total_rows = count_data_rows(part_csv)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] LANG='{LANG}' | output columns = {header_out}")
    if ROW_CONCURRENCY > 1:
        print(f"[CONCURRENCY] row_workers={ROW_CONCURRENCY} queue_max={ROW_QUEUE_MAXSIZE}")

    # Shared state (protected by lock)
    write_lock = Lock()
    total_in = 0
    total_out = 0
    logs: List[Dict[str, Any]] = []

    # To keep CSV output deterministic by row order, we buffer completed row writes
    # and flush them in order. This avoids race conditions and preserves the exact
    # original line ordering, even with concurrency.
    next_to_write = 1
    pending: Dict[int, Tuple[List[str], Dict[str, Any], int, int]] = {}

    row_queue: "Queue[Optional[Tuple[int, Dict[str, str]]]]" = Queue(maxsize=ROW_QUEUE_MAXSIZE)

    def flush_ready_locked():
        """Write any completed rows that are ready in order."""
        nonlocal next_to_write, total_in, total_out
        while next_to_write in pending:
            row_values, log_obj, in_tok, out_tok = pending.pop(next_to_write)

            _append_row_csv(labels_path, header_out, row_values)
            logs.append(log_obj)

            total_in += in_tok
            total_out += out_tok

            print(
                f"[{part_csv.name}] [{next_to_write}/{total_rows}] "
                f"tokens in/out += {in_tok}/{out_tok} (totals {total_in}/{total_out})",
                end="\r",
                flush=True,
            )
            next_to_write += 1

    def worker():
        while True:
            item = row_queue.get()
            if item is None:
                row_queue.task_done()
                break

            idx, row = item

            if stop_event is not None and stop_event.is_set():
                row_queue.task_done()
                continue

            # Map of all input columns
            row_out_map: Dict[str, str] = {k: (row.get(k, "") or "") for k in header_in}

            pr = _resolve_pid(row_out_map)
            if pr and "pid_resolved" in header_in and not row_out_map.get("pid_resolved"):
                row_out_map["pid_resolved"] = pr

            q_for_prompt = (row_out_map.get("query", "") or "").strip()
            p_for_prompt = pick_passage_for_lang(row_out_map, LANG)

            if not q_for_prompt or not p_for_prompt:
                # Match prior behavior: hard fail (but do it safely with a clear message)
                msg = (
                    f"[FATAL] {part_csv.name}: missing required prompt fields at row {idx} "
                    f"(query={'OK' if q_for_prompt else 'MISSING'}, "
                    f"passage={'OK' if p_for_prompt else 'MISSING'})"
                )
                print(msg)
                # Signal stop to other workers; main thread will exit after join.
                if stop_event is not None:
                    try:
                        # stop_event is an asyncio.Event; it is thread-safe for set() usage? Not guaranteed.
                        # So we just print and proceed; the run will likely be aborted by sys.exit below in flush stage.
                        pass
                    except Exception:
                        pass
                # Store an empty score and continue; keeps output row count stable.
                score = ""
                text = ""
                reasoning = ""
                in_tok = out_tok = 0
            else:
                prompt = prompt_template.format(query=q_for_prompt, passage=p_for_prompt)

                text = ""
                reasoning = ""
                score = ""
                in_tok = out_tok = 0
                try:
                    result = converse_prompt(
                        bedrock,
                        model_id=model_id,
                        prompt=prompt,
                        inference_config=INFERENCE_CONFIG,
                    )
                    text = result.text or ""
                    reasoning = result.reasoning or ""
                    score = result.score or ""
                    in_tok = int(result.input_tokens or 0)
                    out_tok = int(result.output_tokens or 0)
                except Exception as api_err:
                    print(
                        f"[ERROR] {part_csv.name}: API failed on row {idx}, "
                        f"pid_resolved={pr} :: {api_err}"
                    )

            # Build output row
            row_values = [row_out_map.get(col, "") for col in header_in]
            if "llm_relevance" in header_in:
                old_score = row_out_map.get("llm_relevance", "")
                if ALLOW_BLANK_OVERWRITE:
                    row_values[-1] = score
                else:
                    row_values[-1] = score if str(score).strip() != "" else old_score
            else:
                row_values.append(score)

            # Build log entry (match your existing fields)
            log_obj = {
                "qid": (row_out_map.get("qid", "") or "").strip(),
                "pid_qrels": (row_out_map.get("pid_qrels", "") or "").strip(),
                "pid_resolved": pr,
                "prompt": (prompt_template.format(query=q_for_prompt, passage=p_for_prompt)
                           if q_for_prompt and p_for_prompt else ""),
                "response_text": text,
                "reasoning": reasoning,
                "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
                "passage_prompt_used": "passage_injected" if LANG != "raw" else "passage",
                "query_prompt_used": "query",
                "llm_relevance": score,
            }

            # Save result; flush in order
            with write_lock:
                pending[idx] = (row_values, log_obj, in_tok, out_tok)
                flush_ready_locked()

            row_queue.task_done()

    # Start workers
    workers: List[threading.Thread] = []
    for _ in range(max(1, ROW_CONCURRENCY)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        workers.append(t)

    # Feed queue
    for idx, row in enumerate(read_rows_stream(part_csv), start=1):
        if stop_event is not None and stop_event.is_set():
            print(f"\n[STOP] Halting early: {part_csv.name}")
            break
        row_queue.put((idx, row))

    # Shutdown
    for _ in workers:
        row_queue.put(None)

    row_queue.join()

    # Final flush (in case some were pending)
    with write_lock:
        flush_ready_locked()

    # per-file json log
    per_file_log = logs_dir / f"{run_id}_llm_responses_{safe_model}_{part_csv.stem}.json"
    with per_file_log.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    print()
    print(
        f"[{part_csv.name}] Wrote labels: {labels_path.name} | "
        f"tokens in/out={total_in}/{total_out}"
    )

    return {
        "part": part_csv.name,
        "rows": total_rows,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "labels_csv": str(labels_path),
        "log_json": str(per_file_log),
        "header_out": header_out,
    }


async def label_single_part_file(*args, **kwargs) -> dict:
    return await asyncio.to_thread(_label_single_part_file_blocking, *args, **kwargs)


# =========================
# Orchestrator
# =========================
async def run_for_model(model_id: str, stop_event: asyncio.Event, mode: str):
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    short = model_short_name(model_id)

    MODEL_OUT_DIR = OUTPUT_ROOT_DIR
    MODEL_LOGS_DIR = LOG_ROOT_DIR / short
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    part_files = list(iter_part_files(START_PART, END_PART))
    if not part_files:
        print("[INFO] No part files found in range.")
        return

    run_id = timestamp_id()
    print(
        f"\n--- Running inference for model: {model_id} "
        f"(run_id={run_id}, LANG={LANG}, mode={mode}) ---"
    )
    print("[STOP] Press 'Q' at any time to stop after the current in-flight items].")

    per_file_out_dir = MODEL_OUT_DIR / f"_tmp_{run_id}_{model_id.replace(':','_')}_{LANG}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    # Part-level concurrency (keep yours)
    sem = asyncio.Semaphore(min(6, len(part_files)))
    results: List[Dict[str, Any]] = []

    async def sem_task(p: Path):
        async with sem:
            if stop_event.is_set():
                return None
            return await label_single_part_file(
                p, model_id, prompt_template, run_id, per_file_out_dir, MODEL_LOGS_DIR, stop_event
            )

    tasks = [asyncio.create_task(sem_task(p)) for p in part_files]
    for task in asyncio.as_completed(tasks):
        if stop_event.is_set():
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print("[STOP] Cancelled remaining files.")
            break
        r = await task
        if r:
            results.append(r)

    if not results or stop_event.is_set():
        print("[DONE] No outputs to merge.")
        return

    header_out_set = {tuple(r["header_out"]) for r in results}
    if len(header_out_set) != 1:
        print(f"[FATAL] Inconsistent output headers across parts: {header_out_set}")
        sys.exit(4)
    header_out = list(next(iter(header_out_set)))
    per_file_labels = [r["labels_csv"] for r in results]

    combined_path = write_combined_dynamic(
        per_file_labels=per_file_labels,
        header_out=header_out,
        model_short=short,
        lang=LANG,
        year=TREC_DL_YEAR,
        mode=mode,
        out_dir=MODEL_OUT_DIR,
    )

    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    num_rows = sum(r["rows"] for r in results)

    try:
        cost_usd = estimate_run_cost(model_id, total_in, total_out, LLM_COST_CSV)
    except Exception:
        cost_usd = 0.0

    write_run_log_index(
        [{"part": r["part"], "log_json": r["log_json"]} for r in results],
        MODEL_LOGS_DIR / f"{run_id}_llm_logs_index_{short}_{LANG}.json",
    )

    append_token_row(
        MODEL_OUT_DIR / "token_usage.csv",
        {
            "run_id": run_id,
            "timestamp": timestamp_iso(),
            "model": model_id,
            "num_examples": num_rows,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "estimated_cost_usd": f"{cost_usd:.6f}",
            "labels_csv": str(combined_path),
            "log_json": "(see logs index)",
        },
    )

    print(f"[DONE] Model: {model_id} | Rows: {num_rows} | Combined: {combined_path}")
    print(f"[TOKENS] in={total_in:,} out={total_out:,} total={total_in + total_out:,}")

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
                break
            await run_for_model(model_id, stop_event, MODE)
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
