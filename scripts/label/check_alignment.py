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

# ===== config =====
cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)

ALIGNMENT_CHECKER_PROMPT_FILE = Path("prompts/alignment_checker_2.txt")
UTILITY_PROMPT_FILE = Path("prompts/label/utility.txt")
LLM_COST_CSV = Path("scripts/report/llm_cost.csv")

LANGUAGES = [
    "eng",
    "eng_instruct"
]                     # list of languages to process
START_PART = 1
END_PART = 6
TREC_DL_YEAR = "2022"
MODE = "replace"       # "append" or "replace"

# Models
MODELS = ['qwen.qwen3-32b-v1:0']
INFERENCE_CONFIG = {"maxTokens": 1000, "temperature": 0.0, "topP": 1.0}

# ===== Concurrency knobs =====
if MODELS[0].startswith("meta.llama3"):
    ROW_CONCURRENCY = 2
else:
    ROW_CONCURRENCY = 50
ROW_QUEUE_MAXSIZE = 2 * ROW_CONCURRENCY

# Debug options
DEBUG_PRINT_PROMPT = True
DEBUG_PROMPT_CHAR_LIMIT = None   # Set to an int like 3000 to truncate printed prompt

# ===== functions =====
bump_field_limit()  # Allow large fields to accommodate passages


def iter_part_files(part_dir: Path, start: int, end: int, year: str):
    part_pattern = f"all_topics_trecdl_{year}_part{{n}}.csv"
    for n in range(start, end + 1):
        p = part_dir / part_pattern.format(n=n)
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


def print_prompt_debug(
    prompt: str,
    model_id: str,
    part_csv: Path,
    idx: int,
    pr: str,
    inference_config: Dict[str, Any],
) -> None:
    if not DEBUG_PRINT_PROMPT:
        return

    shown_prompt = prompt
    if DEBUG_PROMPT_CHAR_LIMIT is not None:
        shown_prompt = prompt[:DEBUG_PROMPT_CHAR_LIMIT]
        if len(prompt) > DEBUG_PROMPT_CHAR_LIMIT:
            shown_prompt += "\n\n[DEBUG] ... prompt truncated ..."

    print("\n" + "=" * 80)
    print(
        f"[DEBUG] Sending prompt to LLM | model={model_id} | "
        f"file={part_csv.name} | row={idx} | pid_resolved={pr}"
    )
    print(f"[DEBUG] inference_config={inference_config}")
    print("=" * 80)
    print(shown_prompt)
    print("=" * 80 + "\n")


def _label_single_part_file_blocking(
    part_csv: Path,
    model_id: str,
    alignment_checker_template: str,
    utility_template: str,
    extracted_task_str: str,
    lang: str,
    run_id: str,
    per_file_out_dir: Path,
    logs_dir: Path,
    stop_event: Optional[asyncio.Event] = None,
) -> dict:
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    per_file_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    header_in = _inspect_header(part_csv)

    # ===== Minimal required columns =====
    required_cols = ["query"]
    if lang == "raw":
        required_cols.append("passage")
    else:
        required_cols.append("passage_injected")

    missing = [c for c in required_cols if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv.name}: missing required columns {missing}.")
        sys.exit(2)

    # ===== Output header = input header + alignment_score =====
    if "alignment_score" in header_in:
        print(f"[WARN] {part_csv.name}: 'alignment_score' already in header; will overwrite values.")
        header_out = header_in
    else:
        header_out = header_in + ["alignment_score"]

    labels_path = per_file_out_dir / f"{part_csv.stem}_alignment_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    # ===== Bedrock client via helper =====
    bedrock = make_bedrock_runtime_client(cfg)

    total_in = total_out = 0
    logs: List[Dict[str, Any]] = []

    total_rows = count_data_rows(part_csv)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] lang='{lang}' | output columns = {header_out}")
    if ROW_CONCURRENCY > 1:
        print(f"[CONCURRENCY] row_workers={ROW_CONCURRENCY} queue_max={ROW_QUEUE_MAXSIZE}")

    write_lock = Lock()
    next_to_write = 1
    pending: Dict[int, Tuple[List[str], Dict[str, Any], int, int]] = {}
    row_queue: "Queue[Optional[Tuple[int, Dict[str, str]]]]" = Queue(maxsize=ROW_QUEUE_MAXSIZE)

    def append_row_csv(path: Path, header: List[str], new_row: List[str]) -> None:
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(header)
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(new_row)

    def flush_ready_locked():
        nonlocal next_to_write, total_in, total_out
        while next_to_write in pending:
            row_values, log_obj, in_tok, out_tok = pending.pop(next_to_write)
            append_row_csv(labels_path, header_out, row_values)
            logs.append(log_obj)
            total_in += in_tok
            total_out += out_tok
            print(
                f"[{part_csv.name}] [{next_to_write}/{total_rows}] tokens in/out += {in_tok}/{out_tok} "
                f"(totals {total_in}/{total_out})",
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

            row_out_map: Dict[str, str] = {k: (row.get(k, "") or "") for k in header_in}

            pr = (row_out_map.get("pid_resolved", "") or "").strip()
            if not pr:
                pr = (
                    row.get("docid", "")
                    or row.get("pid", "")
                    or row.get("pid_qrels", "")
                    or row.get("passage_id", "")
                    or ""
                ).strip()
                if pr and "pid_resolved" in header_in:
                    row_out_map["pid_resolved"] = pr

            q_for_prompt = (row_out_map.get("query", "") or "").strip()
            p_for_prompt = pick_passage_for_lang(row_out_map, lang)

            if not q_for_prompt or not p_for_prompt:
                print(f"[FATAL] {part_csv.name}: missing prompt fields at row {idx}.")
                if stop_event is not None:
                    try:
                        pass
                    except Exception:
                        pass
                score, text, reasoning = "", "", ""
                in_tok = out_tok = 0
                prompt = ""
            else:
                prompt_with_query_and_passage = utility_template.format(
                    query=q_for_prompt, passage=p_for_prompt
                )
                prompt_content = f"{prompt_with_query_and_passage}\n\nPassage:\n{p_for_prompt}"
                prompt = alignment_checker_template.format(
                    extracted_task=extracted_task_str, prompt=prompt_content
                )

                print_prompt_debug(
                    prompt=prompt,
                    model_id=model_id,
                    part_csv=part_csv,
                    idx=idx,
                    pr=pr,
                    inference_config=INFERENCE_CONFIG,
                )

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

                    try:
                        import ast
                        parsed_dict = ast.literal_eval(text.strip())
                        if isinstance(parsed_dict, dict) and "alignment" in parsed_dict:
                            score = str(parsed_dict["alignment"])
                        else:
                            try:
                                parsed_json = json.loads(text.strip())
                                if "alignment" in parsed_json:
                                    score = str(parsed_json["alignment"])
                                else:
                                    score = text.strip()
                            except json.JSONDecodeError:
                                score = text.strip()
                    except Exception:
                        score = text.strip()

                    in_tok = int(result.input_tokens or 0)
                    out_tok = int(result.output_tokens or 0)
                except Exception as api_err:
                    print(f"[ERROR] {part_csv.name}: API failed on row {idx}, pid_resolved={pr} :: {api_err}")

            row_values = [row_out_map.get(col, "") for col in header_in]
            if "alignment_score" in header_in:
                if len(row_values) == len(header_out):
                    row_values[-1] = score
                else:
                    row_values.append(score)
            else:
                row_values.append(score)

            log_obj = {
                "qid": (row_out_map.get("qid", "") or "").strip(),
                "pid_qrels": (row_out_map.get("pid_qrels", "") or "").strip(),
                "pid_resolved": pr,
                "prompt": prompt,
                "response_text": text,
                "reasoning": reasoning,
                "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
                "alignment_score": score,
            }

            with write_lock:
                pending[idx] = (row_values, log_obj, in_tok, out_tok)
                flush_ready_locked()

            row_queue.task_done()

    workers: List[threading.Thread] = []
    for _ in range(max(1, ROW_CONCURRENCY)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        workers.append(t)

    for idx, row in enumerate(read_rows_stream(part_csv), start=1):
        if stop_event is not None and stop_event.is_set():
            print(f"\n[STOP] Halting early: {part_csv.name}")
            break
        row_queue.put((idx, row))

    for _ in workers:
        row_queue.put(None)

    row_queue.join()

    with write_lock:
        flush_ready_locked()

    per_file_log = logs_dir / f"{run_id}_alignment_responses_{safe_model}_{part_csv.stem}.json"
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


async def run_for_model(model_id: str, lang: str, stop_event: asyncio.Event, mode: str):
    alignment_checker_template = ALIGNMENT_CHECKER_PROMPT_FILE.read_text(encoding="utf-8")
    utility_template = UTILITY_PROMPT_FILE.read_text(encoding="utf-8")

    short = model_short_name(model_id)
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    task_file = Path(f"outputs/task_extraction/extracted_tasks_{safe_model}.json")
    
    extracted_task_str = ""
    if task_file.exists():
        with open(task_file, "r", encoding="utf-8") as f:
            try:
                tasks_data = json.load(f)
                for t in tasks_data:
                    extracted_task_str += f"\n- {t.get('Task', '')}: {t.get('Description', '')}"
            except Exception as e:
                print(f"[WARN] Failed to parse target task_extraction json: {e}")
    else:
        print(f"[WARN] Extracted tasks file not found: {task_file}")

    MODEL_OUT_DIR = Path(f"outputs/alignment_checker/trec_dl_{TREC_DL_YEAR}/{short}/")
    MODEL_LOGS_DIR = Path("logs/alignment_checker") / short
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if lang == "raw":
        part_dir = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
    else:
        part_dir = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{lang}/")

    part_files = list(iter_part_files(part_dir, START_PART, END_PART, TREC_DL_YEAR))
    if not part_files:
        print(f"[INFO] No part files found in range for lang={lang} at {part_dir}.")
        return

    run_id = timestamp_id()
    print(
        f"\n--- Running alignment checker inference for model: {model_id} "
        f"(run_id={run_id}, lang={lang}, mode={mode}) ---"
    )
    print("[STOP] Press 'Q' at any time to stop after the current in-flight items.")

    per_file_out_dir = MODEL_OUT_DIR / f"_tmp_{run_id}_{model_id.replace(':','_')}_{lang}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(min(6, len(part_files)))
    results: List[Dict[str, Any]] = []

    async def sem_task(p: Path):
        async with sem:
            if stop_event.is_set():
                return None
            return await label_single_part_file(
                p,
                model_id,
                alignment_checker_template,
                utility_template,
                extracted_task_str,
                lang,
                run_id,
                per_file_out_dir,
                MODEL_LOGS_DIR,
                stop_event,
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
        lang=lang,
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
        MODEL_LOGS_DIR / f"{run_id}_alignment_logs_index_{short}_{lang}.json",
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
        for lang in LANGUAGES:
            if stop_event.is_set():
                break
            for model_id in MODELS:
                if stop_event.is_set():
                    break
                await run_for_model(model_id, lang, stop_event, MODE)
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