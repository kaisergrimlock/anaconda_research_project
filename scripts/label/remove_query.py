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

# =========================
# Repo imports
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import (
    bump_field_limit,
    ensure_csv_with_header,
    model_short_name,
    _inspect_header,
)

from scripts.log_helpers import (
    timestamp_id,
    timestamp_iso,
    estimate_run_cost,
    append_token_row,
    write_run_log_index,
)

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

PROMPT_FILE = Path("prompts/remove_query_lang.txt")
LLM_COST_CSV = Path("scripts/report/llm_cost.csv")

TREC_DL_YEAR = "2022"

LANGS = ["eng", "raw", "vi", "vi_word", "ga", "ga_word"]

START_PART = 1
END_PART = 6

ROWS_PER_OUTPUT_PART = 500
OUTPUT_PART_START = 1
OUTPUT_PART_END = 6

# MODELS = ["meta.llama3-8b-instruct-v1:0"]
# MODELS = ["qwen.qwen3-32b-v1:0"]
MODELS = ["openai.gpt-oss-20b-1:0"]

INFERENCE_CONFIG = {
    "maxTokens": 2000,
    "temperature": 0.0,
    "topP": 1.0,
}

OUTPUT_COL = "passage_removed"

short = model_short_name(MODELS[0])

LOG_ROOT_DIR = Path("logs/removed_query") / short

if MODELS[0].startswith("meta.llama3"):
    ROW_CONCURRENCY = 2
else:
    ROW_CONCURRENCY = 50

ROW_QUEUE_MAXSIZE = 2 * ROW_CONCURRENCY

PART_PATTERN = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

bump_field_limit()


# =========================
# Helpers
# =========================
def get_part_dir(lang: str) -> Path:
    if lang == "raw":
        return Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
    return Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{lang}/")


def get_output_dir(lang: str) -> Path:
    return Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{lang}_qp_rem")


def get_query_lang_col(lang: str) -> str:
    return f"query_{lang}"


def get_output_header(lang: str) -> List[str]:
    return [
        "qid",
        "query",
        "pid",
        "passage",
        "relevance",
        get_query_lang_col(lang),
        "passage_injected",
        OUTPUT_COL,
    ]


def iter_part_files(lang: str, start: int, end: int):
    part_dir = get_part_dir(lang)

    for n in range(start, end + 1):
        p = part_dir / PART_PATTERN.format(n=n)

        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file for lang={lang}: {p}")


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


def start_stop_key_listener(
    loop: asyncio.AbstractEventLoop,
    stop_event: asyncio.Event,
) -> threading.Thread:
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
    return (
        (row.get("pid_resolved", "") or "").strip()
        or (row.get("docid", "") or "").strip()
        or (row.get("pid", "") or "").strip()
        or (row.get("pid_qrels", "") or "").strip()
        or (row.get("passage_id", "") or "").strip()
    )


def _append_row_csv(path: Path, header: List[str], new_row: List[str]) -> None:
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(new_row)


def build_prompt(prompt_template: str, passage: str, query: str, row_idx: int) -> str:
    prompt = (
        prompt_template
        .replace("{passage}", passage)
        .replace("{query}", query)
    )

    unresolved = [
        token for token in ["{passage}", "{query}", "{{passage}}", "{{query}}"]
        if token in prompt
    ]

    if unresolved:
        raise RuntimeError(
            f"Prompt placeholders were not replaced at row {row_idx}. "
            f"Unresolved placeholders: {unresolved}. "
            f"Check that prompts/remove_query_lang.txt uses single braces."
        )

    if not passage.strip():
        raise RuntimeError(f"Empty passage at row {row_idx}")

    if not query.strip():
        raise RuntimeError(f"Empty query at row {row_idx}")

    return prompt


def write_split_csvs(
    per_file_csvs: List[str],
    header_out: List[str],
    lang: str,
    year: str,
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        out_dir / f"all_topics_trecdl_{year}_part{part_no}.csv"
        for part_no in range(OUTPUT_PART_START, OUTPUT_PART_END + 1)
    ]

    for path in output_paths:
        with path.open("w", encoding="utf-8", newline="") as fout:
            csv.writer(fout).writerow(header_out)

    part_index = 0
    rows_written_to_part = 0

    for file_path in sorted(per_file_csvs):
        with Path(file_path).open("r", encoding="utf-8", newline="") as fin:
            reader = csv.reader(fin)
            next(reader, None)

            for row in reader:
                if part_index >= len(output_paths):
                    print("[WARN] More than 3000 rows found; extra rows were not written.")
                    return output_paths

                with output_paths[part_index].open("a", encoding="utf-8", newline="") as fout:
                    csv.writer(fout).writerow(row)

                rows_written_to_part += 1

                if rows_written_to_part >= ROWS_PER_OUTPUT_PART:
                    part_index += 1
                    rows_written_to_part = 0

    return output_paths


# =========================
# Core processing
# =========================
def _process_single_part_file_blocking(
    part_csv: Path,
    model_id: str,
    prompt_template: str,
    run_id: str,
    per_file_out_dir: Path,
    logs_dir: Path,
    lang: str,
    stop_event: Optional[asyncio.Event] = None,
) -> dict:
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")

    per_file_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    header_in = _inspect_header(part_csv)

    query_lang_col = get_query_lang_col(lang)
    header_out = get_output_header(lang)

    required_cols = [
        "qid",
        "query",
        "passage",
        "relevance",
    ]

    if lang == "raw":
        required_cols.append("passage")
    else:
        required_cols.append("passage_injected")

    missing = [c for c in required_cols if c not in header_in]

    if missing:
        print(f"[FATAL] {part_csv.name}: missing required columns {missing}.")
        print(f"[HEADER FOUND] {header_in}")
        sys.exit(2)

    labels_path = per_file_out_dir / f"{part_csv.stem}_removed_query_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    bedrock = make_bedrock_runtime_client(cfg)

    total_rows = count_data_rows(part_csv)

    print(f"[{lang}] [{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] output columns = {header_out}")
    print(f"[CONCURRENCY] row_workers={ROW_CONCURRENCY}")

    write_lock = Lock()

    total_in = 0
    total_out = 0

    logs: List[Dict[str, Any]] = []

    next_to_write = 1
    pending: Dict[int, Tuple[List[str], Dict[str, Any], int, int]] = {}

    row_queue: "Queue[Optional[Tuple[int, Dict[str, str]]]]" = Queue(
        maxsize=ROW_QUEUE_MAXSIZE
    )

    def flush_ready_locked():
        nonlocal next_to_write, total_in, total_out

        while next_to_write in pending:
            row_values, log_obj, in_tok, out_tok = pending.pop(next_to_write)

            _append_row_csv(labels_path, header_out, row_values)
            logs.append(log_obj)

            total_in += in_tok
            total_out += out_tok

            print(
                f"[{lang}] [{part_csv.name}] [{next_to_write}/{total_rows}] "
                f"tokens in/out += {in_tok}/{out_tok} "
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

            row_out_map: Dict[str, str] = {
                k: (row.get(k, "") or "") for k in header_in
            }

            pid = _resolve_pid(row_out_map)
            query = (row_out_map.get("query", "") or "").strip()

            if lang == "raw":
                passage = (row_out_map.get("passage", "") or "").strip()
                passage_injected = passage
            else:
                passage = (row_out_map.get("passage_injected", "") or "").strip()
                passage_injected = passage

            response_text = ""
            reasoning = ""
            in_tok = 0
            out_tok = 0
            prompt = ""

            try:
                prompt = build_prompt(
                    prompt_template=prompt_template,
                    passage=passage,
                    query=query,
                    row_idx=idx,
                )

                if idx == 1:
                    print("\n[PROMPT DEBUG: FIRST ROW]")
                    print(prompt[:1500])
                    print("[END PROMPT DEBUG]\n")

                result = converse_prompt(
                    bedrock,
                    model_id=model_id,
                    prompt=prompt,
                    inference_config=INFERENCE_CONFIG,
                )

                response_text = (result.text or "").strip()
                reasoning = result.reasoning or ""
                in_tok = int(result.input_tokens or 0)
                out_tok = int(result.output_tokens or 0)

            except Exception as err:
                print(
                    f"\n[ERROR] {part_csv.name}: row={idx}, pid={pid}, "
                    f"query={query[:80]!r} :: {err}"
                )

                response_text = ""

            query_lang_value = (
                row_out_map.get(query_lang_col, "")
                or row_out_map.get("query_injected", "")
                or row_out_map.get("query", "")
            )

            row_values = [
                row_out_map.get("qid", ""),
                row_out_map.get("query", ""),
                pid,
                row_out_map.get("passage", ""),
                row_out_map.get("relevance", ""),
                query_lang_value,
                passage_injected,
                response_text,
            ]

            log_obj = {
                "qid": (row_out_map.get("qid", "") or "").strip(),
                "pid": pid,
                "query": query,
                "passage": passage,
                "prompt": prompt,
                "response_text": response_text,
                "reasoning": reasoning,
                "usage": {
                    "inputTokens": in_tok,
                    "outputTokens": out_tok,
                },
                "output_col": OUTPUT_COL,
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

    per_file_log = logs_dir / f"{run_id}_removed_query_{safe_model}_{part_csv.stem}.json"

    with per_file_log.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    print()
    print(
        f"[{lang}] [{part_csv.name}] Wrote: {labels_path.name} | "
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


async def process_single_part_file(*args, **kwargs) -> dict:
    return await asyncio.to_thread(_process_single_part_file_blocking, *args, **kwargs)


# =========================
# Orchestrator
# =========================
async def run_for_model_and_lang(
    model_id: str,
    lang: str,
    stop_event: asyncio.Event,
):
    if not PROMPT_FILE.exists():
        print(f"[FATAL] Prompt file does not exist: {PROMPT_FILE}")
        sys.exit(2)

    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    if "{passage}" not in prompt_template or "{query}" not in prompt_template:
        print("[FATAL] Prompt file must contain both {passage} and {query}.")
        print(f"[PROMPT FILE] {PROMPT_FILE}")
        sys.exit(2)

    if "{{passage}}" in prompt_template or "{{query}}" in prompt_template:
        print("[FATAL] Prompt file uses double braces. Use single braces only.")
        print('Use: "{passage}" and "{query}"')
        sys.exit(2)

    short = model_short_name(model_id)

    model_out_dir = get_output_dir(lang)
    model_logs_dir = LOG_ROOT_DIR / lang

    model_out_dir.mkdir(parents=True, exist_ok=True)
    model_logs_dir.mkdir(parents=True, exist_ok=True)

    part_files = list(iter_part_files(lang, START_PART, END_PART))

    if not part_files:
        print(f"[INFO] No part files found for lang={lang}.")
        return

    run_id = timestamp_id()

    print(
        f"\n--- Removing query text ---\n"
        f"Model: {model_id}\n"
        f"Run ID: {run_id}\n"
        f"Lang: {lang}\n"
        f"Prompt: {PROMPT_FILE}\n"
        f"Output dir: {model_out_dir}\n"
    )

    print("[STOP] Press 'Q' at any time to stop after current in-flight rows.")

    per_file_out_dir = model_out_dir / f"_tmp_{run_id}_{model_id.replace(':', '_')}_{lang}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(min(6, len(part_files)))
    results: List[Dict[str, Any]] = []

    async def sem_task(p: Path):
        async with sem:
            if stop_event.is_set():
                return None

            return await process_single_part_file(
                p,
                model_id,
                prompt_template,
                run_id,
                per_file_out_dir,
                model_logs_dir,
                lang,
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
        print(f"[DONE] No outputs to merge for lang={lang}.")
        return

    header_out_set = {tuple(r["header_out"]) for r in results}

    if len(header_out_set) != 1:
        print(f"[FATAL] Inconsistent output headers across parts: {header_out_set}")
        sys.exit(4)

    header_out = list(next(iter(header_out_set)))
    per_file_csvs = [r["labels_csv"] for r in results]

    split_paths = write_split_csvs(
        per_file_csvs=per_file_csvs,
        header_out=header_out,
        lang=lang,
        year=TREC_DL_YEAR,
        out_dir=model_out_dir,
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
        model_logs_dir / f"{run_id}_removed_query_logs_index_{short}_{lang}.json",
    )

    append_token_row(
        model_out_dir / "token_usage.csv",
        {
            "run_id": run_id,
            "timestamp": timestamp_iso(),
            "model": model_id,
            "num_examples": num_rows,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "estimated_cost_usd": f"{cost_usd:.6f}",
            "labels_csv": ";".join(str(p) for p in split_paths),
            "log_json": "(see logs index)",
        },
    )

    print()
    print(f"[DONE] Model: {model_id}")
    print(f"[DONE] Lang: {lang}")
    print(f"[DONE] Rows: {num_rows}")
    print("[OUTPUT FILES]")
    for p in split_paths:
        print(f"  {p}")
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

            for lang in LANGS:
                if stop_event.is_set():
                    break

                await run_for_model_and_lang(model_id, lang, stop_event)

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