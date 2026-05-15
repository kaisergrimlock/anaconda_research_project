#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue, Empty
from threading import Lock

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import (
    bump_field_limit,
    ensure_csv_with_header,
    model_short_name,
    _inspect_header,
    write_combined_dynamic,
)
from scripts.log_helpers import timestamp_id
# ===============================================================
# Config
# ===============================================================

TREC_DL_YEAR = "2021"
LANGS = [
    "ru_instruct",
    "zh_instruct",
    "ga_instruct",
    "ar_instruct",
    "fr_instruct",
    "vi_instruct",
    "sw_instruct",
    "ga_instruct",
    "eng_instruct",
    "hi_instruct",
    "he_instruct",
    "th_instruct",
]


START_PART = 1
END_PART = 6
MODE = "replace"
MODELS = ["openai.gpt-oss-20b-1:0"]
#MODELS = ["qwen.qwen3-32b-v1:0"]
#MODELS = ["meta.llama3-8b-instruct-v1:0"]

CRITERIA = ["contextuality", "coverage", "exactness", "topicality"]
RELEVANCE_COL = "relevance"

FORCE_REBUILD_CACHE = False

PROMPT_TYPE = "criterion"
PROMPT_NAME = "composition_2"
PROMPT_FILE = Path(f"prompts/{PROMPT_TYPE}/{PROMPT_NAME}.txt")
OUTPUT_SUFFIX = "crit_2"

OUTPUT_ROOT_BASE = PROJECT_ROOT / "outputs" / "llm_label" / f"trec_dl_{TREC_DL_YEAR}"
LOG_ROOT_DIR = PROJECT_ROOT / "logs"

# ===============================================================
# Concurrency knobs
# ===============================================================

PART_CONCURRENCY = 6
ROW_CONCURRENCY = 50
ROW_QUEUE_MAXSIZE = 2 * ROW_CONCURRENCY

if MODELS[0].startswith("meta.llama3"):
    ROW_CONCURRENCY = 1
    ROW_QUEUE_MAXSIZE = 2 * ROW_CONCURRENCY

bump_field_limit()

# ===============================================================
# Part A: Build cache from per-criterion CSVs
# ===============================================================

RowKey = Tuple[str, str]
RowDict = Dict[RowKey, Dict[str, Any]]


def criterion_dir_for_short(short: str) -> Path:
    return OUTPUT_ROOT_BASE / short / "criterion"


def cache_dir_for_short(short: str, lang: str) -> Path:
    return OUTPUT_ROOT_BASE / short / "criteria_composed" / lang


def cache_prefix_for_short(short: str, lang: str) -> str:
    return f"{short}_trecdl_{TREC_DL_YEAR}_{lang}_criterion_cache"


def find_file_for_criterion(crit_dir: Path, criterion: str, lang: str) -> Path:
    pattern = f"*_{lang}_{criterion}_labels.csv"
    matches = list(crit_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {crit_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files found for criterion '{criterion}': {matches}")
    return matches[0]


def passage_col_out(lang: str) -> str:
    return "passage" if lang == "raw" else "passage_injected"


def load_criterion_into_dict(
    data: RowDict,
    csv_path: Path,
    criterion_name: str,
    lang: str,
) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return

        if RELEVANCE_COL not in fieldnames:
            raise KeyError(
                f"Expected relevance column '{RELEVANCE_COL}' not found in {csv_path.name}. "
                f"Available columns: {fieldnames}"
            )

        last_col_name = fieldnames[-1]

        for row in reader:
            qid = (row.get("qid", "") or "").strip()
            pid = (row.get("pid", "") or "").strip()
            query = row.get("query", "") or ""

            if lang == "raw":
                passage_val = row.get("passage", "") or row.get("passage_injected", "") or ""
            else:
                passage_val = row.get("passage_injected", "") or row.get("passage", "") or ""

            criterion_score = (row.get(last_col_name, "") or "").strip()
            relevance_val = (row.get(RELEVANCE_COL, "") or "").strip()

            key: RowKey = (qid, pid)
            if key not in data:
                data[key] = {
                    "qid": qid,
                    "pid": pid,
                    "query": query,
                    passage_col_out(lang): passage_val,
                }
            else:
                if not data[key].get(passage_col_out(lang)):
                    data[key][passage_col_out(lang)] = passage_val

            data[key][criterion_name] = criterion_score
            data[key][RELEVANCE_COL] = relevance_val


def build_combined_dict_for_short(short: str, lang: str) -> RowDict:
    crit_dir = criterion_dir_for_short(short)
    if not crit_dir.exists():
        raise FileNotFoundError(f"Criterion directory not found: {crit_dir}")

    combined: RowDict = {}
    for criterion in CRITERIA:
        p = find_file_for_criterion(crit_dir, criterion, lang)
        print(f"[CACHE] Loading {criterion} from {p.name}")
        load_criterion_into_dict(combined, p, criterion, lang)

    return combined


def write_cache_parts_for_short(
    short: str,
    lang: str,
    data: RowDict,
    chunk_size: int = 500,
) -> List[Path]:
    cache_dir = cache_dir_for_short(short, lang)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prefix = cache_prefix_for_short(short, lang)
    fieldnames = ["qid", "pid", "query", passage_col_out(lang)] + CRITERIA + [RELEVANCE_COL]

    rows = list(data.values())
    total = len(rows)
    if total == 0:
        print(f"[WARN] No data to save cache for {short}.")
        return []

    num_parts = (total + chunk_size - 1) // chunk_size
    out_paths: List[Path] = []

    for part_idx in range(num_parts):
        start = part_idx * chunk_size
        end = min(start + chunk_size, total)
        part_rows = rows[start:end]

        part_path = cache_dir / f"{prefix}_part{part_idx + 1:03d}.csv"
        with part_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in part_rows:
                out_row = {fn: r.get(fn, "") for fn in fieldnames}
                w.writerow(out_row)

        out_paths.append(part_path)
        print(f"[CACHE] Wrote {part_path.name} rows {start}..{end-1}")

    return out_paths


def ensure_cache_exists(short: str, lang: str) -> None:
    cache_dir = cache_dir_for_short(short, lang)
    prefix = cache_prefix_for_short(short, lang)
    existing = sorted(cache_dir.glob(f"{prefix}_part*.csv")) if cache_dir.exists() else []

    if existing and not FORCE_REBUILD_CACHE:
        print(f"[CACHE] Found {len(existing)} existing cache parts in {cache_dir}")
        return

    if existing and FORCE_REBUILD_CACHE:
        print("[CACHE] FORCE_REBUILD_CACHE=True, deleting old cache parts...")
        for p in existing:
            p.unlink(missing_ok=True)

    print(f"[CACHE] Building cache parts for {short} (LANG={lang})...")
    combined = build_combined_dict_for_short(short, lang)
    print(f"[CACHE] Total (qid,pid) pairs: {len(combined)}")
    write_cache_parts_for_short(short, lang, combined)


# ===============================================================
# Part B: Deterministic composition over cache parts
# ===============================================================

def parse_criterion_score(value: Any) -> Optional[float]:
    """Return a numeric criterion score, or None if the cell is unusable."""
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        # Fallback for values such as "Score: 2".
        m = re.search(r"[-+]?\d*\.?\d+", text)
        return float(m.group(0)) if m else None


def average_criteria_scores(row: Dict[str, Any]) -> str:
    """
    Calculate final relevance as the average of all available criterion scores.

    If all four criterion columns are present and numeric, this is:
        (contextuality + coverage + exactness + topicality) / 4

    The result is rounded to 2 decimal places. If no criterion score can be parsed,
    the function falls back to the original relevance value.
    """
    scores = [
        parse_criterion_score(row.get(criterion))
        for criterion in CRITERIA
    ]
    valid_scores = [score for score in scores if score is not None]

    if not valid_scores:
        return (row.get(RELEVANCE_COL, "") or "").strip()

    avg = sum(valid_scores) / len(valid_scores)

    # Keep integers tidy, e.g. 2.0 -> "2".
    if avg.is_integer():
        return str(int(avg))

    return f"{avg:.2f}"


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield row


def count_rows(path: Path) -> int:
    return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)


def start_stop_key_listener(loop, stop_event):
    def listen():
        print("[STOP] Press 'Q' to stop.")
        while not stop_event.is_set():
            line = sys.stdin.readline()
            if line.strip().lower() == "q":
                loop.call_soon_threadsafe(stop_event.set)
                break

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    return t


def signal_stop(stop_event: Optional[asyncio.Event]) -> None:
    if stop_event is not None:
        stop_event.set()


# =========================
# ROW-CONCURRENT deterministic scoring
# =========================
def _label_single_part_file_blocking(
    part_csv: Path,
    model_id: str,
    run_id: str,
    out_dir: Path,
    logs_dir: Path,
    lang: str,
    stop_event: Optional[asyncio.Event] = None,
):
    safe_model = model_id.replace(":", "_")
    header_in = _inspect_header(part_csv)

    # deterministic composition only needs criterion columns
    required_cols = CRITERIA
    missing = [c for c in required_cols if c not in header_in]
    if missing:
        raise RuntimeError(f"{part_csv} missing required columns: {missing}")

    header_out = header_in + (["llm_relevance"] if "llm_relevance" not in header_in else [])

    labels_path = out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    n_rows = count_rows(part_csv)
    print(f"[LOAD] {part_csv.name}: {n_rows} rows | row_workers={ROW_CONCURRENCY}")

    lock = Lock()
    row_queue: Queue[Optional[Tuple[int, Dict[str, str]]]] = Queue(maxsize=ROW_QUEUE_MAXSIZE)

    next_to_write = 1
    pending: Dict[int, Tuple[List[str], Dict[str, Any], int, int]] = {}
    done_count = 0

    total_in = 0
    total_out = 0
    logs_json: List[Dict[str, Any]] = []

    fatal_error: List[Optional[BaseException]] = [None]

    def append_row_csv(new_row: List[str]) -> None:
        with labels_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(new_row)

    def flush_ready_locked():
        nonlocal next_to_write, total_in, total_out
        while next_to_write in pending:
            row_values, log_obj, in_tok, out_tok = pending.pop(next_to_write)

            append_row_csv(row_values)
            logs_json.append(log_obj)
            total_in += in_tok
            total_out += out_tok

            print(
                f"[{part_csv.name}] done={done_count}/{n_rows} | written={next_to_write}/{n_rows} "
                f"+tok {in_tok}/{out_tok} (totals {total_in}/{total_out})",
                end="\r",
                flush=True,
            )
            next_to_write += 1

    def fail_fast(exc: BaseException) -> None:
        with lock:
            if fatal_error[0] is None:
                fatal_error[0] = exc
                signal_stop(stop_event)

    def worker():
        nonlocal done_count

        while True:
            if fatal_error[0] is not None:
                break

            try:
                item = row_queue.get(timeout=0.25)
            except Empty:
                if stop_event and stop_event.is_set():
                    break
                continue

            if item is None:
                row_queue.task_done()
                break

            idx, row = item

            if stop_event and stop_event.is_set():
                row_queue.task_done()
                continue

            row_out_map = dict(row)
            pid_resolved = (row.get("pid_resolved") or row.get("pid") or "").strip()

            score = average_criteria_scores(row_out_map)

            row_values = [row_out_map.get(col, "") for col in header_out]
            try:
                irel = header_out.index("llm_relevance")
                row_values[irel] = score
            except ValueError:
                pass

            log_obj = {
                "qid": row_out_map.get("qid"),
                "pid_resolved": pid_resolved,
                "criteria": {criterion: row_out_map.get(criterion, "") for criterion in CRITERIA},
                "llm_relevance": score,
                "calculation": "average of available criterion scores",
                "usage": {"input": 0, "output": 0},
            }

            in_tok, out_tok = 0, 0

            with lock:
                done_count += 1
                pending[idx] = (row_values, log_obj, in_tok, out_tok)
                flush_ready_locked()

            row_queue.task_done()

    workers: List[threading.Thread] = []
    for _ in range(max(1, ROW_CONCURRENCY)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        workers.append(t)

    try:
        for idx, row in enumerate(read_rows(part_csv), start=1):
            if fatal_error[0] is not None:
                break
            if stop_event and stop_event.is_set():
                print("\n[STOP] Early termination.")
                break
            row_queue.put((idx, row))
    finally:
        for _ in workers:
            row_queue.put(None)

        for t in workers:
            t.join()

    if fatal_error[0] is not None:
        raise fatal_error[0]

    with lock:
        flush_ready_locked()

    log_file = logs_dir / f"{run_id}_{safe_model}_{part_csv.stem}.json"
    log_file.write_text(json.dumps(logs_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[DONE] {part_csv.name} → {labels_path}")
    return {
        "labels_csv": str(labels_path),
        "header_out": header_out,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "rows": n_rows,
    }


async def label_single_part_file(*args, **kwargs):
    return await asyncio.to_thread(_label_single_part_file_blocking, *args, **kwargs)


def write_combined(per_file_csvs, header_out, short, lang, year, mode, out_dir):
    lang_tag = f"{lang}_{OUTPUT_SUFFIX}"
    return write_combined_dynamic(
        per_file_labels=per_file_csvs,
        header_out=header_out,
        model_short=short,
        lang=lang_tag,
        year=year,
        mode=mode,
        out_dir=out_dir,
    )


async def run_for_model(model_id: str, lang: str, stop_event: asyncio.Event, mode: str):
    short = model_short_name(model_id)

    ensure_cache_exists(short, lang)

    part_dir = cache_dir_for_short(short, lang)
    part_pattern = f"{cache_prefix_for_short(short, lang)}_part{{n:03d}}.csv"

    part_files = [
        part_dir / part_pattern.format(n=i)
        for i in range(START_PART, END_PART + 1)
        if (part_dir / part_pattern.format(n=i)).exists()
    ]

    if not part_files:
        print(f"[WARN] No part files found in {part_dir}")
        return

    run_id = timestamp_id()

    out_dir = OUTPUT_ROOT_BASE / short
    logs_dir = LOG_ROOT_DIR / short
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    per_file_tmp = out_dir / f"_tmp_{run_id}_{lang}"
    per_file_tmp.mkdir(exist_ok=True)

    sem = asyncio.Semaphore(PART_CONCURRENCY)
    results = []

    async def task(p: Path):
        async with sem:
            if stop_event.is_set():
                return None
            return await label_single_part_file(
                p, model_id, run_id, per_file_tmp, logs_dir, lang, stop_event
            )

    tasks = [asyncio.create_task(task(p)) for p in part_files]

    try:
        for t in asyncio.as_completed(tasks):
            if stop_event.is_set():
                for tt in tasks:
                    if not tt.done():
                        tt.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                print("[STOP] Cancelled remaining parts.")
                break

            try:
                r = await t
                if r:
                    results.append(r)
            except Exception as e:
                print(f"[FATAL] {e}")
                stop_event.set()
                for tt in tasks:
                    if not tt.done():
                        tt.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                print("[ABORT] Stopping before merge because API call failed.")
                return
    finally:
        pass

    if stop_event.is_set():
        print("[ABORT] Stopping before merge because stop event is set.")
        return

    if not results:
        print("[INFO] No results to merge.")
        return

    header_out = results[0]["header_out"]
    per_files = [r["labels_csv"] for r in results]

    combined = write_combined(per_files, header_out, short, lang, TREC_DL_YEAR, mode, out_dir)
    print(f"[COMBINED] {combined}")


async def main():
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    start_stop_key_listener(loop, stop_event)

    for lang in LANGS:
        if stop_event.is_set():
            break
        for model_id in MODELS:
            if stop_event.is_set():
                break
            await run_for_model(model_id, lang, stop_event, MODE)


if __name__ == "__main__":
    asyncio.run(main())