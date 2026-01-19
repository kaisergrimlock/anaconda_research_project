#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import sys
import threading
import shutil
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue
from threading import Lock

import boto3
from botocore.config import Config

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
)
from scripts.log_helpers import (
    timestamp_id,
    timestamp_iso,
    estimate_run_cost,
    append_token_row,
    write_run_log_index,
)

# =========================
# Bedrock / prompt config
# =========================
cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)

PROMPT_DIR = Path("prompts/criterion")
PROMPT_FILE = PROMPT_DIR / "prompt.txt"
CRITERIA_CSV = PROMPT_DIR / "criteria.csv"
LLM_COST_CSV = Path("scripts/report/llm_cost.csv")

# Filled from criteria.csv at runtime
CRITERION_NAME: str = ""
CRITERION_DESC: str = ""
CRITERION_COL: str = ""   # column name in output CSV (same as CRITERION_NAME)

# Input relevance column
RELEVANCE_COL = "relevance"   # change if needed

# =========================
# Data / run config
# =========================
# Batch languages (e.g. ["raw", "vi", ...])
LANGS = ["eng", "fr", "zh", "vi"," th"]
START_PART = 0
END_PART = 0
TREC_DL_YEAR = "2021"
MODE = "replace"     # "append" or "replace"

# Which criteria to run (names in criteria.csv; case-insensitive)
CRITERION_KEYS = ["topicality"]

# Input part files
PART_PATTERN = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

# Models
MODELS = ["openai.gpt-oss-20b-1:0"]
INFERENCE_CONFIG = {"maxTokens": 128000, "temperature": 0.0, "topP": 1.0}

# Output roots
short = model_short_name(MODELS[0])
OUTPUT_ROOT_DIR = Path(f"outputs/llm_label/trec_dl_{TREC_DL_YEAR}/{short}/criterion/")
LOG_ROOT_DIR = Path("logs")

# =========================
# Concurrency knobs
# =========================
# Part-level concurrency is controlled in run_for_model via asyncio semaphore.
# Row-level concurrency accelerates Bedrock calls within each part file.
ROW_CONCURRENCY = 50
ROW_QUEUE_MAXSIZE = 2 * ROW_CONCURRENCY

# Allow large CSV fields
bump_field_limit()

# Global criterion key mutated in main loop (matches your current structure)
CRITERION_KEY: str = ""


# =========================
# Utilities
# =========================
def load_criterion_from_csv() -> None:
    """
    Populate CRITERION_NAME, CRITERION_DESC, CRITERION_COL
    from prompts/criterion/criteria.csv using CRITERION_KEY.
    """
    global CRITERION_NAME, CRITERION_DESC, CRITERION_COL

    if not CRITERIA_CSV.exists():
        print(f"[FATAL] Criteria CSV not found: {CRITERIA_CSV}")
        sys.exit(1)

    with CRITERIA_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        print(f"[FATAL] No rows found in criteria CSV: {CRITERIA_CSV}")
        sys.exit(1)

    rows: List[Dict[str, str]] = []
    for row in raw_rows:
        norm_row = {(k.strip() if k is not None else ""): (v or "") for k, v in row.items()}
        rows.append(norm_row)

    key_norm = CRITERION_KEY.strip().lower()

    def get_name(row: Dict[str, str]) -> Optional[str]:
        return (
            row.get("criterion")
            or row.get("criterion_name")
            or row.get("Criterion Name")
            or row.get("name")
            or row.get("Name")
        )

    def get_desc(row: Dict[str, str]) -> Optional[str]:
        return (
            row.get("description")
            or row.get("criterion_desc")
            or row.get("Criterion Description")
            or row.get("Description")
        )

    matched_row: Optional[Dict[str, str]] = None
    available_names: List[str] = []

    for row in rows:
        name = get_name(row)
        if name:
            name_stripped = name.strip()
            available_names.append(name_stripped)
            if name_stripped.lower() == key_norm:
                matched_row = row
                break

    if matched_row is None:
        print(
            f"[FATAL] Could not find criterion with name {CRITERION_KEY!r} "
            f"in criteria CSV: {CRITERIA_CSV}\n"
            f"Available names: {available_names}"
        )
        sys.exit(1)

    name = get_name(matched_row)
    desc = get_desc(matched_row)

    if not name:
        print(f"[FATAL] Matched row has no name field: {matched_row}")
        sys.exit(1)
    if not desc:
        print(f"[FATAL] Matched row has no description field: {matched_row}")
        sys.exit(1)

    CRITERION_NAME = name.strip()
    CRITERION_DESC = desc.strip()
    CRITERION_COL = CRITERION_NAME

    print(f"[CRITERION] Using criterion: {CRITERION_NAME!r}")
    print(f"[CRITERION] Description: {CRITERION_DESC}")


def get_part_dir(lang: str) -> Path:
    if lang == "raw":
        return Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
    return Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{lang}/")


def iter_part_files(lang: str, start: int, end: int):
    part_dir = get_part_dir(lang)
    for n in range(start, end + 1):
        p = part_dir / PART_PATTERN.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")


def parse_llm_text_to_score(text: str, model_id: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()

    m = re.fullmatch(r"([0-3])", text)
    if m:
        return m.group(1)

    m = re.search(r"Score\s*:\s*([0-3])", text, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r"\ba\s+score\s+of\s+([0-3])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r"\b([0-3])\b", text)
    if m:
        return m.group(1)

    print(f"[WARN] No score pattern found in output: {text[:100]!r}...")
    return ""


def extract_text_from_resp(model_id: str, resp: dict) -> str:
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        return resp["output"]["message"]["content"][0]["text"]
    except Exception:
        print(f"[WARN] Failed to extract text from response for model {model_id}.")
        return ""


def extract_reasoning_from_resp(model_id: str, resp: dict) -> str:
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][0].get("text", "")
        return ""
    except Exception:
        print(f"[WARN] Failed to extract reasoning from response for model {model_id}.")
        return ""


def usage_from_resp(resp: dict) -> Tuple[int, int]:
    u = resp.get("usage", {}) or {}
    return int(u.get("inputTokens", 0) or 0), int(u.get("outputTokens", 0) or 0)


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


# =========================
# CSV merge (your same logic)
# =========================
def write_combined_dynamic(
    per_file_labels: List[str],
    header_out: List[str],
    model_short: str,
    lang: str,
    year: str,
    mode: str,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / f"{model_short}_trecdl_{year}_{lang}_{CRITERION_KEY}_labels.csv"

    def pick_pid_col(header: list[str]) -> str:
        candidates = ["pid", "pid_qrels", "pid_resolved", "docid", "passage_id", "doc_id"]
        for c in candidates:
            if c in header:
                return c
        raise ValueError(
            f"Cannot do replace by qid+pid: no pid column found in header. "
            f"Need one of {candidates}. Header={header}"
        )

    if "qid" not in header_out:
        raise ValueError(f"Cannot do replace by qid+pid: missing 'qid' in header_out={header_out}")

    pid_col = pick_pid_col(header_out)
    qid_i = header_out.index("qid")
    pid_i = header_out.index(pid_col)

    crit_idx = header_out.index(CRITERION_COL) if CRITERION_COL in header_out else None

    def norm_row_len(r: list[str]) -> list[str]:
        if len(r) < len(header_out):
            return r + [""] * (len(header_out) - len(r))
        if len(r) > len(header_out):
            return r[: len(header_out)]
        return r

    def make_key(r: list[str]) -> str:
        r = norm_row_len(r)
        return f"{(r[qid_i] or '').strip()}|{(r[pid_i] or '').strip()}"

    def crit_val(r: list[str]) -> str:
        if crit_idx is None:
            return ""
        r = norm_row_len(r)
        return (r[crit_idx] or "").strip()

    def is_blank_or_nan_like(v: str) -> bool:
        s = (v or "").strip()
        if s == "":
            return True
        return s.lower() in {"nan", "none", "null"}

    def should_preserve_old(old: str, new: str) -> bool:
        old_s = (old or "").strip()
        new_s = (new or "").strip()
        return (old_s != "") and is_blank_or_nan_like(new_s)

    incoming: dict[str, list[str]] = {}

    for p in per_file_labels:
        pth = Path(p)
        with pth.open("r", encoding="utf-8", newline="") as f_in:
            reader = csv.reader(f_in)
            in_header = next(reader, None)
            if in_header is None:
                continue
            if list(in_header) != list(header_out):
                raise ValueError(
                    f"Inconsistent header in {pth}.\n"
                    f"  got: {in_header}\n"
                    f"  exp: {header_out}"
                )
            for r in reader:
                r = norm_row_len(r)
                incoming[make_key(r)] = r

    if not combined_path.exists():
        with combined_path.open("w", encoding="utf-8", newline="") as f_out:
            w = csv.writer(f_out)
            w.writerow(header_out)
            for r in incoming.values():
                w.writerow(r)
        print(f"[WRITE] Created new combined file with {len(incoming)} rows: {combined_path}")
        return combined_path

    if mode == "append":
        existing_header = _inspect_header(combined_path)
        if existing_header != header_out:
            raise ValueError(
                f"Combined file header mismatch.\n"
                f"  got: {existing_header}\n"
                f"  exp: {header_out}"
            )
        with combined_path.open("a", encoding="utf-8", newline="") as f_out:
            w = csv.writer(f_out)
            for r in incoming.values():
                w.writerow(r)
        print(f"[APPEND] Appended {len(incoming)} rows to: {combined_path}")
        return combined_path

    if mode != "replace":
        raise ValueError(f"Unknown mode: {mode}")

    existing_header = _inspect_header(combined_path)
    if existing_header != header_out:
        raise ValueError(
            f"Combined file header mismatch.\n"
            f"  got: {existing_header}\n"
            f"  exp: {header_out}"
        )

    tmp_path = combined_path.with_suffix(".tmp.csv")

    replaced = kept = appended_new = preserved_old = 0
    used_keys: set[str] = set()

    with combined_path.open("r", encoding="utf-8", newline="") as f_in, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        _ = next(reader, None)
        writer.writerow(header_out)

        line_no = 1
        for old_row in reader:
            line_no += 1
            old_row = norm_row_len(old_row)
            k = make_key(old_row)

            if k in incoming:
                new_row = norm_row_len(incoming[k])
                used_keys.add(k)

                if crit_idx is not None:
                    old_val = crit_val(old_row)
                    new_val = crit_val(new_row)
                    if should_preserve_old(old_val, new_val):
                        new_row[crit_idx] = old_val
                        preserved_old += 1
                        print(
                            f"[REPLACE-PRESERVE] line={line_no} key={k} "
                            f"{CRITERION_COL}: {old_val!r} -> {new_val!r} (kept {old_val!r})"
                        )
                    else:
                        print(
                            f"[REPLACE] line={line_no} key={k} "
                            f"{CRITERION_COL}: {old_val!r} -> {new_val!r}"
                        )
                else:
                    print(f"[REPLACE] line={line_no} key={k}")

                replaced += 1
                writer.writerow(new_row)
            else:
                kept += 1
                writer.writerow(old_row)

        for k, r in incoming.items():
            if k not in used_keys:
                r = norm_row_len(r)
                appended_new += 1
                writer.writerow(r)
                if crit_idx is not None:
                    print(f"[ADD] key={k} {CRITERION_COL}={crit_val(r)!r} (not previously in file)")
                else:
                    print(f"[ADD] key={k} (not previously in file)")

    tmp_path.replace(combined_path)

    print(
        f"[DONE replace] replaced={replaced} kept={kept} appended_new={appended_new} "
        f"preserved_old={preserved_old} key_cols=('qid','{pid_col}') file={combined_path}"
    )
    return combined_path


# =========================
# Core labelling logic (ROW-CONCURRENT)
# =========================
def _label_single_part_file_blocking(
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

    required_cols = ["query"]
    required_cols.append("passage" if lang == "raw" else "passage_injected")

    if RELEVANCE_COL not in header_in:
        print(
            f"[FATAL] {part_csv.name}: missing required relevance column "
            f"'{RELEVANCE_COL}'. Available columns: {header_in}"
        )
        sys.exit(2)

    missing = [c for c in required_cols if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv.name}: missing required columns {missing}.")
        sys.exit(2)

    # Output header: input + criterion column, then filter passage/passage_injected
    base_header = header_in if CRITERION_COL in header_in else header_in + [CRITERION_COL]

    if lang == "raw":
        header_out = [c for c in base_header if c != "passage_injected"]
    else:
        header_out = [c for c in base_header if c != "passage"]

    output_file = per_file_out_dir / f"{part_csv.stem}_labels_{safe_model}_{CRITERION_KEY}.csv"
    ensure_csv_with_header(output_file, header_out)

    bedrock = boto3.client("bedrock-runtime", config=cfg)

    total_rows = count_data_rows(part_csv)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] LANG='{lang}' | output columns = {header_out}")
    print(f"[CONCURRENCY] row_workers={ROW_CONCURRENCY} queue_max={ROW_QUEUE_MAXSIZE}")

    # Shared state
    lock = Lock()
    total_in = 0
    total_out = 0
    logs: List[Dict[str, Any]] = []

    # Deterministic output ordering even with concurrency
    next_to_write = 1
    pending: Dict[int, Tuple[List[str], Dict[str, Any], int, int]] = {}
    done_count = 0

    row_queue: "Queue[Optional[Tuple[int, Dict[str, str]]]]" = Queue(maxsize=ROW_QUEUE_MAXSIZE)

    SYSTEM_PROMPT = (
        "Please assess how well the provided passage meets specific criteria in relation to the query. "
        "Use the following scoring scale (0-3) for evaluation:\n"
        "0: Not relevant at all / No information provided.\n"
        "1: Marginally relevant / Partially addresses the criterion.\n"
        "2: Fairly relevant / Adequately addresses the criterion.\n"
        "3: Highly relevant / Fully satisfies the criterion."
    )

    def append_row_csv(path: Path, header: List[str], new_row: List[str]) -> None:
        # Header already ensured by ensure_csv_with_header, but keep safe.
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(header)
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(new_row)

    def flush_ready_locked():
        nonlocal next_to_write, total_in, total_out
        while next_to_write in pending:
            row_values, log_obj, in_tok, out_tok = pending.pop(next_to_write)

            append_row_csv(output_file, header_out, row_values)
            logs.append(log_obj)
            total_in += in_tok
            total_out += out_tok

            print(
                f"[{part_csv.name}] done={done_count}/{total_rows} | "
                f"written={next_to_write}/{total_rows} | "
                f"+tok {in_tok}/{out_tok} (totals {total_in}/{total_out})",
                end="\r",
                flush=True,
            )
            next_to_write += 1

    def worker():
        nonlocal done_count
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
            if lang == "raw":
                p_for_prompt = (row_out_map.get("passage", "") or "").strip()
            else:
                p_for_prompt = (row_out_map.get("passage_injected", "") or "").strip()

            prompt = (
                prompt_template
                .replace("{Criterion Name}", CRITERION_NAME)
                .replace("{Criterion Description}", CRITERION_DESC)
                .replace("{Query}", q_for_prompt)
                .replace("{Passage}", p_for_prompt)
            )

            messages = [{"role": "user", "content": [{"text": prompt}]}]
            kwargs = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": INFERENCE_CONFIG,
                "system": [{"text": SYSTEM_PROMPT}],
            }

            text = ""
            score = ""
            in_tok = out_tok = 0
            reasoning = ""

            try:
                resp = bedrock.converse(**kwargs)
                text = extract_text_from_resp(model_id, resp) or ""
                reasoning = extract_reasoning_from_resp(model_id, resp) or ""
                score = parse_llm_text_to_score(text, model_id)
                in_tok, out_tok = usage_from_resp(resp)
            except Exception as api_err:
                print(
                    f"\n[ERROR] {part_csv.name}: API failed on row {idx}, "
                    f"pid_resolved={pr} :: {api_err}"
                )

            # Build output row in the same order as header_out
            row_values = [row_out_map.get(col, "") for col in header_out]
            try:
                idx_col = header_out.index(CRITERION_COL)
                row_values[idx_col] = score
            except ValueError:
                # should not happen
                pass

            log_obj = {
                "qid": (row_out_map.get("qid", "") or "").strip(),
                "pid_qrels": (row_out_map.get("pid_qrels", "") or "").strip(),
                "pid_resolved": pr,
                "prompt": prompt,
                "response_text": text,
                "reasoning": reasoning,
                "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
                "passage_prompt_used": "passage" if lang == "raw" else "passage_injected",
                "query_prompt_used": "query",
                "criterion_name": CRITERION_NAME,
                "criterion_desc": CRITERION_DESC,
                "criterion_score": score,
                "relevance_column": RELEVANCE_COL,
                "relevance_score": (row_out_map.get(RELEVANCE_COL, "") or "").strip(),
            }

            with lock:
                done_count += 1
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

    # Final flush
    with lock:
        flush_ready_locked()

    # per-file json log
    per_file_log = logs_dir / f"{run_id}_llm_responses_{safe_model}_{part_csv.stem}.json"
    with per_file_log.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    print()
    print(
        f"[{part_csv.name}] Wrote labels: {output_file.name} | "
        f"tokens in/out={total_in}/{total_out}"
    )

    return {
        "part": part_csv.name,
        "rows": total_rows,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "labels_csv": str(output_file),
        "log_json": str(per_file_log),
        "header_out": header_out,
    }


async def label_single_part_file(*args, **kwargs) -> dict:
    return await asyncio.to_thread(_label_single_part_file_blocking, *args, **kwargs)


# =========================
# Runner per model
# =========================
async def run_for_model_lang(model_id: str, lang: str, stop_event: asyncio.Event, mode: str):
    if not PROMPT_FILE.exists():
        print(f"[FATAL] Prompt template not found: {PROMPT_FILE}")
        sys.exit(1)
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    short = model_short_name(model_id)

    MODEL_OUT_DIR = OUTPUT_ROOT_DIR
    MODEL_LOGS_DIR = LOG_ROOT_DIR / short
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    part_files = list(iter_part_files(lang, START_PART, END_PART))
    if not part_files:
        print(f"[INFO] No part files found in range for LANG='{lang}'.")
        return

    run_id = timestamp_id()
    print(
        f"\n--- Running inference for model: {model_id} "
        f"(run_id={run_id}, LANG={lang}, mode={mode}) ---"
    )
    print("[STOP] Press 'Q' at any time to stop after the current in-flight items].")

    per_file_out_dir = MODEL_OUT_DIR / f"_tmp_{run_id}_{model_id.replace(':','_')}_{lang}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    # Part-level concurrency (same behavior as you had)
    sem = asyncio.Semaphore(min(6, len(part_files)))
    results: List[Dict[str, Any]] = []

    async def sem_task(p: Path):
        async with sem:
            if stop_event.is_set():
                return None
            return await label_single_part_file(
                p,
                model_id,
                prompt_template,
                run_id,
                per_file_out_dir,
                MODEL_LOGS_DIR,
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
    except Exception as e:
        print(f"[WARN] Failed to estimate cost: {e}")
        cost_usd = 0.0

    write_run_log_index(
        [{"part": r["part"], "log_json": r["log_json"]} for r in results],
        MODEL_LOGS_DIR / f"{run_id}_llm_logs_index_{short}_{lang}.json",
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


# =========================
# Entry point
# =========================
async def main():
    global CRITERION_KEY

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    listener_thread = start_stop_key_listener(loop, stop_event)

    criterion_list = CRITERION_KEYS if CRITERION_KEYS else [CRITERION_KEY]
    lang_list = LANGS or ["raw"]

    try:
        for crit_key in criterion_list:
            if stop_event.is_set():
                break

            CRITERION_KEY = crit_key
            print(f"\n=== Running for criterion: {CRITERION_KEY!r} ===")
            load_criterion_from_csv()

            for lang in lang_list:
                if stop_event.is_set():
                    break
                for model_id in MODELS:
                    if stop_event.is_set():
                        break
                    await run_for_model_lang(model_id, lang, stop_event, MODE)
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
