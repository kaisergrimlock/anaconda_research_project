#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import threading
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

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

DEFAULT_MODEL_ID = "openai.gpt-oss-20b-1:0"
DEFAULT_TREC_DL_YEAR = "2021"

LLM_COST_CSV = Path("scripts/report/llm_cost.csv")

PROMPT_INSTRUCTION = "Create a variation of this prompt as a full sentence."

INFERENCE_CONFIG = {
    "maxTokens": 200,
    "temperature": 0.0,
    "topP": 1.0,
}

ROW_CONCURRENCY = 50
ROW_QUEUE_MAXSIZE = ROW_CONCURRENCY * 2

OUTPUT_HEADER = ["qid", "query", "query_variation"]

bump_field_limit()


# =========================
# Helpers
# =========================
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def sort_qid(qid: str):
    qid = str(qid)
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def build_prompt(query: str) -> str:
    return (
        f'{PROMPT_INSTRUCTION}\n\n'
        f'Prompt: "{query}"\n\n'
        f"Return only the rewritten full-sentence variation. Do not add explanation."
    )


def read_queries(path: Path):
    """
    Read exported query CSV:
        qid,query

    Keeps only the first occurrence of each qid.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError(f"Input CSV has no header: {path}")

        missing = [c for c in ["qid", "query"] if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"Input CSV is missing required columns {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        seen_qids = set()

        for row in reader:
            qid = normalize_text(row.get("qid", ""))
            query = normalize_text(row.get("query", ""))

            if not qid or not query:
                continue

            if qid in seen_qids:
                continue

            seen_qids.add(qid)

            yield {
                "qid": qid,
                "query": query,
            }


def append_row_csv(path: Path, header: List[str], new_row: List[str]) -> None:
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(new_row)


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


def load_completed_qids(output_csv: Path) -> set[str]:
    """
    Skip only rows that already have a non-empty query_variation.

    If a qid exists in the output CSV but query_variation is blank,
    it will be prompted again.
    """
    if not output_csv.exists():
        return set()

    completed = set()

    with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return set()

        required = {"qid", "query_variation"}
        if not required.issubset(set(reader.fieldnames)):
            return set()

        for row in reader:
            qid = normalize_text(row.get("qid", ""))
            variation = normalize_text(row.get("query_variation", ""))

            if qid and variation:
                completed.add(qid)

    return completed


def merge_variations_csv(existing_csv: Path, new_csv: Path, final_csv: Path) -> None:
    """
    Merge new generated rows into final output.

    Rules:
      - one row per qid
      - completed old rows are preserved if no new row exists
      - new rows replace old rows for the same qid
      - blank old rows are removed if a new completed row exists
    """
    merged: Dict[str, Dict[str, str]] = {}

    if existing_csv.exists():
        with existing_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames and {"qid", "query", "query_variation"}.issubset(set(reader.fieldnames)):
                for row in reader:
                    qid = normalize_text(row.get("qid", ""))
                    if not qid:
                        continue

                    merged[qid] = {
                        "qid": qid,
                        "query": normalize_text(row.get("query", "")),
                        "query_variation": normalize_text(row.get("query_variation", "")),
                    }

    if new_csv.exists():
        with new_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames and {"qid", "query", "query_variation"}.issubset(set(reader.fieldnames)):
                for row in reader:
                    qid = normalize_text(row.get("qid", ""))
                    if not qid:
                        continue

                    merged[qid] = {
                        "qid": qid,
                        "query": normalize_text(row.get("query", "")),
                        "query_variation": normalize_text(row.get("query_variation", "")),
                    }

    with final_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
        writer.writeheader()

        for qid in sorted(merged.keys(), key=sort_qid):
            writer.writerow(merged[qid])


# =========================
# Core processing
# =========================
def process_queries_blocking(
    input_csv: Path,
    output_csv: Path,
    model_id: str,
    year: str,
    stop_event: Optional[asyncio.Event] = None,
) -> dict:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input query CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    short = model_short_name(model_id)
    log_root_dir = Path("logs/query_variations") / short
    log_root_dir.mkdir(parents=True, exist_ok=True)

    ensure_csv_with_header(output_csv, OUTPUT_HEADER)

    run_id = timestamp_id()
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")

    # Important:
    # Write current-run results to a temporary CSV first.
    # At the end, merge it into output_csv. This prevents duplicated blank rows.
    run_output_csv = output_csv.parent / f"_{run_id}_new_query_variations.csv"
    ensure_csv_with_header(run_output_csv, OUTPUT_HEADER)

    bedrock = make_bedrock_runtime_client(cfg)

    completed_qids = load_completed_qids(output_csv)
    all_rows = list(read_queries(input_csv))

    # Re-run rows if qid is missing from output OR query_variation is blank.
    rows = [row for row in all_rows if row["qid"] not in completed_qids]

    total_rows = len(rows)
    skipped_rows = len(all_rows) - total_rows

    print(f"[INPUT] {input_csv}")
    print(f"[OUTPUT] {output_csv}")
    print(f"[TEMP OUTPUT] {run_output_csv}")
    print(f"[MODEL] {model_id}")
    print(f"[ROWS] total={len(all_rows)} | skipped_completed={skipped_rows} | to_process={total_rows}")
    print(f"[CONCURRENCY] row_workers={ROW_CONCURRENCY}")

    write_lock = Lock()
    row_queue: "Queue[Optional[Tuple[int, Dict[str, str]]]]" = Queue(
        maxsize=ROW_QUEUE_MAXSIZE
    )

    total_in = 0
    total_out = 0
    logs: List[Dict[str, Any]] = []

    next_to_write = 1
    pending: Dict[int, Tuple[List[str], Dict[str, Any], int, int]] = {}

    def flush_ready_locked():
        nonlocal next_to_write, total_in, total_out

        while next_to_write in pending:
            row_values, log_obj, in_tok, out_tok = pending.pop(next_to_write)

            append_row_csv(run_output_csv, OUTPUT_HEADER, row_values)
            logs.append(log_obj)

            total_in += in_tok
            total_out += out_tok

            print(
                f"[{next_to_write}/{total_rows}] "
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

            qid = row["qid"]
            query = row["query"]

            prompt = build_prompt(query)

            response_text = ""
            reasoning = ""
            in_tok = 0
            out_tok = 0

            try:
                if idx == 1:
                    print("\n[PROMPT DEBUG: FIRST ROW]")
                    print(prompt)
                    print("[END PROMPT DEBUG]\n")

                result = converse_prompt(
                    bedrock,
                    model_id=model_id,
                    prompt=prompt,
                    inference_config=INFERENCE_CONFIG,
                )

                response_text = normalize_text(result.text or "")
                reasoning = result.reasoning or ""
                in_tok = int(result.input_tokens or 0)
                out_tok = int(result.output_tokens or 0)

            except Exception as err:
                print(f"\n[ERROR] row={idx}, qid={qid}, query={query[:80]!r} :: {err}")
                response_text = ""

            row_values = [
                qid,
                query,
                response_text,
            ]

            log_obj = {
                "qid": qid,
                "query": query,
                "prompt": prompt,
                "response_text": response_text,
                "reasoning": reasoning,
                "usage": {
                    "inputTokens": in_tok,
                    "outputTokens": out_tok,
                },
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

    for idx, row in enumerate(rows, start=1):
        if stop_event is not None and stop_event.is_set():
            print("\n[STOP] Halting early.")
            break

        row_queue.put((idx, row))

    for _ in workers:
        row_queue.put(None)

    row_queue.join()

    with write_lock:
        flush_ready_locked()

    per_run_log = log_root_dir / f"{run_id}_query_variations_{safe_model}_trec_dl{year}.json"

    with per_run_log.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    merge_variations_csv(
        existing_csv=output_csv,
        new_csv=run_output_csv,
        final_csv=output_csv,
    )

    try:
        run_output_csv.unlink(missing_ok=True)
    except Exception as err:
        print(f"[WARN] Could not remove temp output file: {run_output_csv} :: {err}")

    try:
        cost_usd = estimate_run_cost(model_id, total_in, total_out, LLM_COST_CSV)
    except Exception:
        cost_usd = 0.0

    write_run_log_index(
        [{"part": f"trec_dl{year}", "log_json": str(per_run_log)}],
        log_root_dir / f"{run_id}_query_variations_logs_index_{short}_trec_dl{year}.json",
    )

    append_token_row(
        output_csv.parent / "token_usage.csv",
        {
            "run_id": run_id,
            "timestamp": timestamp_iso(),
            "model": model_id,
            "num_examples": len(logs),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "estimated_cost_usd": f"{cost_usd:.6f}",
            "labels_csv": str(output_csv),
            "log_json": str(per_run_log),
        },
    )

    print()
    print(f"[DONE] Rows processed this run: {len(logs)}")
    print(f"[DONE] Output: {output_csv}")
    print(f"[DONE] Log: {per_run_log}")
    print(f"[TOKENS] in={total_in:,} out={total_out:,} total={total_in + total_out:,}")

    return {
        "rows": len(logs),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "output_csv": str(output_csv),
        "log_json": str(per_run_log),
    }


async def process_queries(*args, **kwargs) -> dict:
    return await asyncio.to_thread(process_queries_blocking, *args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GPT query variations from retrieved/queries/trec_dl{year}.csv"
    )

    parser.add_argument(
        "--year",
        default=DEFAULT_TREC_DL_YEAR,
        help="TREC-DL year, e.g. 2021 or 2022.",
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input CSV with qid,query. Default: retrieved/queries/trec_dl{year}.csv",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV. Default: retrieved/queries/variations/trec_dl{year}_query_variations.csv",
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Bedrock model ID.",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    year = str(args.year)

    input_csv = args.input_csv or Path(f"retrieved/queries/trec_dl{year}.csv")
    output_csv = args.output_csv or Path(
        f"retrieved/queries/variations/trec_dl{year}_query_variations.csv"
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    listener_thread = start_stop_key_listener(loop, stop_event)

    try:
        await process_queries(
            input_csv=input_csv,
            output_csv=output_csv,
            model_id=args.model_id,
            year=year,
            stop_event=stop_event,
        )
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
