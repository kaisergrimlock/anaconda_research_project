#!/usr/bin/env python3
from __future__ import annotations

import asyncio, csv, json, sys, threading, shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import (
    bump_field_limit,
    base_trec_cols,
    extra_trec_cols_for_lang,
    output_header_from_input,
    ensure_csv_with_header,
    pick_query_for_lang,
    pick_passage_for_lang,
    model_short_name,
    _inspect_header,
)
from scripts.log_helpers import (
    timestamp_id, timestamp_iso, estimate_run_cost,
    append_token_row, write_run_log_index,
)

# NEW: writer is a library — we call it directly
from writer_csv import write_combined

# ===== config =====
bump_field_limit()
cfg = Config(region_name="us-west-2", connect_timeout=10, read_timeout=300, retries={"max_attempts": 8, "mode": "standard"})

PROMPT_NAME   = "utility"
PROMPT_FILE   = Path(f"prompts/{PROMPT_NAME}.txt")
LLM_COST_CSV  = Path("scripts/report/llm_cost.csv")

LANG          = "ru"       # "raw", "vi", ...
START_PART    = 46
END_PART      = 46
TREC_DL_YEAR  = "2023"
MODE          = "append"  # "append" or "replace"

if LANG == "raw":
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
else:
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{LANG}/")
PART_PATTERN  = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

MODELS = ["openai.gpt-oss-20b-1:0"]
INFERENCE_CONFIG = {"maxTokens": 2000, "temperature": 0.0, "topP": 1.0}

def iter_part_files(a: int, b: int):
    for n in range(a, b + 1):
        p = PART_DIR / PART_PATTERN.format(n=n)
        if p.exists(): yield p
        else: print(f"[WARN] Missing file: {p}")

def parse_llm_text_to_score(text: str) -> str:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "O" in parsed: return str(parsed["O"])
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "O" in item: return str(item["O"])
    except Exception: pass
    return ""

def extract_text_from_resp(model_id: str, resp: dict) -> str:
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        return resp["output"]["message"]["content"][0]["text"]
    except Exception:
        return ""

def usage_from_resp(resp: dict) -> tuple[int, int]:
    u = resp.get("usage", {}) or {}
    return int(u.get("inputTokens", 0) or 0), int(u.get("outputTokens", 0) or 0)

def read_rows_stream(path: Path):
    f = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(f, skipinitialspace=True)
    try:
        for row in reader: yield row
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
                    if ch and ch.lower() == 'q':
                        loop.call_soon_threadsafe(stop_event.set); break
        except ImportError:
            print("[STOP] Type 'Q' + Enter to stop gracefully.")
            while not stop_event.is_set():
                try: line = sys.stdin.readline()
                except Exception: break
                if not line: break
                if line.strip().lower() == 'q':
                    loop.call_soon_threadsafe(stop_event.set); break
    t = threading.Thread(target=_listen, name="stop-key-listener", daemon=True); t.start(); return t

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
    per_file_out_dir.mkdir(parents=True, exist_ok=True); logs_dir.mkdir(parents=True, exist_ok=True)

    header_in = _inspect_header(part_csv)
    missing = [c for c in set(base_trec_cols()) if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv.name}: missing base columns {missing}."); sys.exit(2)
    for c in extra_trec_cols_for_lang(LANG):
        if c not in header_in:
            print(f"[WARN] {part_csv.name}: expected language column '{c}' not found; falling back as needed.")

    header_out = output_header_from_input(header_in)
    labels_path = per_file_out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    bedrock = boto3.client("bedrock-runtime", config=cfg)
    total_in = total_out = 0
    logs: List[Dict[str, Any]] = []

    total_rows = count_data_rows(part_csv)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] LANG='{LANG}' | output columns = input columns + ['llm_relevance']")

    def append_row_csv(path: Path, header: List[str], new_row: List[str]) -> None:
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as f: csv.writer(f).writerow(header)
        with path.open("a", encoding="utf-8", newline="") as f: csv.writer(f).writerow(new_row)

    for idx, row in enumerate(read_rows_stream(part_csv), start=1):
        if stop_event is not None and stop_event.is_set():
            print(f"\n[STOP] Halting early: {part_csv.name}"); break

        row_out_map: Dict[str, str] = {k: (row.get(k, "") or "") for k in header_in}
        pr = (row_out_map.get("pid_resolved", "") or "").strip()
        if not pr:
            pr = (row.get("docid", "") or row.get("pid", "") or row.get("pid_qrels", "") or "").strip()
            row_out_map["pid_resolved"] = pr

        qid       = (row_out_map.get("qid", "") or "").strip()
        pid_qrels = (row_out_map.get("pid_qrels", "") or "").strip()
        passage   = (row_out_map.get("passage", "") or "").strip()
        if not (qid and pid_qrels and passage):
            print(f"[FATAL] {part_csv.name}: missing qid/pid_qrels/passage at row {idx}."); sys.exit(3)

        q_for_prompt = pick_query_for_lang(row_out_map, LANG)
        p_for_prompt = pick_passage_for_lang(row_out_map, LANG)
        prompt = prompt_template.format(query=q_for_prompt, passage=p_for_prompt)
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs   = {"modelId": model_id, "messages": messages, "inferenceConfig": INFERENCE_CONFIG}

        text = ""; score = ""; in_tok = out_tok = 0
        try:
            resp  = bedrock.converse(**kwargs)
            text  = extract_text_from_resp(model_id, resp) or ""
            score = parse_llm_text_to_score(text)
            in_tok, out_tok = usage_from_resp(resp)
            total_in  += in_tok; total_out += out_tok
        except KeyboardInterrupt:
            print(f"[INTERRUPTED] {part_csv.name} at qid {qid} (row {idx})"); break
        except Exception as api_err:
            print(f"[ERROR] {part_csv.name}: API failed on qid={qid}, pid_resolved={pr} (row {idx}) :: {api_err}")

        row_out = [row_out_map.get(col, "") for col in header_in] + [score]
        append_row_csv(labels_path, header_out, row_out)

        logs.append({
            "qid": qid, "pid_qrels": pid_qrels, "pid_resolved": pr,
            "prompt": prompt, "response_text": text,
            "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
            "passage_prompt_used": "passage_injected" if (LANG != "raw" and row_out_map.get("passage_injected")) else "passage",
            "query_prompt_used": f"query_{LANG}" if (LANG != "raw" and row_out_map.get(f"query_{LANG}")) else "query",
            "llm_relevance": score,
        })

        print(f"[{part_csv.name}] [{idx}/{total_rows}] tokens in/out += {in_tok}/{out_tok} (totals {total_in}/{total_out})",
              end="\r", flush=True)

    # per-file json log
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    per_file_log = logs_dir / f"{run_id}_llm_responses_{safe_model}_{part_csv.stem}.json"
    with per_file_log.open("w", encoding="utf-8") as logf:
        json.dump(logs, logf, indent=2, ensure_ascii=False)

    print(); print(f"[{part_csv.name}] Wrote labels: {labels_path.name} | tokens in/out={total_in}/{total_out}")

    return {
        "part": part_csv.name,
        "rows": total_rows,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "labels_csv": str(labels_path),
        "log_json": str(per_file_log),
        "header_out": output_header_from_input(header_in),
    }

async def label_single_part_file(*args, **kwargs) -> dict:
    return await asyncio.to_thread(_label_single_part_file_blocking, *args, **kwargs)

async def run_for_model(model_id: str, stop_event: asyncio.Event, mode: str):
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    short = model_short_name(model_id)
    MODEL_OUT_DIR  = Path("outputs/llm_label") / short
    MODEL_LOGS_DIR = Path("logs") / short
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True); MODEL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    part_files = list(iter_part_files(START_PART, END_PART))
    if not part_files:
        print("[INFO] No part files found in range."); return

    run_id = timestamp_id()
    print(f"\n--- Running inference for model: {model_id} (run_id={run_id}, LANG={LANG}, mode={mode}) ---")
    print("[STOP] Press 'Q' at any time to stop after the current in-flight items].")

    per_file_out_dir = MODEL_OUT_DIR / f"_tmp_{run_id}_{model_id.replace(':','_')}_{LANG}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(min(6, len(part_files)))
    results: List[Dict[str, Any]] = []

    async def sem_task(p: Path):
        async with sem:
            if stop_event.is_set(): return None
            return await label_single_part_file(p, model_id, prompt_template, run_id, per_file_out_dir, MODEL_LOGS_DIR, stop_event)

    tasks = [asyncio.create_task(sem_task(p)) for p in part_files]
    for task in asyncio.as_completed(tasks):
        if stop_event.is_set():
            for t in tasks:
                if not t.done(): t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print("[STOP] Cancelled remaining files."); break
        r = await task
        if r: results.append(r)

    if not results or stop_event.is_set():
        print("[DONE] No outputs to merge."); return

    # verify consistent headers & collect per-file CSVs
    header_out_set = {tuple(r["header_out"]) for r in results}
    if len(header_out_set) != 1:
        print(f"[FATAL] Inconsistent output headers: {header_out_set}"); sys.exit(4)
    header_out = list(next(iter(header_out_set)))
    per_file_labels = [r["labels_csv"] for r in results]

    # >>> DIRECT CALL to writer (no manifest) <<<
    combined_path = write_combined(
        per_file_labels=per_file_labels,
        header_out=header_out,
        model_short=short,
        lang=LANG,
        year=TREC_DL_YEAR,
        mode=mode,
    )

    total_in  = sum(r["input_tokens"]  for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    num_rows  = sum(r["rows"] for r in results)

    try:
        cost_usd = estimate_run_cost(model_id, total_in, total_out, LLM_COST_CSV)
    except Exception:
        cost_usd = 0.0

    # combined list of per-file logs
    write_run_log_index(
        [{"part": r["part"], "log_json": r["log_json"]} for r in results],
        (Path("logs") / short / f"{run_id}_llm_logs_index_{short}_{LANG}.json"),
    )

    append_token_row(MODEL_OUT_DIR / "token_usage.csv", {
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
    })

    print(f"[DONE] Model: {model_id} | Rows: {num_rows} | Combined: {combined_path}")
    print(f"[TOKENS] in={total_in:,} out={total_out:,} total={total_in + total_out:,}")

    # optional: clean up temp per-file outputs for this run
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
            if stop_event.is_set(): break
            await run_for_model(model_id, stop_event, MODE)
    finally:
        stop_event.set()
        try: listener_thread.join(timeout=0.2)
        except Exception: pass

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Top-level stop.")
