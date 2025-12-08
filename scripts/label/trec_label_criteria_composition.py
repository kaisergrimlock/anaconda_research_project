#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import re
import sys
import threading
import shutil
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
    ensure_csv_with_header,
    pick_passage_for_lang,
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

# ===== Bedrock / prompt config =====
cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)

PROMPT_TYPE = "criterion"
PROMPT_NAME = "composition"  # point this at your new relevance prompt file
PROMPT_FILE = Path(f"prompts/{PROMPT_TYPE}/{PROMPT_NAME}.txt")
LLM_COST_CSV = PROJECT_ROOT / "scripts" / "report" / "llm_cost.csv"

# ===== Experiment config =====
LANG = "raw"          # language variant used for prompt / passage selection
START_PART = 1
END_PART = 6
TREC_DL_YEAR = "2022"
MODE = "append"       # "append" or "replace"

# Models (Bedrock IDs)
#   qwen.qwen3-32b-v1:0
#   openai.gpt-oss-20b-1:0
#   meta.llama3-70b-instruct-v1:0
MODELS = ["openai.gpt-oss-20b-1:0"]
INFERENCE_CONFIG = {"maxTokens": 2000, "temperature": 0.0, "topP": 1.0}

# Output base dirs (per-model subdirs are created later)
OUTPUT_ROOT_BASE = PROJECT_ROOT / "outputs" / "llm_label" / f"trec_dl_{TREC_DL_YEAR}"
LOG_ROOT_DIR = PROJECT_ROOT / "logs"

# ===== helpers =====
bump_field_limit()  # allow huge csv fields

RowDict = Dict[str, Any]


def iter_part_files(
    start: int,
    end: int,
    part_dir: Path,
    pattern_template: str,
):
    """Yield each existing criterion-cache part file."""
    for n in range(start, end + 1):
        p = part_dir / pattern_template.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")


def parse_llm_text_to_score(text: str) -> str:
    """
    Parse the model's text output into a score in {0,1,2,3}.

    The new prompt says: "The output must be only a score (0-3)".
    We still make this robust and look for the first standalone digit 0-3,
    in case the model returns e.g. "2", "Score: 2", or "2\n".
    """
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # First, try a strict whole-string match like "0", "1", "2", "3"
    if text in {"0", "1", "2", "3"}:
        return text

    # Fall back: find first standalone digit 0-3
    m = re.search(r"\b([0-3])\b", text)
    if m:
        return m.group(1)

    print(f"[WARN] Could not parse score from text: {text[:100]!r}...")
    return ""


def extract_text_from_resp(model_id: str, resp: dict) -> str:
    """
    Return the main text content from the model's response.
    For openai.* we assume:
      content[0] = reasoning / hidden block
      content[1] = final short answer (the score as text)
    For others we take content[0].
    """
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        return resp["output"]["message"]["content"][0]["text"]
    except Exception:
        print(f"[WARN] Failed to extract text from response for model {model_id}.")
        return ""


def extract_reasoning_from_resp(model_id: str, resp: dict) -> str:
    """Return the model's hidden/chain-of-thought reasoning block when present (openai.*)."""
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
        # total lines minus header
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
    Label all rows in one criterion-cache part file with the relevance prompt,
    writing a *_labels_<model>.csv file and a JSON log. Returns stats dict.
    """
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    per_file_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    header_in = _inspect_header(part_csv)

    # Criterion-cache rows MUST have at least these:
    required_cols = ["query"]
    if LANG == "raw":
        required_cols.append("passage")
    else:
        required_cols.append("passage_injected")

    missing = [c for c in required_cols if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv.name}: missing required columns {missing}.")
        sys.exit(2)

    # Output header = input header + llm_relevance
    if "llm_relevance" in header_in:
        print(f"[WARN] {part_csv.name}: 'llm_relevance' already in header; will overwrite values.")
        header_out = header_in
    else:
        header_out = header_in + ["llm_relevance"]

    labels_path = per_file_out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    bedrock = boto3.client("bedrock-runtime", config=cfg)
    total_in = total_out = 0
    logs: List[Dict[str, Any]] = []

    total_rows = count_data_rows(part_csv)
    print(f"[{part_csv.name}] Loaded {total_rows} rows")
    print(f"[HEADER] LANG='{LANG}' | output columns = {header_out}")

    def append_row_csv(path: Path, header: List[str], new_row: List[str]) -> None:
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(header)
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(new_row)

    for idx, row in enumerate(read_rows_stream(part_csv), start=1):
        if stop_event is not None and stop_event.is_set():
            print(f"\n[STOP] Halting early: {part_csv.name}")
            break

        # Map all input columns
        row_out_map: Dict[str, str] = {k: (row.get(k, "") or "") for k in header_in}

        # best-effort pid_resolved (cache rows might not have it)
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

        # Prompt fields: query + passage
        q_for_prompt = (row_out_map.get("query", "") or "").strip()
        p_for_prompt = pick_passage_for_lang(row_out_map, LANG)

        if not q_for_prompt:
            print(f"[FATAL] {part_csv.name}: missing 'query' at row {idx}.")
            sys.exit(3)
        if not p_for_prompt:
            print(
                f"[FATAL] {part_csv.name}: could not find passage for LANG='{LANG}' "
                f"(expected 'passage' or 'passage_injected*') at row {idx}."
            )
            sys.exit(3)

        # ---- NEW: fetch the four criterion grades (best-effort) ----
        exactness_grade = (row_out_map.get("exactness", "") or "").strip()
        topicality_grade = (row_out_map.get("topicality", "") or "").strip()
        coverage_grade = (row_out_map.get("coverage", "") or "").strip()
        contextuality_grade = (row_out_map.get("contextuality", "") or "").strip()


        # Build prompt using query, passage, and the four grades.
        # This assumes your prompt file uses placeholders like:
        #   {query} {passage} {Exactness_Grade} {Topicality_Grade} {Coverage_Grade} {Contextual_Fit_Grade}
        try:
            prompt = prompt_template.format(
                query=q_for_prompt,
                passage=p_for_prompt,
                exactness=exactness_grade,
                topicality=topicality_grade,
                coverage=coverage_grade,
                contextual=contextuality_grade,
            )
        except KeyError:
            # Fallback if the prompt file doesn't (yet) have those placeholders
            prompt = prompt_template.format(query=q_for_prompt, passage=p_for_prompt)

        SYSTEM_PROMPT = (
            "You are a search quality rater evaluating the relevance of passages. "
            "Given a query and passage, you must provide a score on an integer "
            "scale of 0 to 3 with the following meanings:\n"
            "3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.\n"
            "2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, "
            "or hidden amongst extraneous information.\n"
            "1 = Related: The passage seems related to the query but does not answer it.\n"
            "0 = Irrelevant: The passage has nothing to do with the query.\n"
            "Assume that you are writing an answer to the query. If the passage seems to be related to the query "
            "but does not include any answer to the query, mark it 1. If you would use any of the information "
            "contained in the passage in such an answer, mark it 2. If the passage is primarily about the query, "
            "or contains vital information about the topic, mark it 3. Otherwise, mark it 0."
        )

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {"modelId": model_id, "messages": messages, "inferenceConfig": INFERENCE_CONFIG, "system":[{"text": SYSTEM_PROMPT}]}

        text = ""
        score = ""
        in_tok = out_tok = 0
        reasoning = ""

        try:
            resp = bedrock.converse(**kwargs)
            text = extract_text_from_resp(model_id, resp) or ""
            reasoning = extract_reasoning_from_resp(model_id, resp) or ""
            score = parse_llm_text_to_score(text)
            in_tok, out_tok = usage_from_resp(resp)
            total_in += in_tok
            total_out += out_tok
        except KeyboardInterrupt:
            print(f"[INTERRUPTED] {part_csv.name} at row {idx}")
            break
        except Exception as api_err:
            print(
                f"[ERROR] {part_csv.name}: API failed on row {idx}, "
                f"pid_resolved={pr} :: {api_err}"
            )

        # Build output row
        row_values = [row_out_map.get(col, "") for col in header_in]
        if "llm_relevance" in header_in:
            if len(row_values) == len(header_out):
                row_values[-1] = score
            else:
                row_values.append(score)
        else:
            row_values.append(score)

        append_row_csv(labels_path, header_out, row_values)

        logs.append(
            {
                "qid": (row_out_map.get("qid", "") or "").strip(),
                "pid_qrels": (row_out_map.get("pid_qrels", "") or "").strip(),
                "pid_resolved": pr,
                "prompt": prompt,
                "response_text": text,
                "reasoning": reasoning,
                "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
                "passage_prompt_used": "passage_injected" if LANG != "raw" else "passage",
                "query_prompt_used": "query",
                "llm_relevance": score,
            }
        )

        print(
            f"[{part_csv.name}] [{idx}/{total_rows}] tokens in/out += {in_tok}/{out_tok} "
            f"(totals {total_in}/{total_out})",
            end="\r",
            flush=True,
        )

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


def write_combined_dynamic(
    per_file_labels: List[str],
    header_out: List[str],
    model_short: str,
    lang: str,
    year: str,
    mode: str,
    out_dir: Path,
) -> Path:
    """
    Combine per-part label CSVs into a single CSV, without enforcing specific
    qid/pid schema. Just header + rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / f"{model_short}_trecdl_{year}_{lang}_labels.csv"

    if mode == "replace" or not combined_path.exists():
        # Fresh file: write header and all rows
        with combined_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header_out)
            for p in per_file_labels:
                with Path(p).open("r", encoding="utf-8", newline="") as f_in:
                    reader = csv.reader(f_in)
                    in_header = next(reader, None)
                    if in_header is None:
                        continue
                    if in_header != header_out:
                        print(
                            f"[FATAL] Inconsistent header in {p}.\n"
                            f"  got: {in_header}\n"
                            f"  exp: {header_out}"
                        )
                        sys.exit(4)
                    for row in reader:
                        writer.writerow(row)
    else:
        # Append mode: check header once, then append rows
        with combined_path.open("r", encoding="utf-8", newline="") as f_ex:
            existing_header = next(csv.reader(f_ex), None)
        if existing_header != header_out:
            print(
                f"[FATAL] Combined file header mismatch.\n"
                f"  got: {existing_header}\n"
                f"  exp: {header_out}"
            )
            sys.exit(4)

        with combined_path.open("a", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out)
            for p in per_file_labels:
                with Path(p).open("r", encoding="utf-8", newline="") as f_in:
                    reader = csv.reader(f_in)
                    in_header = next(reader, None)
                    if in_header != header_out:
                        print(
                            f"[FATAL] Inconsistent header in {p}.\n"
                            f"  got: {in_header}\n"
                            f"  exp: {header_out}"
                        )
                        sys.exit(4)
                    for row in reader:
                        writer.writerow(row)

    return combined_path


async def run_for_model(model_id: str, stop_event: asyncio.Event, mode: str):
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    short = model_short_name(model_id)

    # Input criterion-cache chunk files for THIS model
    part_dir = (
        OUTPUT_ROOT_BASE
        / short
        / "criteria_composed"
        / LANG
    )
    part_pattern = f"{short}_trecdl_{TREC_DL_YEAR}_{LANG}_criterion_cache_part{{n:03d}}.csv"

    # Output dirs for label CSVs & logs
    model_out_dir = OUTPUT_ROOT_BASE / short / "temp"
    model_logs_dir = LOG_ROOT_DIR / short
    model_out_dir.mkdir(parents=True, exist_ok=True)
    model_logs_dir.mkdir(parents=True, exist_ok=True)

    part_files = list(iter_part_files(START_PART, END_PART, part_dir, part_pattern))
    if not part_files:
        print(f"[INFO] No part files found in range in {part_dir}.")
        return

    run_id = timestamp_id()
    print(
        f"\n--- Running inference for model: {model_id} "
        f"(run_id={run_id}, LANG={LANG}, mode={mode}) ---"
    )
    print("[STOP] Press 'Q' at any time to stop after the current in-flight items].")

    per_file_out_dir = model_out_dir / f"_tmp_{run_id}_{model_id.replace(':','_')}_{LANG}"
    per_file_out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(min(6, len(part_files)))
    results: List[Dict[str, Any]] = []

    async def sem_task(p: Path):
        async with sem:
            if stop_event.is_set():
                return None
            return await label_single_part_file(
                p, model_id, prompt_template, run_id, per_file_out_dir, model_logs_dir, stop_event
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

    # verify consistent headers & collect per-file CSVs
    header_out_set = {tuple(r["header_out"]) for r in results}
    if len(header_out_set) != 1:
        print(f"[FATAL] Inconsistent output headers across parts: {header_out_set}")
        sys.exit(4)
    header_out = list(next(iter(header_out_set)))
    per_file_labels = [r["labels_csv"] for r in results]

    # write combined CSV (dynamic, no qid/pid_qrels enforcement)
    combined_path = write_combined_dynamic(
        per_file_labels=per_file_labels,
        header_out=header_out,
        model_short=short,
        lang=LANG,
        year=TREC_DL_YEAR,
        mode=mode,
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
        model_logs_dir / f"{run_id}_llm_logs_index_{short}_{LANG}.json",
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
            "labels_csv": str(combined_path),
            "log_json": "(see logs index)",
        },
    )

    print(f"[DONE] Model: {model_id} | Rows: {num_rows} | Combined: {combined_path}")
    print(
        f"[TOKENS] in={total_in:,} out={total_out:,} total={total_in + total_out:,}"
    )

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
