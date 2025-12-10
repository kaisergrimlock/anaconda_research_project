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
)

# ===== Bedrock / prompt config =====
cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)

PROMPT_TYPE = "criterion"
PROMPT_NAME = "composition"
PROMPT_FILE = Path(f"prompts/{PROMPT_TYPE}/{PROMPT_NAME}.txt")
LLM_COST_CSV = PROJECT_ROOT / "scripts" / "report" / "llm_cost.csv"

# ===== Experiment config =====
LANG = "eng_word"
START_PART = 1
END_PART = 6

TREC_DL_YEAR = "2022"
MODE = "append"

MODELS = ["openai.gpt-oss-20b-1:0"]

INFERENCE_CONFIG = {
    "maxTokens": 2000,
    "temperature": 0.0,
    "topP": 1.0
}

OUTPUT_ROOT_BASE = PROJECT_ROOT / "outputs" / "llm_label" / f"trec_dl_{TREC_DL_YEAR}"
LOG_ROOT_DIR = PROJECT_ROOT / "logs"

bump_field_limit()


# ===============================================================
# Helper Functions
# ===============================================================

def parse_llm_text_to_score(text: str) -> str:
    """Extract a 0–3 score from model output."""
    if not text:
        return ""
    text = text.strip()

    if text in {"0", "1", "2", "3"}:
        return text

    m = re.search(r"\b([0-3])\b", text)
    return m.group(1) if m else ""


def extract_text_from_resp(model_id: str, resp: dict) -> str:
    """
    Unified Bedrock text extractor.
    Supports:
      - 1-block responses  (most OpenAI models in Bedrock)
      - 2-block responses  (reasoning + short answer)
    """
    try:
        blocks = resp["output"]["message"]["content"]
        if not blocks:
            return ""
        # If 2 blocks, last is short answer.
        return blocks[-1].get("text", "") or ""
    except Exception:
        return ""


def extract_reasoning_from_resp(model_id: str, resp: dict) -> str:
    """Optional chain-of-thought extraction."""
    try:
        blocks = resp["output"]["message"]["content"]
        if len(blocks) > 1:
            return "\n".join(b.get("text", "") for b in blocks[:-1])
        return ""
    except Exception:
        return ""


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


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


def count_rows(path: Path) -> int:
    return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)


# ===============================================================
# Core Processing
# ===============================================================

def _label_single_part_file_blocking(
    part_csv: Path,
    model_id: str,
    prompt_template: str,
    run_id: str,
    out_dir: Path,
    logs_dir: Path,
    stop_event: Optional[asyncio.Event] = None,
):
    safe_model = model_id.replace(":", "_")
    header_in = _inspect_header(part_csv)

    # REQUIREMENT CHECKS
    required_cols = ["query", "passage" if LANG == "raw" else "passage_injected"]
    missing = [c for c in required_cols if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv} missing required {missing}")
        sys.exit(2)

    # OUTPUT HEADER
    if "llm_relevance" not in header_in:
        header_out = header_in + ["llm_relevance"]
    else:
        header_out = header_in[:]     # overwrite existing

    labels_path = out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    bedrock = boto3.client("bedrock-runtime", config=cfg)

    total_in = total_out = 0
    logs_json = []

    n_rows = count_rows(part_csv)
    print(f"[LOAD] {part_csv.name}: {n_rows} rows")

    for idx, row in enumerate(read_rows(part_csv), start=1):

        if stop_event and stop_event.is_set():
            print("\n[STOP] Early termination.")
            break

        row_out_map = dict(row)

        # fallback pid
        pid_resolved = (
            row.get("pid_resolved") or row.get("pid") or ""
        ).strip()

        query = row_out_map.get("query", "").strip()
        passage = pick_passage_for_lang(row_out_map, LANG)

        # Read criterion grades
        exactness = row_out_map.get("exactness", "").strip()
        topicality = row_out_map.get("topicality", "").strip()
        coverage = row_out_map.get("coverage", "").strip()
        contextual = row_out_map.get("contextuality", "").strip()

        # DEFAULT score = existing "relevance"
        score = (row_out_map.get("relevance", "") or "").strip()

        # Build prompt
        try:
            prompt = prompt_template.format(
                query=query,
                passage=passage,
                exactness=exactness,
                topicality=topicality,
                coverage=coverage,
                contextual=contextual,
            )
        except KeyError:
            prompt = prompt_template.format(query=query, passage=passage)

        SYSTEM_PROMPT = (
            "You are a search-quality rater.\n"
            "Given query and passage, output ONLY a relevance score 0-3.\n"
            "0 = irrelevant, 1 = related, 2 = highly relevant, 3 = perfectly relevant.\n"
        )

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": INFERENCE_CONFIG,
            "system": [{"text": SYSTEM_PROMPT}],
        }

        try:
            resp = bedrock.converse(**kwargs)
            txt = extract_text_from_resp(model_id, resp)
            parsed = parse_llm_text_to_score(txt)

            if parsed != "":
                score = parsed

            in_tok, out_tok = extract_usage = (
                resp.get("usage", {}).get("inputTokens", 0),
                resp.get("usage", {}).get("outputTokens", 0),
            )
            total_in += in_tok
            total_out += out_tok

            reasoning = extract_reasoning_from_resp(model_id, resp)

        except Exception as e:
            print(f"[ERR] API failed on row {idx}: {e}")
            txt = ""
            reasoning = ""

        # Build row in correct order
        row_values = [row_out_map.get(col, "") for col in header_out]

        # Set llm_relevance
        try:
            irel = header_out.index("llm_relevance")
            row_values[irel] = score
        except ValueError:
            print(f"[WARN] Missing llm_relevance in header_out")

        # Write
        with labels_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row_values)

        # LOG
        logs_json.append(
            {
                "qid": row_out_map.get("qid"),
                "pid_resolved": pid_resolved,
                "prompt": prompt,
                "response_text": txt,
                "llm_relevance": score,
                "reasoning": reasoning,
                "usage": {"input": in_tok, "output": out_tok},
            }
        )

        print(f"[{part_csv.name}] {idx}/{n_rows}  score={score}", end="\r")

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


# ===============================================================
# Combine Outputs
# ===============================================================

def write_combined(per_file_csvs, header_out, short, lang, year, mode, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / f"{short}_trecdl_{year}_{lang}_labels.csv"

    if mode == "replace" or not combined.exists():
        with combined.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header_out)

            for p in per_file_csvs:
                with open(p, "r", encoding="utf-8") as fin:
                    r = csv.reader(fin)
                    next(r)
                    w.writerows(r)
    else:
        # append mode
        with combined.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for p in per_file_csvs:
                with open(p, "r", encoding="utf-8") as fin:
                    r = csv.reader(fin)
                    next(r)
                    w.writerows(r)

    return combined


# ===============================================================
# Main Model Runner
# ===============================================================

async def run_for_model(model_id: str, stop_event: asyncio.Event, mode: str):
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    short = model_short_name(model_id)

    part_dir = OUTPUT_ROOT_BASE / short / "criteria_composed" / LANG
    part_pattern = f"{short}_trecdl_{TREC_DL_YEAR}_{LANG}_criterion_cache_part{{n:03d}}.csv"

    part_files = [
        part_dir / part_pattern.format(n=i)
        for i in range(START_PART, END_PART + 1)
        if (part_dir / part_pattern.format(n=i)).exists()
    ]

    if not part_files:
        print("[WARN] No part files found.")
        return

    run_id = timestamp_id()

    out_dir = OUTPUT_ROOT_BASE / short / "temp"
    logs_dir = LOG_ROOT_DIR / short
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    per_file_tmp = out_dir / f"_tmp_{run_id}"
    per_file_tmp.mkdir(exist_ok=True)

    sem = asyncio.Semaphore(4)
    results = []

    async def task(p):
        async with sem:
            if stop_event.is_set():
                return None
            return await label_single_part_file(
                p, model_id, prompt_template, run_id, per_file_tmp, logs_dir, stop_event
            )

    tasks = [asyncio.create_task(task(p)) for p in part_files]

    for t in asyncio.as_completed(tasks):
        r = await t
        if r:
            results.append(r)

    if not results:
        print("[INFO] No results to merge.")
        return

    header_out = results[0]["header_out"]
    per_files = [r["labels_csv"] for r in results]

    combined = write_combined(
        per_files, header_out, short, LANG, TREC_DL_YEAR, mode, out_dir
    )

    print(f"[COMBINED] {combined}")


# ===============================================================
# Entry
# ===============================================================

async def main():
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    listener = start_stop_key_listener(loop, stop_event)

    try:
        for model_id in MODELS:
            if stop_event.is_set():
                break
            await run_for_model(model_id, stop_event, MODE)
    finally:
        stop_event.set()


if __name__ == "__main__":
    asyncio.run(main())
