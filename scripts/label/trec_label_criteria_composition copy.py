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
from scripts.log_helpers import timestamp_id

# ===== use the same Bedrock helper as the working script =====
from scripts.bedrock_client import (
    make_bedrock_runtime_client,
    converse_prompt,
)

# ===============================================================
# Config
# ===============================================================

TREC_DL_YEAR = "2021"
LANG = "ar"                       # e.g. raw/eng/vi/ru/...
START_PART = 1
END_PART = 6
MODE = "replace"                  # replace|append
MODELS = ["meta.llama3-8b-instruct-v1:0"]

CRITERIA = ["contextuality", "coverage", "exactness", "topicality"]
RELEVANCE_COL = "relevance"
PASSAGE_COL_OUT = "passage" if LANG == "raw" else "passage_injected"

FORCE_REBUILD_CACHE = False

PROMPT_TYPE = "criterion"
PROMPT_NAME = "composition_2"
PROMPT_FILE = Path(f"prompts/{PROMPT_TYPE}/{PROMPT_NAME}.txt")
OUTPUT_SUFFIX = "crit_2"

cfg = Config(
    region_name="us-west-2",
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)

INFERENCE_CONFIG = {
    "maxTokens": 2000,
    "temperature": 0.0,
    "topP": 1.0,
}

OUTPUT_ROOT_BASE = PROJECT_ROOT / "outputs" / "llm_label" / f"trec_dl_{TREC_DL_YEAR}"
LOG_ROOT_DIR = PROJECT_ROOT / "logs"

bump_field_limit()

# ===============================================================
# Part A: Build cache from per-criterion CSVs
# ===============================================================

RowKey = Tuple[str, str]
RowDict = Dict[RowKey, Dict[str, Any]]


def criterion_dir_for_short(short: str) -> Path:
    return OUTPUT_ROOT_BASE / short / "criterion"


def cache_dir_for_short(short: str) -> Path:
    return OUTPUT_ROOT_BASE / short / "criteria_composed" / LANG


def cache_prefix_for_short(short: str) -> str:
    return f"{short}_trecdl_{TREC_DL_YEAR}_{LANG}_criterion_cache"


def find_file_for_criterion(crit_dir: Path, criterion: str) -> Path:
    pattern = f"*_{LANG}_{criterion}_labels.csv"
    matches = list(crit_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {crit_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files found for criterion '{criterion}': {matches}")
    return matches[0]


def load_criterion_into_dict(
    data: RowDict,
    csv_path: Path,
    criterion_name: str,
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

            if LANG == "raw":
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
                    PASSAGE_COL_OUT: passage_val,
                }
            else:
                if not data[key].get(PASSAGE_COL_OUT):
                    data[key][PASSAGE_COL_OUT] = passage_val

            data[key][criterion_name] = criterion_score
            data[key][RELEVANCE_COL] = relevance_val


def build_combined_dict_for_short(short: str) -> RowDict:
    crit_dir = criterion_dir_for_short(short)
    if not crit_dir.exists():
        raise FileNotFoundError(f"Criterion directory not found: {crit_dir}")

    combined: RowDict = {}
    for criterion in CRITERIA:
        p = find_file_for_criterion(crit_dir, criterion)
        print(f"[CACHE] Loading {criterion} from {p.name}")
        load_criterion_into_dict(combined, p, criterion)

    return combined


def write_cache_parts_for_short(short: str, data: RowDict, chunk_size: int = 500) -> List[Path]:
    cache_dir = cache_dir_for_short(short)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prefix = cache_prefix_for_short(short)
    fieldnames = ["qid", "pid", "query", PASSAGE_COL_OUT] + CRITERIA + [RELEVANCE_COL]

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
        print(f"[CACHE] Wrote {part_path.name}  rows {start}..{end - 1}")

    return out_paths


def ensure_cache_exists(short: str) -> None:
    cache_dir = cache_dir_for_short(short)
    prefix = cache_prefix_for_short(short)
    existing = sorted(cache_dir.glob(f"{prefix}_part*.csv")) if cache_dir.exists() else []

    if existing and not FORCE_REBUILD_CACHE:
        print(f"[CACHE] Found {len(existing)} existing cache parts in {cache_dir}")
        return

    if existing and FORCE_REBUILD_CACHE:
        print("[CACHE] FORCE_REBUILD_CACHE=True, deleting old cache parts...")
        for p in existing:
            p.unlink(missing_ok=True)

    print(f"[CACHE] Building cache parts for {short} (LANG={LANG})...")
    combined = build_combined_dict_for_short(short)
    print(f"[CACHE] Total (qid,pid) pairs: {len(combined)}")
    write_cache_parts_for_short(short, combined)


# ===============================================================
# Part B: Bedrock composition over cache parts
# ===============================================================

def parse_llm_text_to_score(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text in {"0", "1", "2", "3"}:
        return text
    m = re.search(r"\b([0-3])\b", text)
    return m.group(1) if m else ""


def extract_text_from_helper_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "response_text", "answer", "output_text"):
            val = result.get(key)
            if isinstance(val, str):
                return val
    return str(result)


def extract_usage_from_helper_result(result: Any) -> tuple[int, int]:
    if isinstance(result, dict):
        usage = result.get("usage", {})
        if isinstance(usage, dict):
            return int(usage.get("input", 0) or usage.get("inputTokens", 0) or 0), int(
                usage.get("output", 0) or usage.get("outputTokens", 0) or 0
            )
        return int(result.get("input_tokens", 0) or 0), int(result.get("output_tokens", 0) or 0)
    return 0, 0


def extract_reasoning_from_helper_result(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("reasoning", "thoughts", "analysis"):
            val = result.get(key)
            if isinstance(val, str):
                return val
    return ""


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


def signal_stop(loop, stop_event):
    if loop and stop_event:
        loop.call_soon_threadsafe(stop_event.set)
    elif stop_event:
        stop_event.set()


def _label_single_part_file_blocking(
    part_csv: Path,
    model_id: str,
    prompt_template: str,
    run_id: str,
    out_dir: Path,
    logs_dir: Path,
    stop_event: Optional[asyncio.Event] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
):
    safe_model = model_id.replace(":", "_")
    header_in = _inspect_header(part_csv)

    required_cols = CRITERIA
    missing = [c for c in required_cols if c not in header_in]
    if missing:
        print(f"[FATAL] {part_csv} missing required {missing}")
        sys.exit(2)

    header_out = header_in + ["llm_relevance"] if "llm_relevance" not in header_in else header_in[:]

    labels_path = out_dir / f"{part_csv.stem}_labels_{safe_model}.csv"
    ensure_csv_with_header(labels_path, header_out)

    # ===== use the same helper as the working script =====
    bedrock = make_bedrock_runtime_client(cfg)

    total_in = 0
    total_out = 0
    logs_json = []

    n_rows = count_rows(part_csv)
    print(f"[LOAD] {part_csv.name}: {n_rows} rows")

    for idx, row in enumerate(read_rows(part_csv), start=1):
        if stop_event and stop_event.is_set():
            print("\n[STOP] Early termination.")
            break

        row_out_map = dict(row)

        pid_resolved = (row.get("pid_resolved") or row.get("pid") or "").strip()

        exactness = (row_out_map.get("exactness", "") or "").strip()
        topicality = (row_out_map.get("topicality", "") or "").strip()
        coverage = (row_out_map.get("coverage", "") or "").strip()
        contextual = (row_out_map.get("contextuality", "") or "").strip()

        score = (row_out_map.get("relevance", "") or "").strip()

        prompt = prompt_template.format(
            exactness=exactness,
            topicality=topicality,
            coverage=coverage,
            contextual=contextual,
        )

        SYSTEM_PROMPT = (
            "You are a search-quality rater.\n"
            "Given criterion scores, output ONLY a relevance score 0-3.\n"
            "0 = irrelevant, 1 = related, 2 = highly relevant, 3 = perfectly relevant.\n"
        )

        try:
            result = converse_prompt(
                bedrock=bedrock,
                model_id=model_id,
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                inference_config=INFERENCE_CONFIG,
            )

            txt = extract_text_from_helper_result(result)
            parsed = parse_llm_text_to_score(txt)
            if parsed != "":
                score = parsed

            in_tok, out_tok = extract_usage_from_helper_result(result)
            total_in += in_tok
            total_out += out_tok

            reasoning = extract_reasoning_from_helper_result(result)

        except Exception as e:
            print(f"[ERR] API failed on row {idx}: {e}")
            signal_stop(loop, stop_event)
            raise RuntimeError(f"Bedrock call failed for {part_csv.name} row {idx}") from e

        row_values = [row_out_map.get(col, "") for col in header_out]
        try:
            irel = header_out.index("llm_relevance")
            row_values[irel] = score
        except ValueError:
            pass

        with labels_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row_values)

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


def write_combined(per_file_csvs, header_out, short, lang, year, mode, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lang_tag = f"{lang}_{OUTPUT_SUFFIX}"
    combined = out_dir / f"{short}_trecdl_{year}_{lang_tag}_labels.csv"

    def norm_row_len(r: list[str]) -> list[str]:
        if len(r) < len(header_out):
            return r + [""] * (len(header_out) - len(r))
        if len(r) > len(header_out):
            return r[: len(header_out)]
        return r

    def make_key(r: list[str]) -> str:
        r = norm_row_len(r)
        qid = (r[qid_i] or "").strip()
        pid = (r[pid_i] or "").strip()
        return f"{qid}|{pid}"

    def is_blank_or_nan_like(v: str) -> bool:
        s = (v or "").strip()
        if s == "":
            return True
        return s.lower() in {"nan", "none", "null"}

    def should_preserve_old(old: str, new: str) -> bool:
        old_s = (old or "").strip()
        new_s = (new or "").strip()
        return (old_s != "") and is_blank_or_nan_like(new_s)

    def load_incoming() -> dict[str, list[str]]:
        incoming: dict[str, list[str]] = {}
        for p in per_file_csvs:
            with open(p, "r", encoding="utf-8") as fin:
                r = csv.reader(fin)
                in_header = next(r, None)
                if in_header is None:
                    continue
                if list(in_header) != list(header_out):
                    raise ValueError(
                        f"Inconsistent header in {p}.\n"
                        f"  got: {in_header}\n"
                        f"  exp: {header_out}"
                    )
                for row in r:
                    row = norm_row_len(row)
                    incoming[make_key(row)] = row
        return incoming

    if "qid" not in header_out or "pid" not in header_out:
        raise ValueError(f"Cannot replace without qid+pid columns. header_out={header_out}")

    qid_i = header_out.index("qid")
    pid_i = header_out.index("pid")
    rel_i = header_out.index("llm_relevance") if "llm_relevance" in header_out else None

    incoming = load_incoming()

    if mode == "replace":
        if not combined.exists():
            with combined.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header_out)
                for row in incoming.values():
                    w.writerow(row)
            return combined

        existing_header = _inspect_header(combined)
        if existing_header != header_out:
            raise ValueError(
                f"Combined file header mismatch.\n"
                f"  got: {existing_header}\n"
                f"  exp: {header_out}"
            )

        tmp_path = combined.with_suffix(".tmp.csv")
        used_keys: set[str] = set()

        with combined.open("r", encoding="utf-8", newline="") as f_in, tmp_path.open(
            "w", encoding="utf-8", newline=""
        ) as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            _ = next(reader, None)
            writer.writerow(header_out)

            for old_row in reader:
                old_row = norm_row_len(old_row)
                k = make_key(old_row)
                if k in incoming:
                    new_row = norm_row_len(incoming[k])
                    used_keys.add(k)
                    if rel_i is not None and should_preserve_old(old_row[rel_i], new_row[rel_i]):
                        new_row[rel_i] = old_row[rel_i]
                    writer.writerow(new_row)
                else:
                    writer.writerow(old_row)

            for k, row in incoming.items():
                if k not in used_keys:
                    writer.writerow(norm_row_len(row))

        tmp_path.replace(combined)
        return combined

    if not combined.exists():
        with combined.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header_out)
            for row in incoming.values():
                w.writerow(row)
        return combined

    with combined.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in incoming.values():
            w.writerow(row)

    return combined


async def run_for_model(model_id: str, stop_event: asyncio.Event, mode: str):
    loop = asyncio.get_running_loop()
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    short = model_short_name(model_id)

    ensure_cache_exists(short)

    part_dir = cache_dir_for_short(short)
    part_pattern = f"{cache_prefix_for_short(short)}_part{{n:03d}}.csv"

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

    per_file_tmp = out_dir / f"_tmp_{run_id}"
    per_file_tmp.mkdir(exist_ok=True)

    sem = asyncio.Semaphore(4)
    results = []

    async def task(p: Path):
        async with sem:
            if stop_event.is_set():
                return None
            return await label_single_part_file(
                p, model_id, prompt_template, run_id, per_file_tmp, logs_dir, stop_event, loop
            )

    tasks = [asyncio.create_task(task(p)) for p in part_files]
    pending = set(tasks)

    while pending:
        if stop_event.is_set():
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            break

        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=0.1,
        )

        for t in done:
            try:
                r = t.result()
                if r:
                    results.append(r)
            except Exception as e:
                print(f"[FATAL] Task failed: {e}")
                stop_event.set()
                for pt in pending:
                    pt.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                return

    if stop_event.is_set():
        print("[ABORT] Stopping before merge because at least one part failed.")
        return

    if not results:
        print("[INFO] No results to merge.")
        return

    header_out = results[0]["header_out"]
    per_files = [r["labels_csv"] for r in results]

    combined = write_combined(per_files, header_out, short, LANG, TREC_DL_YEAR, mode, out_dir)
    print(f"[COMBINED] {combined}")


async def main():
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    start_stop_key_listener(loop, stop_event)

    for model_id in MODELS:
        if stop_event.is_set():
            break
        await run_for_model(model_id, stop_event, MODE)


if __name__ == "__main__":
    asyncio.run(main())