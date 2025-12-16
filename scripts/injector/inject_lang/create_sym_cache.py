#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Optional

import boto3
from botocore.config import Config

# ===============================================================
# Path setup
# ===============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.injector.helper import allow_huge_csv_fields

# ===============================================================
# Config
# ===============================================================
TREC_DL_YEAR = "2022"
LANG = "eng"  # used only for cache naming

MODELS = ["meta.llama3-8b-instruct-v1:0"]
GLOB_PATTERN = "*.csv"

INPUT_DIR = PROJECT_ROOT / f"retrieved/trec_dl_{TREC_DL_YEAR}/judged"

CACHE_ROOT = PROJECT_ROOT / f"retrieved/trec_dl_{TREC_DL_YEAR}/translate_cache"
SYN_CACHE_DIR = CACHE_ROOT / "sym_lang"
SYN_MAP_FILE = SYN_CACHE_DIR / f"symnonyms_map_{LANG}.csv"

PROMPT_QUERY_FILE = PROJECT_ROOT / "prompts" / "symnonyms.txt"
PROMPT_WORD_FILE  = PROJECT_ROOT / "prompts" / "symnonyms_word.txt"

QUERY_COL = "query"

AWS_REGION = "us-west-2"
INFERENCE_CONFIG = {
    "maxTokens": 300,
    "temperature": 0.0,
    "topP": 1.0,
}

SYSTEM_PROMPT = (
    "You are an SEO assistant.\n"
    "Return ONLY valid JSON.\n"
)

PRINT_EACH_NEW = True

# ===============================================================

allow_huge_csv_fields()

cfg = Config(
    region_name=AWS_REGION,
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)
bedrock = boto3.client("bedrock-runtime", config=cfg)

# ===============================================================
# Utilities
# ===============================================================
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def collect_unique_queries(files: Iterable[Path]) -> Set[str]:
    out: Set[str] = set()
    for f in files:
        with f.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                q = (row.get(QUERY_COL) or "").strip()
                if q:
                    out.add(q)
    return out


# ===============================================================
# Prompt rendering (brace-safe)
# ===============================================================
def _render_prompt(template: str, key: str, value: str) -> str:
    sentinel = "<<<PLACEHOLDER>>>"
    s = template.replace(f"{{{key}}}", sentinel)
    s = s.replace("{", "{{").replace("}", "}}")
    s = s.replace(sentinel, f"{{{key}}}")
    return s.format(**{key: value})


def render_query_prompt(tpl: str, query: str) -> str:
    return _render_prompt(tpl, "query", query)


def render_keyword_prompt(tpl: str, keyword: str) -> str:
    return _render_prompt(tpl, "keyword", keyword)


# ===============================================================
# Bedrock + parsing
# ===============================================================
def bedrock_call(model_id: str, text: str) -> str:
    resp = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": text}]}],
        inferenceConfig=INFERENCE_CONFIG,
        system=[{"text": SYSTEM_PROMPT}],
    )
    blocks = resp.get("output", {}).get("message", {}).get("content", []) or []
    return "".join(b.get("text", "") for b in blocks if "text" in b).strip()


def extract_json(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s:
        return None

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None

    blob = m.group(0)
    blob = blob.replace("“", '"').replace("”", '"').replace("’", "'")
    blob = re.sub(r'(\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1 "\2":', blob)

    try:
        return json.loads(blob)
    except Exception:
        return None


# ===============================================================
# Cache I/O
# ===============================================================
def load_syn_map(path: Path) -> Dict[str, Tuple[str, str]]:
    m: Dict[str, Tuple[str, str]] = {}
    if not path.exists():
        return m
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            q = (row.get("query") or "").strip()
            k = (row.get("keyword") or "").strip()
            s = (row.get("symnonyms") or "").strip()
            if q:
                m[q] = (k, s)
    return m


def save_syn_map(path: Path, m: Dict[str, Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["query", "keyword", "symnonyms"])
        w.writeheader()
        for q, (k, s) in m.items():
            w.writerow({"query": q, "keyword": k, "symnonyms": s})


# ===============================================================
# LLM fetchers
# ===============================================================
def fetch_from_query(model: str, query: str, tpl: str) -> Tuple[str, str]:
    out = bedrock_call(model, render_query_prompt(tpl, query))
    obj = extract_json(out)
    if not obj:
        return ("", "")
    return (
        str(obj.get("keyword", "") or "").strip(),
        str(obj.get("symnonyms", "") or "").strip(),
    )


def fetch_from_keyword(model: str, keyword: str, tpl: str) -> str:
    out = bedrock_call(model, render_keyword_prompt(tpl, keyword))
    obj = extract_json(out)
    if not obj:
        return ""
    return str(obj.get("symnonyms", "") or "").strip()


# ===============================================================
# Main
# ===============================================================
def run_for_model(model_id: str):
    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit("No input CSV files found")

    tpl_query = read_text(PROMPT_QUERY_FILE)
    tpl_word  = read_text(PROMPT_WORD_FILE)

    SYN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    queries = collect_unique_queries(files)
    syn_map = load_syn_map(SYN_MAP_FILE)

    print(f"\n=== BUILD SYN CACHE | MODEL={model_id} | QUERIES={len(queries)} ===")
    print(f"Existing cache entries: {len(syn_map)}")

    for i, q in enumerate(sorted(queries), 1):
        if q not in syn_map:
            kw, syns = fetch_from_query(model_id, q, tpl_query)
            syn_map[q] = (kw, syns)
            src = "QUERY"
        else:
            kw, syns = syn_map[q]
            if kw and not syns:
                syns = fetch_from_keyword(model_id, kw, tpl_word)
                syn_map[q] = (kw, syns)
                src = "KEYWORD"
            else:
                continue

        if PRINT_EACH_NEW:
            preview = " ".join(syns.split())[:200]
            print(f"[{src}] {i:04d} kw='{kw}' syn='{preview}'")
            print(f"        q='{q}'")

    save_syn_map(SYN_MAP_FILE, syn_map)
    print(f"\n[DONE] Saved {len(syn_map)} entries → {SYN_MAP_FILE}")


def main():
    for m in MODELS:
        run_for_model(m)


if __name__ == "__main__":
    main()
