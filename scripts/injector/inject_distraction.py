#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

import boto3
from botocore.config import Config
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))
from helper import allow_huge_csv_fields

allow_huge_csv_fields()

# =========================
# Paths / config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / "gpt-oss-20b"
    / "gpt-oss-20b_trec_dl_2023_nr.csv"
)

PROMPT_FILE = PROJECT_ROOT / "prompts" / "paraphrase" / "distracting-passage-gen-hypo.txt"
prompt_suffix = PROMPT_FILE.stem.split("-")[-1]

# Cache file for unique query -> generated passage mappings
# Store cache in the same directory as this script (injector folder)
CACHE_DIR = THIS_FILE.parent / "cache"
CACHE_FILE = CACHE_DIR / f"{prompt_suffix}_query_response_cache.csv"

# Bedrock / model config (hard-coded for GPT-OSS)
MODEL_ID = "openai.gpt-oss-20b-1:0"
assert MODEL_ID.startswith("openai.gpt-oss-20b"), "generate_fake_passage_rewrite_oss.py is intended for gpt-oss-20b only"
BEDROCK_REGION = "us-west-2"

OUTPUT_CSV = (
    PROJECT_ROOT
    / "retrieved"
    / "trec_dl_2023"
    / f"para_{prompt_suffix}"
    / "all_topics_trecdl_2023_part1.csv"
)


# Target output column order
OUTPUT_COLS = [
    "qid",
    "query",
    "pid_qrels",
    "pid_resolved",
    "passage",
    "relevance",
    "query_nr",  # This will be replaced
    "passage_injected",
]

# Inference config (for converse)
INFERENCE_CONFIG: Dict[str, Any] = {
    "maxTokens": 1000,
    "temperature": 1.0,
    "topP": 0.6,
}

cfg = Config(
    region_name=BEDROCK_REGION,
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)
bedrock = boto3.client("bedrock-runtime", config=cfg)

# Load prompt template once (with validation)
if not PROMPT_FILE.exists():
    prompts_dir = PROJECT_ROOT / "prompts"
    print(f"[ERROR] Prompt file not found: {PROMPT_FILE}")
    if prompts_dir.exists():
        available = sorted(prompts_dir.glob("**/*.txt"))
        print(f"[INFO] Available prompt files in {prompts_dir}:")
        for p in available:
            print(f"  - {p.relative_to(prompts_dir)}")
    else:
        print(f"[ERROR] Prompts directory does not exist: {prompts_dir}")
    sys.exit(1)

PROMPT_TEMPLATE = PROMPT_FILE.read_text(encoding="utf-8")


# =========================
# Helpers
# =========================
def strip_reasoning_tags(text: str) -> str:
    """
    Safety helper: if the visible answer accidentally contains <reasoning>...</reasoning>,
    strip those blocks and keep the rest.
    """
    return re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()


def build_prompt(query: str) -> str:
    """Fill the template with {query} only (no {passage})."""
    q = str(query)
    tmpl = PROMPT_TEMPLATE

    # literal replace only the {query} placeholder
    if "{query}" in tmpl:
        tmpl = tmpl.replace("{query}", q)

    return tmpl


def extract_text_from_resp(model_id: str, resp: dict) -> str:
    """
    For openai.* reasoning models, Bedrock 'converse' returns:
      resp["output"]["message"]["content"][0]["text"] -> reasoning
      resp["output"]["message"]["content"][1]["text"] -> visible answer

    For others, fall back to the first block.
    """
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        return resp["output"]["message"]["content"][0]["text"]
    except Exception:
        return ""


def usage_from_resp(resp: dict) -> Tuple[int, int]:
    """Extract token usage from a Converse response."""
    u = resp.get("usage", {}) or {}
    return int(u.get("inputTokens", 0) or 0), int(u.get("outputTokens", 0) or 0)


def get_question_from_row(row: pd.Series) -> str:
    """Return the question text from the 'query' column."""
    if "query" not in row.index:
        raise ValueError("Required column 'query' not found in input dataframe")
    q_val = row["query"]
    if pd.isna(q_val) or not str(q_val).strip():
        raise ValueError(f"Empty 'query' for qid={row.get('qid', '<unknown>')} (abort)")
    return str(q_val).strip()


def call_bedrock_incorrect_passage(question: str) -> Tuple[str, int, int, str, str]:
    """
    Call GPT-OSS-20B via bedrock.converse and return:
      - cleaned_answer (for passage_injected)
      - input_tokens
      - output_tokens
      - prompt_used
      - raw_answer_text (as returned from content[1].text)
    """
    prompt = build_prompt(question)

    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    resp = bedrock.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig=INFERENCE_CONFIG,
    )

    raw_answer = extract_text_from_resp(MODEL_ID, resp) or ""
    cleaned_answer = strip_reasoning_tags(raw_answer)
    in_tok, out_tok = usage_from_resp(resp)

    return cleaned_answer, in_tok, out_tok, prompt, raw_answer


# =========================
# Cache helpers
# =========================
def load_cache() -> Dict[str, str]:
    """Load query -> response cache from CSV. Returns empty dict if file doesn't exist."""
    if not CACHE_FILE.exists():
        return {}
    cache = {}
    try:
        import csv
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("query") and row.get("response"):
                    cache[row["query"]] = row["response"]
    except Exception as e:
        print(f"[WARN] Failed to load cache: {e}")
    return cache


def save_cache(cache: Dict[str, str]) -> None:
    """Save query -> response cache to CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import csv
        with CACHE_FILE.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["query", "response"])
            writer.writeheader()
            for query, response in cache.items():
                writer.writerow({"query": query, "response": response})
        print(f"[INFO] Saved cache with {len(cache)} entries to {CACHE_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save cache: {e}")


# =========================
# Main logic
# =========================
def main() -> None:
    print(f"[INFO] Reading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Ensure required columns exist (except passage_injected, which we create/overwrite)
    missing = [
        c for c in OUTPUT_COLS
        if c != "passage_injected" and c not in df.columns
    ]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Validate that 'query' column exists and has no blanks
    if "query" not in df.columns:
        raise ValueError("Required column 'query' not found in input CSV")

    blank_mask = df["query"].astype(str).str.strip() == ""
    if blank_mask.any():
        bad_rows = df.loc[blank_mask, ["qid", "pid_qrels", "pid_resolved", "query"]].copy()
        OUT_DIR = OUTPUT_CSV.parent
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        missing_path = OUT_DIR / "missing_query_rows.csv"
        bad_rows.to_csv(missing_path, index=False, encoding="utf-8")
        sample_qids = bad_rows["qid"].dropna().astype(str).tolist()[:10]
        raise ValueError(
            f"Found {len(bad_rows)} rows with empty 'query'. "
            f"Wrote details to: {missing_path}. Example qids: {sample_qids}"
        )

    print(f"[INFO] Using prompt file: {PROMPT_FILE}")

    # Load existing cache
    query_cache = load_cache()
    print(f"[INFO] Loaded cache with {len(query_cache)} entries")

    # Ensure column exists
    if "passage_injected" not in df.columns:
        df["passage_injected"] = ""
    else:
        df["passage_injected"] = df["passage_injected"].fillna("")

    # Find unique queries that are not yet in cache
    unique_queries = df["query"].unique()
    queries_to_generate = [q for q in unique_queries if q not in query_cache]

    print(f"[INFO] Found {len(unique_queries)} unique queries")
    print(f"[INFO] {len(queries_to_generate)} queries need generation")

    total_in = 0
    total_out = 0

    # Generate responses for missing queries
    for query_idx, query in enumerate(queries_to_generate, start=1):
        try:
            answer, in_tok, out_tok, prompt_used, raw_answer_text = call_bedrock_incorrect_passage(query)
            query_cache[query] = answer
            total_in += in_tok
            total_out += out_tok

            if query_idx % 5 == 0 or query_idx == len(queries_to_generate):
                print(
                    f"[INFO] Generated {query_idx}/{len(queries_to_generate)} unique responses | "
                    f"tokens in/out totals = {total_in}/{total_out}"
                )
        except Exception as e:
            print(f"[ERROR] Bedrock call failed for query: {query[:50]}... :: {e}")
            query_cache[query] = ""

    # Save updated cache
    save_cache(query_cache)

    # Apply cached responses to all rows
    print(f"[INFO] Applying cached responses to all {len(df)} rows...")
    for idx, row in df.iterrows():
        query = row["query"]
        passage = (row.get("passage", "") or "").strip()
        response = query_cache.get(query, "")
        
        # Append response to the existing passage (no extra newlines)
        if response:
            passage_injected = passage + " " + response
        else:
            passage_injected = passage
        
        df.at[idx, "passage_injected"] = passage_injected

        # Replace query_* columns with query_para_{prompt_suffix}
        for col in row.index:
            if col.startswith("query_"):
                df.at[idx, f"query_para_{prompt_suffix}"] = row[col]

    # Prepare new output columns
    new_output_cols = []
    for col in OUTPUT_COLS:
        if col.startswith("query_"):
            new_col_name = f"query_para_{prompt_suffix}"
            new_output_cols.append(new_col_name)
        else:
            new_output_cols.append(col)

    # Reorder and trim to the desired output format
    out_df = df[new_output_cols]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Wrote injected CSV to: {OUTPUT_CSV}")
    print(f"[INFO] Columns: {list(out_df.columns)}")
    print(f"[TOKENS] input={total_in} output={total_out} total={total_in + total_out}")


if __name__ == "__main__":
    main()
