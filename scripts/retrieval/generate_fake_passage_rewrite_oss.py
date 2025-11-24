#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

import boto3
from botocore.config import Config
import pandas as pd

# =========================
# Paths / config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]   # scripts/... -> scripts -> project root

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / "gpt-oss-20b"
    / "gpt-oss-20b_trec_dl_2023_nr.csv"
)

# Prompt template file (with {query} and/or {passage} placeholders)
PROMPT_FILE = PROJECT_ROOT / "prompts" / "distracting-passage-gen-neg.txt"

# Bedrock / model config
BEDROCK_REGION = "us-west-2"
MODEL_ID = "openai.gpt-oss-20b-1:0"  # hard-coded for GPT-OSS
PROMPT_SUFFIX = PROMPT_FILE.stem.split("-")[-1]  # e.g. 'neg'

# dynamic query_nr column name based on prompt suffix
QUERY_NR_COL = f"query_{PROMPT_SUFFIX}"

OUTPUT_CSV = (
    PROJECT_ROOT
    / "retrieved"
    / "trec_dl_2023"
    / "gpt-oss-20b"
    / f"{PROMPT_SUFFIX}"
    / "all_topics_trecdl_2023_part1.csv"
)

# Target output column order (use dynamic query_{x} name)
OUTPUT_COLS = [
    "qid",
    "query",
    "pid_qrels",
    "pid_resolved",
    "passage",
    "relevance",
    QUERY_NR_COL,
    "passage_injected",
]

# Inference config (for converse)
INFERENCE_CONFIG: Dict[str, Any] = {
    "maxTokens": 3000,
    "temperature": 1.0,
    "topP": 0.5,
}

cfg = Config(
    region_name=BEDROCK_REGION,
    connect_timeout=10,
    read_timeout=300,
    retries={"max_attempts": 8, "mode": "standard"},
)
bedrock = boto3.client("bedrock-runtime", config=cfg)

# Load prompt template once
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


def build_prompt(query: str, passage: str) -> str:
    """
    Safely insert query and passage into the template by literal replacement.
    Avoid using str.format() so any other braces in the template (JSON, examples)
    are not interpreted as format fields.
    """
    q = str(query)
    p = str(passage)
    tmpl = PROMPT_TEMPLATE

    # literal replace only the intended placeholders
    if "{query}" in tmpl:
        tmpl = tmpl.replace("{query}", q)
    if "{passage}" in tmpl:
        tmpl = tmpl.replace("{passage}", p)

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


def call_bedrock_incorrect_passage(query: str, passage: str) -> Tuple[str, int, int, str, str]:
    """
    Call GPT-OSS-20B via bedrock.converse and return:
      - cleaned_answer (for passage_injected)
      - input_tokens
      - output_tokens
      - prompt_used
      - raw_answer_text (as returned from content[1].text)
    Both the query and passage are used when building the prompt.
    """
    prompt = build_prompt(query=query, passage=passage)

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
# Main logic
# =========================
def main() -> None:
    print(f"[INFO] Reading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # If input still has 'query_nr', move it to the dynamic query_{suffix} name
    if "query_nr" in df.columns and QUERY_NR_COL not in df.columns:
        df[QUERY_NR_COL] = df.pop("query_nr")

    # Ensure required columns exist (except passage_injected, which we create/overwrite)
    for col in OUTPUT_COLS:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column: {col}")

    # Validate that both 'query' and 'passage' exist and have no blanks
    for col in ("query", "passage"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV")

        blank_mask = df[col].astype(str).str.strip() == ""
        if blank_mask.any():
            bad_rows = df.loc[
                blank_mask,
                ["qid", "pid_qrels", "pid_resolved", "query", "passage"],
            ].copy()
            OUT_DIR = OUTPUT_CSV.parent
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            missing_path = OUT_DIR / f"missing_{col}_rows.csv"
            bad_rows.to_csv(missing_path, index=False, encoding="utf-8")
            sample_qids = bad_rows["qid"].dropna().astype(str).tolist()[:10]
            raise ValueError(
                f"Found {len(bad_rows)} rows with empty '{col}'. "
                f"Wrote details to: {missing_path}. Example qids: {sample_qids}"
            )

    print(f"[INFO] Using prompt file: {PROMPT_FILE}")

    # Ensure column exists for injected passage
    if "passage_injected" not in df.columns:
        df["passage_injected"] = ""
    else:
        # We'll overwrite any existing values
        df["passage_injected"] = df["passage_injected"].fillna("")

    total = len(df)
    print(f"[INFO] Generating incorrect passages for {total} rows using both 'query' and 'passage'")

    total_in = 0
    total_out = 0

    for idx, row in df.iterrows():
        query_text = str(row["query"]).strip()
        passage_text = str(row["passage"]).strip()

        answer = ""
        in_tok = out_tok = 0

        try:
            # call_bedrock_incorrect_passage returns:
            # (cleaned_answer, input_tokens, output_tokens, prompt_used, raw_answer_text)
            answer, in_tok, out_tok, prompt_used, raw_answer_text = call_bedrock_incorrect_passage(
                query_text,
                passage_text,
            )
        except Exception as e:
            raise RuntimeError(f"Bedrock call failed at row {idx} (qid={row.get('qid')}): {e}") from e

        # Write answer to DataFrame
        df.at[idx, "passage_injected"] = answer

        total_in += in_tok
        total_out += out_tok

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(
                f"[INFO] Processed {idx + 1}/{total} rows | "
                f"tokens in/out totals = {total_in}/{total_out}"
            )

    # Reorder and trim to the desired output format
    out_df = df[OUTPUT_COLS]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Wrote injected CSV to: {OUTPUT_CSV}")
    print(f"[INFO] Columns: {list(out_df.columns)}")
    print(f"[TOKENS] input={total_in} output={total_out} total={total_in + total_out}")


if __name__ == "__main__":
    main()
