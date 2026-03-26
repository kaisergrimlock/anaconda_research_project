# scripts/bedrock_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import boto3
from botocore.config import Config
import boto3
from botocore.exceptions import ProfileNotFound

AWS_REGION = "us-west-2"


@dataclass(frozen=True)
class BedrockResult:
    text: str
    reasoning: str
    score: str
    input_tokens: int
    output_tokens: int
    raw_response: Dict[str, Any]


def make_bedrock_runtime_client(cfg: Config):
    try:
        session = boto3.Session(profile_name="rmit", region_name=AWS_REGION)
    except ProfileNotFound:
        session = boto3.Session(region_name=AWS_REGION)

    """Create a Bedrock Runtime client using the provided botocore Config."""
    return session.client("bedrock-runtime")


def parse_llm_text_to_score(text: str) -> str:
    """
    Parse the model's text and return the 'O' score as a string.
    Accepts either raw JSON or JSON embedded inside surrounding text/code fences.
    """
    if not text:
        return ""

    text = text.strip()

    # First try direct parse in case the whole response is already JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("O"), int) and 0 <= parsed["O"] <= 3:
            return str(parsed["O"])
        if isinstance(parsed, list):
            for item in parsed:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("O"), int)
                    and 0 <= item["O"] <= 3
                ):
                    return str(item["O"])
    except Exception:
        pass

    # Fallback: find JSON-like objects inside the text
    import re

    candidates = re.findall(r'\{[^{}]*\}', text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and isinstance(parsed.get("O"), int) and 0 <= parsed["O"] <= 3:
                return str(parsed["O"])
        except Exception:
            continue

    return ""

    
def extract_text_from_resp(model_id: str, resp: dict) -> str:
    """
    Return the main text content from the model's response.
    Your current assumption:
      openai.* => content[0] is reasoning, content[1] is JSON output
      others   => content[0] is output
    """
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][1]["text"]
        return resp["output"]["message"]["content"][0]["text"]
    except Exception:
        return ""


def extract_reasoning_from_resp(model_id: str, resp: dict) -> str:
    """Return the model's hidden/chain-of-thought reasoning block when present (openai.*)."""
    try:
        if model_id.startswith("openai."):
            return resp["output"]["message"]["content"][0].get("text", "")
        return ""
    except Exception:
        return ""


def usage_from_resp(resp: dict) -> Tuple[int, int]:
    u = resp.get("usage", {}) or {}
    return int(u.get("inputTokens", 0) or 0), int(u.get("outputTokens", 0) or 0)


def build_converse_kwargs(model_id: str, prompt: str, inference_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the kwargs for bedrock.converse().
    Kept separate so your calling code doesn't touch Bedrock request schema.
    """
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    return {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": inference_config,
    }


def converse_prompt(
    bedrock_runtime_client,
    *,
    model_id: str,
    prompt: str,
    inference_config: Dict[str, Any],
) -> BedrockResult:
    """
    Single Bedrock call + parse response into (text, reasoning, score, usage).
    """
    resp = bedrock_runtime_client.converse(**build_converse_kwargs(model_id, prompt, inference_config))

    text = extract_text_from_resp(model_id, resp) or ""
    reasoning = extract_reasoning_from_resp(model_id, resp) or ""
    score = parse_llm_text_to_score(text)

    in_tok, out_tok = usage_from_resp(resp)

    return BedrockResult(
        text=text,
        reasoning=reasoning,
        score=score,
        input_tokens=in_tok,
        output_tokens=out_tok,
        raw_response=resp,
    )
