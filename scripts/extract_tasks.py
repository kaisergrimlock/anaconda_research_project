#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from botocore.config import Config

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bedrock_client import (
    make_bedrock_runtime_client,
    converse_prompt,
)

# Configuration
# Default model (you can change it to meta.llama3-8b-instruct-v1:0 or others)
MODEL_ID = "qwen.qwen3-32b-v1:0" 
PROMPT_FILE = PROJECT_ROOT / "prompts" / "task_extraction.txt"
INPUT_FILE = PROJECT_ROOT / "prompts" / "label" / "utility.txt" # Define your input text file here
SAFE_MODEL_ID = MODEL_ID.replace(":", "_").replace("/", "_")
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "task_extraction" / f"extracted_tasks_{SAFE_MODEL_ID}.json"

def extract_tasks_from_text(text: str, model_id: str = MODEL_ID) -> str:
    cfg = Config(
        region_name="us-west-2",
        connect_timeout=10,
        read_timeout=300,
        retries={"max_attempts": 8, "mode": "standard"},
    )
    
    bedrock = make_bedrock_runtime_client(cfg)
    
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Missing prompt file at: {PROMPT_FILE}")
        
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    
    # Combine system prompt with the input text
    prompt = f"{system_prompt}\n\nMessage content:\n{text}"
    
    inference_config = {"maxTokens": 2000, "temperature": 0.0, "topP": 1.0}
    
    result = converse_prompt(
        bedrock_runtime_client=bedrock,
        model_id=model_id,
        prompt=prompt,
        inference_config=inference_config
    )
    
    return result.text

def main():
    parser = argparse.ArgumentParser(description="Extract tasks from text using Bedrock LLM.")
    parser.add_argument("--input", type=str, default=str(INPUT_FILE), help="Path to input text file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="Path to output file")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Bedrock Model ID")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)
        
    input_text = input_path.read_text(encoding="utf-8").strip()
        
    if not input_text:
        print("Error: Input file is empty.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Extracting tasks from {input_path.name} using model: {args.model}...", file=sys.stderr)
    try:
        output = extract_tasks_from_text(input_text, args.model)
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        
        print(f"Successfully saved extracted tasks to: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error calling LLM: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
