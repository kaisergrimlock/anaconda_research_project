import math
import boto3, json, re
from pathlib import Path

# --- Read prompt & inputs ---
prompt_template = Path("prompt.txt").read_text(encoding="utf-8")
docs_file = Path("outputs/topic_and_docs_q524332_modified_eng.txt").read_text(encoding="utf-8")

# --- Extract topic + qid ---
qid_match = re.search(r"Query ID:\s*(\d+)", docs_file)
qid = qid_match.group(1) if qid_match else ""

topic_match = re.search(r"Topic:\s*(.+)", docs_file)
topic = topic_match.group(1).strip() if topic_match else ""
topic_safe = topic.replace(" ", "_").replace("'", "")

# --- Extract doc blocks (num, docid, JSON passage) ---
doc_blocks = re.findall(
    r'Doc (\d+): ([^\s]+).*?Passage:\n({.*?})\n', docs_file, re.DOTALL
)

# --- Bedrock setup ---
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

# --- List of models to evaluate ---
models = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "openai.gpt-oss-20b-1:0"
]

# --- Generation configuration ---
inference_config = {
    "maxTokens": 2000,
    "temperature": 0.0,
    "topP": 1.0,
}

def round_half_up(n):
    try:
        return math.floor(float(n) + 0.5)
    except (TypeError, ValueError):
        return ""

for model_id in models:
    print(f"\n--- Running inference for model: {model_id} ---")
    llm_labels = []
    response_log = []   # Collect all responses for this model
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, (doc_num, docid, passage_json) in enumerate(doc_blocks, 1):
        # Parse passage JSON; fallback to raw text if parsing fails
        try:
            passage_obj = json.loads(passage_json)
            passage_text = passage_obj.get("contents", "")
        except Exception:
            passage_text = passage_json

        # Format the prompt
        prompt = prompt_template.format(query=topic, passage=passage_text)

        # Build message payload for Converse
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        # Build request payload
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        # Call Bedrock Converse API
        resp = bedrock.converse(**kwargs)

        # Extract model response text
        try:
            if(model_id.startswith("openai.")):
                text = resp["output"]["message"]["content"][1]["text"]
            else:
                text = resp["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = ""

        # Parse output as JSON to get "O" field if possible
        try:
            scores = json.loads(text)
            llm_score = scores.get("O", "")
        except Exception:
            llm_score = ""

        llm_labels.append((docid, llm_score))

        # --- Log the response ---
        response_log.append({
            "doc_num": doc_num,
            "docid": docid,
            "prompt": prompt,
            "response_text": text,
            "full_response": resp
        })

        # --- Token counting ---
        usage = resp.get("usage", {})  # usage is at the top level
        total_input_tokens  += int(usage.get("inputTokens", 0) or 0)
        total_output_tokens += int(usage.get("outputTokens", 0) or 0)

    # --- Write results for this model ---
    safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    out_path = Path(f"outputs/llm_label/llm_labels_q{qid}_{safe_model}_modified_eng.tsv")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# Query ID: {qid}\n")
        f.write("docid\trelevance\n")
        for docid, llm_score in llm_labels:
            f.write(f"{docid}\t{llm_score}\n")

    # --- Write response log to JSON ---
    log_path = Path(f"outputs/logs/llm_responses_q{qid}_{safe_model}_modified_eng.json")
    with log_path.open("w", encoding="utf-8") as log_file:
        json.dump(response_log, log_file, indent=2, ensure_ascii=False)

    print(f"Wrote results to: {out_path}")
    print(f"Saved detailed responses to: {log_path}")
    print(f"Total input tokens] used by {model_id}: {total_input_tokens}")
    print(f"Total output tokens used by {model_id}: {total_output_tokens}")

