import math
import boto3, json, re
from pathlib import Path

# Read prompt template
prompt_template = Path("prompt.txt").read_text(encoding="utf-8")
#TREC_DL colelction
# Read topic and docs file
docs_file = Path("outputs/topic_and_docs_q524332.txt").read_text(encoding="utf-8")

# Extract topic
qid_match = re.search(r"Query ID:\s*(\d+)", docs_file)
qid = qid_match.group(1) if qid_match else ""
# Extract topic text
topic_match = re.search(r"Topic:\s*(.+)", docs_file)
topic = topic_match.group(1).strip() if topic_match else ""
topic_safe = topic.replace(" ", "_").replace("'", "")

# Extract docids and passages
doc_blocks = re.findall(r'Doc (\d+): ([^\s]+).*?Passage:\n({.*?})\n', docs_file, re.DOTALL)

# Setup Bedrock
#session = boto3.Session(profile_name="khoi", region_name="us-west-2")
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-2")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

llm_labels = []
for idx, (doc_num, docid, passage_json) in enumerate(doc_blocks, 1):
    try:
        passage_obj = json.loads(passage_json)
        passage_text = passage_obj.get("contents", "")
    except Exception:
        passage_text = passage_json  # fallback: raw

    prompt = prompt_template.format(query=topic, passage=passage_text)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }

    resp = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    out = json.loads(resp["body"].read())
    try:
        scores = json.loads(out["content"][0]["text"])
        llm_score = scores.get("O", "")
    except Exception:
        llm_score = ""
    llm_labels.append((docid, llm_score))

def round_half_up(n):
    return math.floor(n + 0.5)

# Write to TSV
safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
out_path = Path(f"outputs/llm_labels_q{qid}_{safe_model}.tsv")
with out_path.open("w", encoding="utf-8", newline="") as f:
    f.write(f"# Query ID: {qid}\n")
    f.write("docid\trelevance\n")
    for docid, llm_score in llm_labels:
        rounded_score = round_half_up(float(llm_score)) if llm_score else ""
        f.write(f"{docid}\t{llm_score}\n")

print(f"Wrote: {out_path}")


