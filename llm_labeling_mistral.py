import json, re
from pathlib import Path
import boto3

# -------- Helpers --------
def to_text(x):
    """Ensure we always pass a string to the model."""
    if x is None:
        return ""
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def coerce_json_text(txt: str) -> dict:
    """Extract the first JSON object from the model text (handles extra prose or code fences)."""
    t = txt.strip()
    if t.startswith("```"):
        # strip generic fenced block
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        t = m.group(0)
    return json.loads(t)

# -------- Inputs --------
prompt_template = Path("prompt.txt").read_text(encoding="utf-8")
docs_file = Path("outputs/topic_and_docs_q524332.txt").read_text(encoding="utf-8")

# Extract query metadata
qid_match = re.search(r"Query ID:\s*(\d+)", docs_file)
qid = qid_match.group(1) if qid_match else ""
topic_match = re.search(r"Topic:\s*(.+)", docs_file)
topic = topic_match.group(1).strip() if topic_match else ""

# Extract doc blocks: (doc_num, docid, passage_json)
doc_blocks = re.findall(
    r'Doc\s+(\d+):\s+(\S+).*?Passage:\n({.*?})\n',
    docs_file,
    re.DOTALL
)

# -------- Bedrock (Mistral) --------
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-2")
model_id = "mistral.mistral-large-2402-v1:0"

llm_labels = []
for idx, (doc_num, docid, passage_json) in enumerate(doc_blocks, 1):
    # Parse passage JSON, prefer "contents" (your file format), else fallback
    try:
        pobj = json.loads(passage_json)
        passage_text = pobj.get("contents") or pobj.get("content") or pobj.get("text") or passage_json
    except Exception:
        passage_text = passage_json

    prompt = prompt_template.format(query=to_text(topic), passage=to_text(passage_text))

    # Mistral schema for invoke_model: send {"prompt": "..."} and read outputs[0].text
    body = {
        "prompt": to_text(prompt),
        "max_tokens": 300,
        "temperature": 0,
        "top_p": 1.0,
    }

    resp = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    out = json.loads(resp["body"].read())
    # Expected Mistral text here:
    text = out["outputs"][0]["text"]
    try:
        scores = coerce_json_text(text)
        relevance = scores.get("O", "")
    except Exception:
        relevance = ""

    llm_labels.append((docid, str(relevance)))

# -------- Write TSV --------
safe_model = model_id.replace(":", "_").replace("/", "_").replace("\\", "_")
out_path = Path(f"outputs/llm_labels_q{qid}_{safe_model}.tsv")
out_path.parent.mkdir(exist_ok=True)

with out_path.open("w", encoding="utf-8", newline="") as f:
    f.write(f"# Query ID: {qid}\n")
    f.write("docid\trelevance\n")
    for docid, relevance in llm_labels:
        f.write("{0}\t{1}\n".format(docid, relevance))

print(f"Wrote: {out_path.resolve()}")
