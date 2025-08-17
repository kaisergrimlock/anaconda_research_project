import boto3, json

session = boto3.Session(profile_name="khoi", region_name="us-west-2")
bedrock = session.client("bedrock-runtime")

model_id = "anthropic.claude-3-haiku-20240307-v1:0"
prompt = "Tell me a short story about a space-faring cat."

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 300,
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
# Text lives under content[0].text
print(out["content"][0]["text"])
