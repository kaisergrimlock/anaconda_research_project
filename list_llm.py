import boto3

# Create Bedrock client
bedrock = boto3.client("bedrock", region_name="us-west-2")  # Change region if needed

# List available models
response = bedrock.list_foundation_models()

print("Available Bedrock Models:\n")
for model in response.get("modelSummaries", []):
    print(f"Model ID     : {model['modelId']}")
    print(f"Provider     : {model['providerName']}")
    print(f"Input Modalities : {', '.join(model.get('inputModalities', []))}")
    print(f"Output Modalities: {', '.join(model.get('outputModalities', []))}")
    print(f"Model Name   : {model['modelName']}")
    print("-" * 50)
