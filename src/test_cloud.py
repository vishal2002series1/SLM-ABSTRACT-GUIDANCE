import boto3

def test_bedrock_opus():
    print("Testing connection to AWS Bedrock (Opus 4.8)...")

    # Initialize the Bedrock Runtime client
    # It will automatically pick up your AWS CLI credentials
    client = boto3.client("bedrock-runtime", region_name="us-east-1") 

    model_id = "us.anthropic.claude-opus-4-8"

    messages = [
        {
            "role": "user",
            "content": [{"text": "Write a 1-sentence abstract mathematical representation of a sorting algorithm."}]
        }
    ]

    try:
        # Call Opus 4.8 via Bedrock
        response = client.converse(
            modelId=model_id,
            messages=messages #,
            #inferenceConfig={"temperature": 0.0}
        )

        # Extract the text and the token usage (critical for your paper!)
        output_text = response['output']['message']['content'][0]['text']
        token_usage = response['usage']

        print(f"\nResponse from Opus 4.8:\n{output_text}\n")
        print(f"Token Usage Tracking: {token_usage}")

    except Exception as e:
        print(f"AWS Bedrock Error: {e}")

if __name__ == "__main__":
    test_bedrock_opus()