import boto3

def run_opus_baseline():
    print("--- Running Standalone Opus 4.8 Baseline Test ---")
    
    # 1. Read the raw sandbox files
    try:
        with open("sandbox/target_app.py", "r") as f:
            target_code = f.read()
        with open("sandbox/test_suite.py", "r") as f:
            test_code = f.read()
    except FileNotFoundError:
        print("Error: Sandbox files not found. Run the orchestrator once to generate them.")
        return

    # 2. Construct the heavy "Standard Agent" prompt
    prompt = "You are an autonomous software engineer. The following Python code has a bug.\n\nFile: target_app.py\n\n" + target_code + "\n\nFile: test_suite.py\n\n" + test_code + "\n\nTask: Rewrite target_app.py so that it passes all tests. Output the full rewritten code."

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "us.anthropic.claude-opus-4-8"

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("Sending full codebase context to Opus 4.8...")
    response = client.converse(modelId=model_id, messages=messages)

    usage = response['usage']
    print("\n================ BASELINE METRICS ================")
    print(f"Input Tokens (Full Context): {usage['inputTokens']}")
    print(f"Output Tokens (Full Rewrite): {usage['outputTokens']}")
    print(f"Total Tokens Consumed: {usage['totalTokens']}")
    print("==================================================")

if __name__ == "__main__":
    run_opus_baseline()