#!/usr/bin/env python3
"""
test_llm.py  –  minimal Bedrock / Anthropic connectivity check

❯ python test_llm.py                # uses the default prompt
❯ python test_llm.py "Write a haiku about GPUs"
❯ python test_llm.py -p "Hi!" -m anthropic.claude-3-sonnet-20240229-v1:0 -r us-east-1
"""

import argparse, json, sys, time
from datetime import datetime
import boto3, botocore
import tiktoken                                     # pip install tiktoken

DEFAULT_PROMPT   = "Respond with the word: pong"
DEFAULT_MODEL_ID = "arn:aws:bedrock:us-east-1:343218220592:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
DEFAULT_REGION   = "us-east-1"
MAX_REPLY_TOKENS = 256                              # you can raise this

enc = tiktoken.get_encoding("cl100k_base")          # ≈ Anthropic tokeniser


def print_banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Quick Bedrock LLM checker (supports Anthropic models)")
    ap.add_argument("-p", "--prompt", default=DEFAULT_PROMPT,
                    help="Prompt to send to the model")
    ap.add_argument("-m", "--model-id", default=DEFAULT_MODEL_ID,
                    help="Bedrock modelId (full ARN or short form)")
    ap.add_argument("-r", "--region", default=DEFAULT_REGION,
                    help="AWS region that hosts the model")
    args = ap.parse_args()

    prompt = args.prompt
    model  = args.model_id
    region = args.region

    # 1️⃣  count prompt tokens so you know where you stand
    n_tok = len(enc.encode(prompt))
    print_banner(f"Prompt tokens: {n_tok}")

    # 2️⃣  build the standard Anthropic payload
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_REPLY_TOKENS,             # reply length
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }

    # 3️⃣  call Bedrock
    client = boto3.client("bedrock-runtime", region_name=region)
    try:
        t0   = time.perf_counter()
        resp = client.invoke_model(
            modelId   = model,
            body      = json.dumps(body),
            contentType = "application/json",
            accept      = "application/json",
        )
        dt_ms = (time.perf_counter() - t0) * 1_000
        result = json.loads(resp["body"].read())

        print_banner(f"✅ Model replied in {dt_ms:,.0f} ms")
        print(result["content"][0]["text"].strip())

    except botocore.exceptions.ClientError as err:
        # Bedrock sticks the *real* error message in the JSON body even for 500s
        maybe_msg = err.response.get("Error", {}).get("Message")
        print_banner("🛑 Bedrock error")
        print("Status code :", err.response["ResponseMetadata"]["HTTPStatusCode"])
        print("Error type  :", err.response['Error'].get("Code"))
        if maybe_msg:
            print("\nFull message from Bedrock:")
            print(maybe_msg)
        else:
            print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
