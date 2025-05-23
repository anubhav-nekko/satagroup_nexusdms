# test_pdf.py

"""
Usage:
    python test_pdf.py sample.pdf

AWS credentials:
    The script uses your default AWS profile / environment variables.
    Make sure the IAM identity has textract:DetectDocumentText,
    textract:StartDocumentTextDetection and textract:GetDocumentTextDetection.

Author: you
"""

import sys, os, time, json, boto3
from pathlib import Path

REGION = "us-east-1"   # change if needed
textract = boto3.client("textract", region_name=REGION)

# ────────────────────────────────────────────────────────────────────────────
def detect_small_pdf(pdf_bytes: bytes):
    """≤ 5 MB → synchronous DetectDocumentText"""
    resp = textract.detect_document_text(Document={"Bytes": pdf_bytes})
    return resp

def detect_large_pdf(s3_bucket: str, s3_key: str):
    """> 5 MB → async StartDocumentTextDetection + polling"""
    start = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}}
    )
    job_id = start["JobId"]
    print(f"Started Textract job {job_id}")

    while True:
        resp = textract.get_document_text_detection(JobId=job_id)
        status = resp["JobStatus"]
        if status in ("IN_PROGRESS", "SUCCEEDED"):
            if status == "SUCCEEDED":
                return resp
            time.sleep(2)   # wait a bit and poll again
        else:
            raise RuntimeError(f"Textract job failed: {status}")

def main(pdf_path: Path):
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    size_mb = pdf_path.stat().st_size / 1_048_576
    print(f"PDF size: {size_mb:.2f} MB")

    if size_mb <= 5:
        # read bytes and call sync API
        resp = detect_small_pdf(pdf_path.read_bytes())
    else:
        # upload to a temp S3 object (same bucket every time)
        bucket = "textract-demo-upload"
        key    = f"test/{pdf_path.name}"
        print(f"Uploading to s3://{bucket}/{key} …")
        s3 = boto3.client("s3", region_name=REGION)
        s3.upload_file(str(pdf_path), bucket, key)
        resp = detect_large_pdf(bucket, key)

    # ── print the lines we got back ─────────────────────────────────────────
    lines = [
        block["Text"]
        for block in resp.get("Blocks", [])
        if block["BlockType"] == "LINE" and "Text" in block
    ]
    if not lines:
        print("⚠️  Textract returned ZERO lines.")
    else:
        print(f"\n🟢 Textract returned {len(lines)} line(s):\n")
        for ln in lines[:50]:           # first 50 lines
            print("•", ln)
        if len(lines) > 50:
            print("… (truncated)")

# ── entry-point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pdf.py <file.pdf>")
        sys.exit(1)

    main(Path(sys.argv[1]))
