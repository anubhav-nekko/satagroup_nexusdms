#!/usr/bin/env python3
"""
test_pdf_pages.py  –  render each page of a PDF with PyMuPDF,
send the PNG bytes to Textract DetectDocumentText, print results.

Usage:
    python test_pdf_pages.py sample.pdf
"""

import sys, time, boto3, fitz
from pathlib import Path

REGION  = "us-east-1"          # change if necessary
textract = boto3.client("textract", region_name=REGION)

def ocr_image(png_bytes: bytes) -> list[str]:
    """Return a list of LINE texts from Textract DetectDocumentText."""
    resp = textract.detect_document_text(
        Document={"Bytes": png_bytes}
    )
    return [
        b["Text"] for b in resp.get("Blocks", [])
        if b["BlockType"] == "LINE" and "Text" in b
    ]

def main(pdf_path: Path):
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    total_lines = 0

    for page in doc:
        png_bytes = page.get_pixmap(dpi=300).tobytes("png")  # 300 dpi improves OCR
        lines = ocr_image(png_bytes)
        print(f"\nPage {page.number + 1} – {len(lines)} line(s)")
        for ln in lines[:10]:          # show first 10 lines
            print("  •", ln)
        if len(lines) > 10:
            print("  …")

        total_lines += len(lines)

    print(f"\n🟢 Textract returned {total_lines} total line(s) across {len(doc)} pages.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pdf_pages.py <file.pdf>")
        sys.exit(1)
    main(Path(sys.argv[1]))
