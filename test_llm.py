import requests, json

backend = "http://127.0.0.1:8000"

payload = {
    "selected_files": [],              # all files
    "selected_page_ranges": {},        # all pages
    "prompt": "Give me a one-line summary of the first document.",
    "top_k": 5,
    "last_messages": [],
    "web_search": False                # turn off while debugging
}

r = requests.post(f"{backend}/query_documents_with_page_range",
                  json=payload, timeout=60)
print(r.status_code, r.text)
