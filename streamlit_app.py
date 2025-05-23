# streamlit_app.py

system_message = """
# You are an advanced HR-tech assistant that specialises in parsing and analysing
# resumes, CVs and Job Descriptions (JDs) for an HR-consultancy client.
# ...
"""

prompt_library = {
    "custom": ""
}

import os
import json
from typing import List, Dict, Tuple
import re
import requests
import tempfile
import pickle
import fitz  # PyMuPDF
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import boto3
from tavily import TavilyClient
from streamlit_cookies_manager import EncryptedCookieManager
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from pptx import Presentation
import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
from io import BytesIO
from PIL import Image
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import copy
import uuid
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")  # ~15 % error on Claude 3
CTX_LIMIT = 200_000
BACKEND_URL = "http://127.0.0.1:8000"  # or your server endpoint

def n_tokens(text: str) -> int:
    return len(ENC.encode(text))

def used_tokens(sys_msg, rag_context):
    total  = len(ENC.encode(sys_msg))
    total += len(ENC.encode(rag_context))
    return total

def post_json(route: str, payload: dict, **kwargs):
    """Helper to post JSON to the backend."""
    url = f"{BACKEND_URL}{route}"
    r = requests.post(url, json=payload, timeout=300, **kwargs)
    r.raise_for_status()
    return r.json()

if "Authenticator" not in st.session_state:
    st.session_state["Authenticator"] = None
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "logged_out" not in st.session_state:
    st.session_state["logged_out"] = False

# Instantiate the Cookie Manager at the very top
cookies = EncryptedCookieManager(
    prefix="chola-nexusdms/",
    password="nexusdms"
)

if not cookies.ready():
    st.spinner("Loading cookies...")
    st.stop()

def remove_last_line(text):
    lines = text.splitlines()
    filtered_lines = [
        line
        for line in lines
        if 'python' not in line.lower() and 'plotly' not in line.lower() and "code" not in line.lower()
    ]
    return "\n".join(filtered_lines)

def load_dict_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

REGION = "us-east-1"
TAVILY_API = "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"

FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_STORE_PATH = "metadata_store.pkl"
s3_bucket_name = "satagroup-test"
users_file = "users.json"

def display_logo():
    st.write("<Your Logo Here>")

textract_client = boto3.client('textract', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)

def get_presigned_url(file_key, expiration=3600):
    """Generate a pre-signed URL for the S3 object."""
    return s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': s3_bucket_name, 'Key': file_key},
        ExpiresIn=expiration
    )

def save_chat_history(chat_history, blob_name="chat_history.json"):
    try:
        local_file_path = "chat_history.json"
        with open(local_file_path, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        s3_client.upload_file(local_file_path, s3_bucket_name, blob_name)
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")

def load_chat_history(blob_name="chat_history.json"):
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=blob_name)
        if 'Contents' in response:
            local_file_path = "chat_history.json"
            s3_client.download_file(s3_bucket_name, blob_name, local_file_path)
            chat_history = json.load(open(local_file_path, encoding="utf-8"))
            if not isinstance(chat_history, dict):
                chat_history = {}
            return chat_history
        return {}
    except Exception as e:
        st.error(f"Failed to load chat history: {str(e)}")
        return {}

def file_exists_in_blob(file_name):
    """Check if a file with the same name exists in S3."""
    try:
        s3_client.head_object(Bucket=s3_bucket_name, Key=file_name)
        return True
    except Exception as e:
        if hasattr(e, 'response') and e.response['Error']['Code'] == '404':
            return False
        else:
            raise e

def upload_to_blob_storage(local_file_path, bucket_name, s3_key):
    try:
        with open(local_file_path, "rb") as data:
            s3_client.upload_fileobj(data, bucket_name, s3_key)
    except Exception as e:
        st.error(f"Failed to upload file to S3: {str(e)}")

def download_from_blob_storage(s3_bucket_name, s3_key, local_file_path):
    """Download a file from S3, or return False if not found."""
    try:
        with open(local_file_path, "wb") as file:
            s3_client.download_fileobj(s3_bucket_name, s3_key, file)
        return True
    except Exception as e:
        if hasattr(e, 'response') and e.response['Error']['Code'] == '404':
            print(f"File not found in S3: {s3_key}")
            return False
        else:
            print(f"Failed to download {s3_key}: {str(e)}")
            return False

def create_word_doc(text):
    doc = Document()
    doc.add_heading("Chat Answer", level=1)
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def generate_titan_embeddings(text):
    try:
        embedding = mpnet_model.encode(text, normalize_embeddings=True)
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def call_llm_api(system_message, user_query):
    prompt = system_message + user_query
    client = boto3.client("bedrock-runtime", region_name=REGION)

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(native_request)
        )
        body_str = response["body"].read().decode("utf-8")
        body = json.loads(body_str)
        return body['content'][0]['text']
    except Exception as e:
        return f"An error occurred: {str(e)}"

dimension = 768
faiss_index = faiss.IndexFlatL2(dimension)
metadata_store = []

def save_index_and_metadata():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    try:
        upload_to_blob_storage(FAISS_INDEX_PATH, s3_bucket_name, os.path.basename(FAISS_INDEX_PATH))
        upload_to_blob_storage(METADATA_STORE_PATH, s3_bucket_name, os.path.basename(METADATA_STORE_PATH))
    except Exception as e:
        print(f"Error uploading index or metadata to Blob Storage: {str(e)}")

def load_index_and_metadata():
    global faiss_index, metadata_store

    index_blob_name = os.path.basename(FAISS_INDEX_PATH)
    metadata_blob_name = os.path.basename(METADATA_STORE_PATH)

    index_downloaded = download_from_blob_storage(s3_bucket_name, index_blob_name, FAISS_INDEX_PATH)
    metadata_downloaded = download_from_blob_storage(s3_bucket_name, metadata_blob_name, METADATA_STORE_PATH)

    if index_downloaded and metadata_downloaded:
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_STORE_PATH, "rb") as f:
                metadata_store = pickle.load(f)
            print("Index and metadata loaded from Storage.")
        except Exception as e:
            print(f"Failed to load index or metadata: {str(e)}")
            faiss_index = faiss.IndexFlatL2(dimension)
            metadata_store = []
    else:
        print("Index or metadata not found in S3. Initializing new.")
        faiss_index = faiss.IndexFlatL2(dimension)
        metadata_store = []

#########################################
# ... your other local functions (extract_text_from_pdf, etc.) ...
#########################################

def login():
    display_logo()
    st.title("Ready to Dive In? Sign In!")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and password == USERS[username]:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
            cookies["username"] = username
            cookies.save()

            Authenticator = stauth.Authenticate(credentials, cookie_name='nexusdms/', key='abcdefgh', cookie_expiry_days=0)
            st.session_state["Authenticator"] = Authenticator
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")

def logout():
    authenticator = st.session_state.get("Authenticator")
    if authenticator is not None:
        try:
            logout_button = authenticator.logout('Log Out', 'sidebar')
        except KeyError:
            logout_button = True
        except Exception as err:
            st.error(f'Unexpected exception during logout: {err}')
            return
    else:
        logout_button = True

    if logout_button:
        st.session_state["logged_out"] = True
        st.session_state["logged_in"] = False
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.session_state["Authenticator"] = None

        if "username" in cookies:
            cookies["username"] = ""
            cookies.save()
        st.rerun()

USERS = load_dict_from_json(users_file)
credentials = {"usernames": {}}
for user in USERS:
    credentials["usernames"][user] = {
        "name": user,
        "password": USERS[user]
    }

def user_input():
    prompt_options = list(prompt_library.keys())
    selected_prompt = st.selectbox("Select a predefined prompt:", prompt_options, index=0)

    default_text = prompt_library[selected_prompt]
    user_message = st.text_area("Enter your message:", value=default_text, height=150)
    ol1, ol2, ol3, ol4, ol5, ol6, ol7, ol8 = st.columns(8)
    with ol8:
       submit = st.button("Send")
    if submit and user_message.strip():
        st.session_state.user_message = ""
        return user_message.strip()
    return None

def main():
    if not st.session_state["authenticated"]:
        cookie_username = cookies.get("username")
        if cookie_username != "" and cookie_username is not None:
            st.session_state["authenticated"] = True
            st.session_state["username"] = cookie_username
            st.session_state["logged_in"] = True

    if not st.session_state["authenticated"]:
        login()
        return

    display_logo()
    st.title("Document Query Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'selected_page_ranges' not in st.session_state:
        st.session_state.selected_page_ranges = {}
    if 'file_summaries' not in st.session_state:
        st.session_state.file_summaries = {}
    if "rename_mode" not in st.session_state:
        st.session_state["rename_mode"] = None
    if "share_chat_mode" not in st.session_state:
        st.session_state["share_chat_mode"] = False
    if "share_chat_conv" not in st.session_state:
        st.session_state["share_chat_conv"] = None
    if "share_chat_conv_id" not in st.session_state:
        st.session_state["share_chat_conv_id"] = None

    available_usernames = list(USERS.keys())

    current_user = st.session_state["username"]
    load_index_and_metadata()

    st.sidebar.header(f"Hello `{current_user}`")
    if st.sidebar.button("Log Out"):
        logout()

    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose an option", ["Query Documents", "Upload Documents", "File Manager", "Usage Monitoring"])

    # Updated Upload Documents block
    if option == "Upload Documents":
        st.header("Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload one or more documents",
            type=["pdf", "jpg", "jpeg", "png", "doc", "docx", "pptx", "xlsx", "csv"],
            accept_multiple_files=True
        )

        # if uploaded_files:
        #     for f in uploaded_files:
        #         with st.spinner(f"Uploading & indexing **{f.name}** …"):
        #             files = {"file": (f.name, f.getvalue())}
        #             data  = {"owner": current_user}  # store the correct user

        #             try:
        #                 resp = requests.post(
        #                     f"{BACKEND_URL}/upload_document",
        #                     files=files,
        #                     data=data,
        #                     timeout=900
        #                 )
        #             except requests.exceptions.RequestException as err:
        #                 st.error(f"Network error while uploading {f.name}: {err}")
        #                 continue

        #         if resp.status_code == 200:
        #             info = resp.json()  # {status:"done", filename, pages_indexed}
        #             st.success(
        #                 f"Indexed **{info['pages_indexed']}** pages "
        #                 f"from **{info['filename']}** ✔️"
        #             )
        #             load_index_and_metadata()
        #         else:
        #             st.error(f"Upload failed for {f.name}: {resp.text}")
        if uploaded_files:
            for f in uploaded_files:
                with st.spinner(f"Uploading & indexing **{f.name}** …"):
                    resp = requests.post(
                        f"{BACKEND_URL}/upload_document",
                        files={"file": (f.name, f.getvalue())},
                        data={"owner": current_user},          # owner now compulsory
                        timeout=900
                    )
                if resp.status_code == 200:
                    info = resp.json()
                    st.success(f"Indexed **{info['pages_indexed']}** pages from **{info['filename']}** ✔️")
                    load_index_and_metadata()                 # immediately reload
                else:
                    st.error(f"Upload failed for {f.name}: {resp.text}")

    elif option == "File Manager":
        st.header("My Uploaded Files")
        current_user = st.session_state.get("username", "unknown")
        # available_files = list({
        #     md["filename"]
        #     for md in metadata_store
        #     if md.get("owner") == current_user or current_user in md.get("shared_with", [])
        # })

        available_files = sorted({
            rec["filename"] for rec in metadata_store
            if rec.get("owner") == current_user or current_user in rec.get("shared_with", [])
        })

        if available_files:
            for i, fname in enumerate(available_files):
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    st.write(fname)
                with col2:
                    if st.button("Delete", key=f"del_{fname}_{i}"):
                        # implement your delete_file function
                        pass
        else:
            st.sidebar.info("No files uploaded yet.")

        # Sharing UI
        st.sidebar.header("Share a File")
        file_to_share = st.sidebar.selectbox("Select a file to share", available_files)
        share_with = st.sidebar.multiselect("Select user(s) to share with", options=available_usernames)
        if st.sidebar.button("Share File"):
            for md in metadata_store:
                if md["filename"] == file_to_share and md.get("owner") == current_user:
                    md.setdefault("shared_with", []).extend(share_with)
                    md["shared_with"] = list(set(md["shared_with"]))
                    st.success(f"Shared {file_to_share} with {', '.join(share_with)}")
            save_index_and_metadata()

    elif option == "Query Documents":
        st.header("Query Documents")
        st.sidebar.header("Settings")
        llm_model = st.sidebar.selectbox("Choose Your Model", ["Claude 3"])
        if st.sidebar.button("New Chat"):
            st.session_state.current_conversation_id = None
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.session_state.sources = []
            st.session_state.selected_files = []
            st.session_state.selected_page_ranges = {}
            st.success("Started a new conversation.")

        # toggles from the user
        web_search = st.sidebar.toggle("Enable Web Search")
        top_k = st.sidebar.slider("Select Top-K Results", min_value=1, max_value=100, value=50, step=1)

        current_user = st.session_state.get("username", "unknown")
        # available_files = list({
        #     md["filename"]
        #     for md in metadata_store
        #     if md.get("owner") == current_user or current_user in md.get("shared_with", [])
        # })

        available_files = sorted({
            rec["filename"] for rec in metadata_store
            if rec.get("owner") == current_user or current_user in rec.get("shared_with", [])
        })

        if available_files:
            st.session_state.selected_files = st.multiselect(
                "Select files to include in the query:",
                available_files,
                default=st.session_state.selected_files
            )
            if len(st.session_state.selected_files) > 4:
                st.warning("For best results, select a maximum of 4 files.")

            page_ranges = {}
            for file in st.session_state.selected_files:
                # find min & max page for that file
                all_pages = [r["page"] for r in metadata_store if r["filename"] == file]
                if all_pages:
                    minp, maxp = min(all_pages), max(all_pages)
                else:
                    minp, maxp = (1, 1)
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    start_page = st.number_input(
                        f"Start page for {file}",
                        min_value=minp,
                        max_value=maxp,
                        value=minp,
                        key=f"start_{file}"
                    )
                with col2:
                    end_page = st.number_input(
                        f"End page for {file}",
                        min_value=minp,
                        max_value=maxp,
                        value=maxp,
                        key=f"end_{file}"
                    )
                page_ranges[file] = (start_page, end_page)

            st.session_state.selected_page_ranges = page_ranges

        st.sidebar.header("Previous Conversations")
        user_conversations = st.session_state.chat_history.get(current_user, [])
        unique_conversations = sorted(user_conversations, key=lambda x: x.get("timestamp", ""), reverse=True)
        # ... your existing code for listing/saving/deleting/sharing conversations ...

        # Show the chat messages ...
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                msg_time = message.get("time", "")
                role_title = message["role"].title()
                st.markdown(f"**`[{role_title} @ {msg_time}]`**\n\n{message['content']}")
                with st.expander("Show Copyable Text"):
                    st.code(message["content"], language="text")

        user_message = user_input()
        if user_message:
            ist_timezone = pytz.timezone("Asia/Kolkata")
            timestamp_now = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M:%S")

            st.session_state.messages.append({
                "role": "user",
                "content": user_message,
                "time": timestamp_now
            })
            with st.chat_message("user"):
                st.markdown(user_message)

            last_messages = st.session_state.messages[-5:] if len(st.session_state.messages) >= 5 else st.session_state.messages
            with st.spinner("Searching documents..."):
                st.markdown("**While you wait, feel free to refer to original documents**")

                for file_key in st.session_state.selected_files:
                    preview_url = get_presigned_url(file_key)
                    st.markdown(f"[**{file_key}**]({preview_url})", unsafe_allow_html=True)

                payload = {
                    "selected_files": st.session_state.selected_files,
                    "selected_page_ranges": st.session_state.selected_page_ranges,
                    "prompt": user_message,
                    "top_k": top_k,
                    "last_messages": [json.dumps(last_messages)],
                    "web_search": web_search
                }

                try:
                    answer_json = post_json("/query_documents_with_page_range", payload)
                    answer = answer_json["answer"]
                except Exception as e:
                    answer = f"Backend error: {e}"

                st.session_state.sources.append({"answer": answer})

            assistant_answer = answer
            current_time = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_answer,
                "time": current_time
            })
            with st.chat_message("assistant"):
                st.markdown(assistant_answer)

            user = st.session_state.username
            if user not in st.session_state.chat_history:
                st.session_state.chat_history[user] = []
            # Save conversation logic ...
            # Then st.rerun()

    elif option == "Usage Monitoring":
        st.header("Usage Monitoring")
        # ... your usage analytics code ...
        pass

if __name__ == "__main__":
    main()
