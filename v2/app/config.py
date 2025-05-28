"""
Configuration settings for the FastAPI application.
"""
import os
from typing import Dict, Any

# AWS Configuration
AWS_REGION = "us-east-1"
S3_BUCKET = "satagroup-test"

# Bedrock Configuration
BEDROCK_MODEL_ID = "arn:aws:bedrock:us-east-1:343218220592:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# File Storage Configuration
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata_store.pkl"
EMBEDDING_DIMENSION = 768

# Authentication Configuration
JWT_SECRET_KEY = "hardcoded_secret_key_for_development_only"  # For development only
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# API Configuration
API_V1_PREFIX = "/v1"
PROJECT_NAME = "Skill Analysis API"
API_VERSION = "1.0.0"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# File Upload Configuration
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Analysis Configuration
DEFAULT_TOP_K = 20
