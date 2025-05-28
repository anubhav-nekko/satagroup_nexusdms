"""
JD (Job Description) models for the API.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class JDResponse(BaseModel):
    """Job description response model."""
    jdId: str
    title: str
    uploadedAt: str

class JDDetailResponse(BaseModel):
    """Job description detail response model."""
    jdId: str
    title: str
    content: str
    uploadedAt: str
