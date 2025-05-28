"""
CV (Curriculum Vitae) models for the API.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class CVResponse(BaseModel):
    """CV response model."""
    cvId: str
    fileName: str
    level: str
    uploadedAt: str
