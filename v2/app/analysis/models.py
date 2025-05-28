"""
Analysis models for the API.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

class AnalysisOptions(BaseModel):
    """Analysis options model."""
    includeScores: Optional[bool] = True
    language: Optional[str] = "en"

class AnalysisRequest(BaseModel):
    """Analysis request model."""
    jdId: str
    cvIds: List[str]
    options: Optional[AnalysisOptions] = None

class AnalysisResult(BaseModel):
    """Analysis result model for a single CV."""
    cvId: str
    matchScore: Optional[float] = None
    skillsFound: List[str]
    missingSkills: List[str]
    error: Optional[str] = None

class AnalysisResponse(BaseModel):
    """Analysis response model."""
    analysisId: str
    timestamp: str
    results: List[AnalysisResult]
    status: Optional[str] = "completed"
