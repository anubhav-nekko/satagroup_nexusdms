"""
Analysis router for the API.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
from datetime import datetime

from app.analysis.models import AnalysisRequest, AnalysisResponse, AnalysisResult
from app.core.security import get_current_active_user
from app.core.aws_client import aws_client
from app.jds.router import MOCK_JDS
from app.cvs.router import MOCK_CVS

router = APIRouter(prefix="/v1/analysis", tags=["Analysis"])

# Mock analysis database for development
MOCK_ANALYSES = {}

@router.post("", response_model=AnalysisResponse)
async def create_analysis(
    analysis_request: AnalysisRequest,
    current_user: Dict = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Perform skill analysis between JD and CVs.
    
    Args:
        analysis_request: Analysis request data
        current_user: Current authenticated user
        
    Returns:
        Analysis results
    """
    # Validate JD exists
    if analysis_request.jdId not in MOCK_JDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    # Validate CVs exist
    for cv_id in analysis_request.cvIds:
        if cv_id not in MOCK_CVS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CV not found: {cv_id}"
            )
    
    # Generate analysis ID
    analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().isoformat()  # Using datetime.now() instead of utcnow()
    
    # In a real implementation, fetch JD and CV content from S3 or database
    # For development, use mock content
    jd_content = "This is a mock job description content for development purposes."
    cv_contents = {cv_id: f"This is mock CV content for {cv_id}" for cv_id in analysis_request.cvIds}
    
    # Perform analysis
    options = analysis_request.options.model_dump() if analysis_request.options else {}  # Using model_dump() instead of dict()
    include_scores = options.get("includeScores", True)
    
    # Mock analysis results
    results = []
    for cv_id in analysis_request.cvIds:
        # In a real implementation, use AWS client to analyze skills
        # For development, use mock results
        if include_scores:
            # Include matchScore only if includeScores is True
            result = {
                "cvId": cv_id,
                "skillsFound": ["Python", "FastAPI", "AWS"],
                "missingSkills": ["Docker", "Kubernetes"],
                "matchScore": 75.5,
                "error": None
            }
        else:
            # Completely omit matchScore if includeScores is False
            result = {
                "cvId": cv_id,
                "skillsFound": ["Python", "FastAPI", "AWS"],
                "missingSkills": ["Docker", "Kubernetes"],
                "error": None
            }
        
        results.append(result)
    
    # Store analysis results
    MOCK_ANALYSES[analysis_id] = {
        "timestamp": timestamp,
        "results": results,
        "jdId": analysis_request.jdId,
        "cvIds": analysis_request.cvIds,
        "options": options,
        "requestedBy": current_user["userId"]
    }
    
    return {
        "analysisId": analysis_id,
        "timestamp": timestamp,
        "results": results
    }

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: Dict = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Retrieve analysis results by ID.
    
    Args:
        analysis_id: Analysis ID
        current_user: Current authenticated user
        
    Returns:
        Analysis results
    """
    # Check if analysis exists
    if analysis_id not in MOCK_ANALYSES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    analysis_data = MOCK_ANALYSES[analysis_id]
    
    return {
        "analysisId": analysis_id,
        "timestamp": analysis_data["timestamp"],
        "results": analysis_data["results"]
    }
