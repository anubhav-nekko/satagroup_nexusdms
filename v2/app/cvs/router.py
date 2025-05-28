"""
CV (Curriculum Vitae) router for the API.
"""
import os
import tempfile
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Dict, Any, List
from datetime import datetime

from app.cvs.models import CVResponse
from app.core.security import get_current_active_user
from app.core.aws_client import aws_client
from app.utils.file_processing import process_file
from app.config import ALLOWED_EXTENSIONS, S3_BUCKET

router = APIRouter(prefix="/v1/cvs", tags=["CVs"])

# Mock CV database for development
MOCK_CVS = {}

@router.get("", response_model=List[CVResponse])
async def list_cvs(current_user: Dict = Depends(get_current_active_user)) -> List[Dict[str, Any]]:
    """
    List all uploaded CVs.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of CVs
    """
    return [
        {
            "cvId": cv_id,
            "fileName": cv_data["fileName"],
            "level": cv_data["level"],
            "uploadedAt": cv_data["uploadedAt"]
        }
        for cv_id, cv_data in MOCK_CVS.items()
    ]

@router.post("", response_model=CVResponse, status_code=status.HTTP_201_CREATED)
async def upload_cv(
    file: UploadFile = File(...),
    level: str = Form(...),
    current_user: Dict = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Upload a new CV with experience level.
    
    Args:
        file: CV file (PDF/DOCX)
        level: Experience level (jr|mid|sr)
        current_user: Current authenticated user
        
    Returns:
        Uploaded CV data
    """
    # Validate file extension
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Validate level
    if level not in ["jr", "mid", "sr"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid level. Must be one of: jr, mid, sr"
        )
    
    # Generate CV ID
    cv_id = f"cv_{uuid.uuid4().hex[:8]}"
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        # Upload to S3
        s3_key = f"cvs/{cv_id}{file_ext}"
        aws_client.upload_file(temp_file_path, s3_key)
        
        # Process file for text extraction and indexing
        pages_processed = process_file(temp_file_path, current_user["userId"], filename)
        
        # Store CV metadata
        upload_time = datetime.utcnow().isoformat()
        MOCK_CVS[cv_id] = {
            "fileName": filename,
            "level": level,
            "s3Key": s3_key,
            "uploadedAt": upload_time,
            "uploadedBy": current_user["userId"],
            "pagesProcessed": pages_processed
        }
        
        return {
            "cvId": cv_id,
            "fileName": filename,
            "level": level,
            "uploadedAt": upload_time
        }
    finally:
        # Clean up temporary file
        os.remove(temp_file_path)
