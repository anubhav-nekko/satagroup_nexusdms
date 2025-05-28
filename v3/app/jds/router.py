"""
JD (Job Description) router for the API.
"""
import os
import tempfile
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.jds.models import JDResponse, JDDetailResponse
from app.core.security import get_current_active_user
from app.core.aws_client import aws_client
from app.utils.file_processing import process_file
from app.config import ALLOWED_EXTENSIONS, S3_BUCKET

router = APIRouter(prefix="/v1/jds", tags=["Job Descriptions"])

# Mock JD database for development
MOCK_JDS = {}

@router.get("", response_model=List[JDResponse])
async def list_jds(current_user: Dict = Depends(get_current_active_user)) -> List[Dict[str, Any]]:
    """
    List all job descriptions.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of job descriptions
    """
    return [
        {
            "jdId": jd_id,
            "title": jd_data["title"],
            "uploadedAt": jd_data["uploadedAt"]
        }
        for jd_id, jd_data in MOCK_JDS.items()
    ]

@router.post("", response_model=JDResponse, status_code=status.HTTP_201_CREATED)
async def upload_jd(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: Dict = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Upload a new job description.
    
    Args:
        file: Job description file (PDF/DOCX)
        title: Optional title for the job description
        current_user: Current authenticated user
        
    Returns:
        Uploaded job description data
    """
    # Validate file extension
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate JD ID
    jd_id = f"jd_{uuid.uuid4().hex[:8]}"
    
    # Use filename as title if not provided
    if not title:
        title = os.path.splitext(filename)[0]
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        # Upload to S3
        s3_key = f"jds/{jd_id}{file_ext}"
        aws_client.upload_file(temp_file_path, s3_key)
        
        # Process file for text extraction and indexing
        pages_processed = process_file(temp_file_path, current_user["userId"], filename)
        
        # Store JD metadata
        upload_time = datetime.utcnow().isoformat()
        MOCK_JDS[jd_id] = {
            "title": title,
            "fileName": filename,
            "s3Key": s3_key,
            "uploadedAt": upload_time,
            "uploadedBy": current_user["userId"],
            "pagesProcessed": pages_processed
        }
        
        return {
            "jdId": jd_id,
            "title": title,
            "fileName": filename,
            "uploadedAt": upload_time
        }
    finally:
        # Clean up temporary file
        os.remove(temp_file_path)

@router.get("/{jd_id}", response_model=JDDetailResponse)
async def get_jd(
    jd_id: str,
    current_user: Dict = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get job description details.
    
    Args:
        jd_id: Job description ID
        current_user: Current authenticated user
        
    Returns:
        Job description details
    """
    # Check if JD exists
    if jd_id not in MOCK_JDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    jd_data = MOCK_JDS[jd_id]
    
    # In a real implementation, fetch content from S3 or database
    # For development, use mock content
    content = "This is a mock job description content for development purposes."
    
    return {
        "jdId": jd_id,
        "title": jd_data["title"],
        "content": content,
        "uploadedAt": jd_data["uploadedAt"]
    }
