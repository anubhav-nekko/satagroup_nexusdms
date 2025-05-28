"""
Unit tests for CV endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import io

from app.main import app
from app.core.security import HARDCODED_TOKEN

client = TestClient(app)

def test_list_cvs():
    """Test listing CVs."""
    response = client.get(
        "/v1/cvs",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_upload_cv():
    """Test uploading a CV."""
    # Create a mock PDF file
    file_content = b"%PDF-1.5\nMock CV content for testing"
    files = {"file": ("test_cv.pdf", io.BytesIO(file_content), "application/pdf")}
    data = {"level": "mid"}
    
    response = client.post(
        "/v1/cvs",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=files,
        data=data
    )
    assert response.status_code == 201
    data = response.json()
    assert "cvId" in data
    assert data["fileName"] == "test_cv.pdf"
    assert data["level"] == "mid"
    assert "uploadedAt" in data

def test_upload_cv_invalid_level():
    """Test uploading a CV with invalid level."""
    # Create a mock PDF file
    file_content = b"%PDF-1.5\nMock CV content for testing"
    files = {"file": ("test_cv.pdf", io.BytesIO(file_content), "application/pdf")}
    data = {"level": "invalid_level"}
    
    response = client.post(
        "/v1/cvs",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=files,
        data=data
    )
    assert response.status_code == 400
