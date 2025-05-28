"""
Unit tests for JD endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import io

from app.main import app
from app.core.security import HARDCODED_TOKEN

client = TestClient(app)

def test_list_jds():
    """Test listing job descriptions."""
    response = client.get(
        "/v1/jds",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_upload_jd():
    """Test uploading a job description."""
    # Create a mock PDF file
    file_content = b"%PDF-1.5\nMock PDF content for testing"
    files = {"file": ("test_jd.pdf", io.BytesIO(file_content), "application/pdf")}
    data = {"title": "Test Job Description"}
    
    response = client.post(
        "/v1/jds",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=files,
        data=data
    )
    assert response.status_code == 201
    data = response.json()
    assert "jdId" in data
    assert data["title"] == "Test Job Description"
    assert "uploadedAt" in data

def test_get_jd():
    """Test getting a job description."""
    # First upload a JD
    file_content = b"%PDF-1.5\nMock PDF content for testing"
    files = {"file": ("test_jd.pdf", io.BytesIO(file_content), "application/pdf")}
    data = {"title": "Test Job Description"}
    
    upload_response = client.post(
        "/v1/jds",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=files,
        data=data
    )
    jd_id = upload_response.json()["jdId"]
    
    # Now get the JD
    response = client.get(
        f"/v1/jds/{jd_id}",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jdId"] == jd_id
    assert data["title"] == "Test Job Description"
    assert "content" in data
    assert "uploadedAt" in data

def test_get_nonexistent_jd():
    """Test getting a non-existent job description."""
    response = client.get(
        "/v1/jds/nonexistent_id",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"}
    )
    assert response.status_code == 404
