"""
Unit tests for analysis endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import io

from app.main import app
from app.core.security import HARDCODED_TOKEN

client = TestClient(app)

def test_create_analysis():
    """Test creating an analysis."""
    # First upload a JD
    jd_file_content = b"%PDF-1.5\nMock JD content for testing"
    jd_files = {"file": ("test_jd.pdf", io.BytesIO(jd_file_content), "application/pdf")}
    jd_data = {"title": "Test Job Description"}
    
    jd_response = client.post(
        "/v1/jds",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=jd_files,
        data=jd_data
    )
    jd_id = jd_response.json()["jdId"]
    
    # Then upload a CV
    cv_file_content = b"%PDF-1.5\nMock CV content for testing"
    cv_files = {"file": ("test_cv.pdf", io.BytesIO(cv_file_content), "application/pdf")}
    cv_data = {"level": "mid"}
    
    cv_response = client.post(
        "/v1/cvs",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=cv_files,
        data=cv_data
    )
    cv_id = cv_response.json()["cvId"]
    
    # Now create an analysis
    response = client.post(
        "/v1/analysis",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        json={
            "jdId": jd_id,
            "cvIds": [cv_id],
            "options": {
                "includeScores": True,
                "language": "en"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "analysisId" in data
    assert "timestamp" in data
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["cvId"] == cv_id
    assert "skillsFound" in data["results"][0]
    assert "missingSkills" in data["results"][0]
    assert "matchScore" in data["results"][0]

def test_create_analysis_without_scores():
    """Test creating an analysis without scores."""
    # First upload a JD
    jd_file_content = b"%PDF-1.5\nMock JD content for testing"
    jd_files = {"file": ("test_jd.pdf", io.BytesIO(jd_file_content), "application/pdf")}
    jd_data = {"title": "Test Job Description"}
    
    jd_response = client.post(
        "/v1/jds",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=jd_files,
        data=jd_data
    )
    jd_id = jd_response.json()["jdId"]
    
    # Then upload a CV
    cv_file_content = b"%PDF-1.5\nMock CV content for testing"
    cv_files = {"file": ("test_cv.pdf", io.BytesIO(cv_file_content), "application/pdf")}
    cv_data = {"level": "mid"}
    
    cv_response = client.post(
        "/v1/cvs",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=cv_files,
        data=cv_data
    )
    cv_id = cv_response.json()["cvId"]
    
    # Now create an analysis without scores
    response = client.post(
        "/v1/analysis",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        json={
            "jdId": jd_id,
            "cvIds": [cv_id],
            "options": {
                "includeScores": False,
                "language": "en"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "analysisId" in data
    assert "timestamp" in data
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["cvId"] == cv_id
    assert "skillsFound" in data["results"][0]
    assert "missingSkills" in data["results"][0]
    assert "matchScore" not in data["results"][0]

def test_get_analysis():
    """Test getting an analysis."""
    # First create an analysis
    # Upload a JD
    jd_file_content = b"%PDF-1.5\nMock JD content for testing"
    jd_files = {"file": ("test_jd.pdf", io.BytesIO(jd_file_content), "application/pdf")}
    jd_data = {"title": "Test Job Description"}
    
    jd_response = client.post(
        "/v1/jds",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=jd_files,
        data=jd_data
    )
    jd_id = jd_response.json()["jdId"]
    
    # Upload a CV
    cv_file_content = b"%PDF-1.5\nMock CV content for testing"
    cv_files = {"file": ("test_cv.pdf", io.BytesIO(cv_file_content), "application/pdf")}
    cv_data = {"level": "mid"}
    
    cv_response = client.post(
        "/v1/cvs",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        files=cv_files,
        data=cv_data
    )
    cv_id = cv_response.json()["cvId"]
    
    # Create analysis
    analysis_response = client.post(
        "/v1/analysis",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"},
        json={
            "jdId": jd_id,
            "cvIds": [cv_id],
            "options": {
                "includeScores": True,
                "language": "en"
            }
        }
    )
    analysis_id = analysis_response.json()["analysisId"]
    
    # Now get the analysis
    response = client.get(
        f"/v1/analysis/{analysis_id}",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["analysisId"] == analysis_id
    assert "timestamp" in data
    assert "results" in data

def test_get_nonexistent_analysis():
    """Test getting a non-existent analysis."""
    response = client.get(
        "/v1/analysis/nonexistent_id",
        headers={"Authorization": f"Bearer {HARDCODED_TOKEN}"}
    )
    assert response.status_code == 404
