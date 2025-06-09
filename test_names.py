"""
Test candidate name extraction with new documents
"""

import requests
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

BASE_URL = "http://localhost:8000"

def create_test_pdf(content, filename):
    """Create a PDF file from text content"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    lines = content.split('\n')
    y_position = height - 50
    
    for line in lines:
        if y_position < 50:
            c.showPage()
            y_position = height - 50
        
        c.drawString(50, y_position, line[:80])
        y_position -= 15
    
    c.save()

def test_name_extraction():
    """Test candidate name extraction functionality"""
    print("🚀 TESTING CANDIDATE NAME EXTRACTION")
    print("=" * 60)
    
    # Login
    login_data = {"email": "admin@system.com", "password": "admin123"}
    response = requests.post(f"{BASE_URL}/v1/auth/login", json=login_data)
    token = response.json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authentication successful")
    
    # Create test documents with clear candidate names
    jd_content = """
    Senior Software Engineer Position
    
    We are seeking a Senior Software Engineer with:
    - 5+ years Python experience
    - Django/Flask frameworks
    - AWS cloud experience
    - Database management (PostgreSQL, MongoDB)
    - REST API development
    """
    
    cv_content = """
    SARAH JOHNSON
    Senior Software Developer
    
    Email: sarah.johnson@email.com
    Phone: (555) 123-4567
    Location: San Francisco, CA
    
    PROFESSIONAL EXPERIENCE:
    
    Senior Software Developer at TechCorp (2020-2023)
    - Developed Python applications using Django
    - Built REST APIs for microservices
    - Managed PostgreSQL databases
    - Deployed applications on AWS
    
    Software Developer at StartupXYZ (2018-2020)
    - Built web applications with Flask
    - Worked with MongoDB databases
    - Implemented CI/CD pipelines
    
    TECHNICAL SKILLS:
    - Programming: Python, JavaScript, SQL
    - Frameworks: Django, Flask, React
    - Databases: PostgreSQL, MongoDB, Redis
    - Cloud: AWS (EC2, S3, RDS)
    - Tools: Git, Docker, Jenkins
    
    EDUCATION:
    Bachelor of Science in Computer Science
    University of California, Berkeley (2014-2018)
    """
    
    # Create PDF files
    create_test_pdf(jd_content, "test_jd_names.pdf")
    create_test_pdf(cv_content, "test_cv_names.pdf")
    
    # Upload JD
    print("📄 Uploading Job Description...")
    with open("test_jd_names.pdf", "rb") as f:
        files = {"file": f}
        data = {"title": "Senior Software Engineer"}
        response = requests.post(f"{BASE_URL}/v1/jds", files=files, data=data, headers=headers, timeout=60)
    
    if response.status_code == 200:
        jd_result = response.json()
        jd_id = jd_result["jdId"]
        print(f"✅ JD uploaded: {jd_result['title']}")
    else:
        print(f"❌ JD upload failed: {response.text}")
        return
    
    # Upload CV
    print("📄 Uploading CV...")
    with open("test_cv_names.pdf", "rb") as f:
        files = {"file": f}
        data = {"level": "sr"}
        response = requests.post(f"{BASE_URL}/v1/cvs", files=files, data=data, headers=headers, timeout=60)
    
    if response.status_code == 200:
        cv_result = response.json()
        cv_id = cv_result["cvId"]
        print(f"✅ CV uploaded: {cv_result['fileName']}")
    else:
        print(f"❌ CV upload failed: {response.text}")
        return
    
    # Wait for processing
    print("⏳ Waiting for document processing...")
    import time
    time.sleep(5)
    
    # Perform skill analysis
    print("🧠 Performing skill analysis with name extraction...")
    analysis_data = {
        "jdId": jd_id,
        "cvIds": [cv_id],
        "options": {"includeScores": True, "language": "en"}
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/analysis",
        json=analysis_data,
        headers=headers,
        timeout=180
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SKILL ANALYSIS WITH NAMES SUCCESSFUL!")
        print(f"📊 Analysis ID: {result['analysisId']}")
        print("\n" + "=" * 60)
        print("📋 ANALYSIS RESULTS WITH CANDIDATE NAMES:")
        print("=" * 60)
        print(result['results'])
        print("=" * 60)
        
        # Check if candidate name is included
        if "Sarah Johnson" in result['results'] or "SARAH JOHNSON" in result['results']:
            print("✅ CANDIDATE NAME SUCCESSFULLY EXTRACTED!")
        else:
            print("⚠️  Candidate name may not have been extracted properly")
            
    else:
        print(f"❌ Analysis failed: {response.status_code} - {response.text}")
    
    # Cleanup
    try:
        os.remove("test_jd_names.pdf")
        os.remove("test_cv_names.pdf")
    except:
        pass
    
    print("\n🎉 NAME EXTRACTION TEST COMPLETED!")

if __name__ == "__main__":
    test_name_extraction()

