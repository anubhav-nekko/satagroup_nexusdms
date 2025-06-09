"""
Debug script to check metadata and document IDs
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def debug_metadata():
    """Debug what's in the metadata"""
    # Login
    login_data = {"email": "admin@system.com", "password": "admin123"}
    response = requests.post(f"{BASE_URL}/v1/auth/login", json=login_data)
    token = response.json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get JDs
    response = requests.get(f"{BASE_URL}/v1/jds", headers=headers)
    jds = response.json()
    print("JDs in system:")
    for jd in jds:
        print(f"  - JD ID: {jd.get('jdId')}")
        print(f"    Title: {jd.get('title')}")
        print(f"    Uploaded: {jd.get('uploadedAt')}")
    
    # Get CVs
    response = requests.get(f"{BASE_URL}/v1/cvs", headers=headers)
    cvs = response.json()
    print("\nCVs in system:")
    for cv in cvs:
        print(f"  - CV ID: {cv.get('cvId')}")
        print(f"    Filename: {cv.get('fileName')}")
        print(f"    Level: {cv.get('level')}")
        print(f"    Uploaded: {cv.get('uploadedAt')}")
    
    if jds and cvs:
        # Try analysis with actual IDs
        jd_id = jds[0]['jdId']
        cv_id = cvs[0]['cvId']
        
        print(f"\nTrying analysis with:")
        print(f"  JD ID: {jd_id}")
        print(f"  CV ID: {cv_id}")
        
        analysis_data = {
            "jdId": jd_id,
            "cvIds": [cv_id],
            "options": {"includeScores": True, "language": "en"}
        }
        
        response = requests.post(
            f"{BASE_URL}/v1/analysis",
            json=analysis_data,
            headers=headers,
            timeout=120
        )
        
        print(f"\nAnalysis result: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Analysis ID: {result['analysisId']}")
            print("Analysis successful!")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    debug_metadata()

