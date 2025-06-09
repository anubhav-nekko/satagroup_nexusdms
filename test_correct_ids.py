"""
Test with correct document IDs to verify name extraction
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_with_correct_ids():
    """Test using correct document IDs"""
    print("🚀 TESTING NAME EXTRACTION WITH CORRECT IDS")
    print("=" * 60)
    
    # Login
    login_data = {"email": "admin@system.com", "password": "admin123"}
    response = requests.post(f"{BASE_URL}/v1/auth/login", json=login_data)
    token = response.json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authentication successful")
    
    # Get existing documents
    response = requests.get(f"{BASE_URL}/v1/jds", headers=headers)
    jds = response.json()
    print(f"✅ Found {len(jds)} job descriptions")
    
    response = requests.get(f"{BASE_URL}/v1/cvs", headers=headers)
    cvs = response.json()
    print(f"✅ Found {len(cvs)} CVs")
    
    if jds and cvs:
        # Use the most recent documents
        jd_id = jds[-1]["jdId"]  # Last uploaded JD
        cv_id = cvs[-1]["cvId"]  # Last uploaded CV
        
        print(f"\n🧠 Testing name extraction with:")
        print(f"   JD: {jds[-1]['title']} (ID: {jd_id})")
        print(f"   CV: {cvs[-1]['fileName']} (ID: {cv_id})")
        
        # Perform analysis
        analysis_data = {
            "jdId": jd_id,
            "cvIds": [cv_id],
            "options": {"includeScores": True, "language": "en"}
        }
        
        print("⏳ Running analysis with name extraction...")
        response = requests.post(
            f"{BASE_URL}/v1/analysis",
            json=analysis_data,
            headers=headers,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ANALYSIS WITH NAME EXTRACTION SUCCESSFUL!")
            print(f"📊 Analysis ID: {result['analysisId']}")
            print("\n" + "=" * 60)
            print("📋 ANALYSIS RESULTS:")
            print("=" * 60)
            analysis_text = result['results']
            print(analysis_text)
            print("=" * 60)
            
            # Check for candidate names in the results
            if any(name in analysis_text for name in ["Sarah Johnson", "SARAH JOHNSON", "Sarah", "Johnson"]):
                print("\n✅ SUCCESS: CANDIDATE NAME FOUND IN ANALYSIS!")
            else:
                print("\n⚠️  Candidate name not clearly visible in analysis")
                
            # Check for the new format with candidate names
            if "### CV 1:" in analysis_text and ("Sarah" in analysis_text or "Name not found" in analysis_text):
                print("✅ SUCCESS: Updated report format with candidate names working!")
            else:
                print("⚠️  Report format may need adjustment")
            
        else:
            print(f"❌ Analysis failed: {response.status_code} - {response.text}")
    
    else:
        print("⚠️  No documents found for testing")
    
    print("\n🎉 NAME EXTRACTION TEST COMPLETED!")

if __name__ == "__main__":
    test_with_correct_ids()

