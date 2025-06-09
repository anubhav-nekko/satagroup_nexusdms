"""
Working Final Test - Uses correct document IDs
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_with_existing_documents():
    """Test with documents already in the system"""
    print("🚀 TESTING WITH EXISTING DOCUMENTS")
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
        # Use the first available JD and CV
        jd_id = jds[0]["jdId"]
        cv_id = cvs[0]["cvId"]
        
        print(f"\n🧠 Testing skill analysis with:")
        print(f"   JD: {jds[0]['title']} (ID: {jd_id})")
        print(f"   CV: {cvs[0]['fileName']} (ID: {cv_id})")
        
        # Perform analysis
        analysis_data = {
            "jdId": jd_id,
            "cvIds": [cv_id],
            "options": {"includeScores": True, "language": "en"}
        }
        
        print("⏳ Running LLM analysis...")
        response = requests.post(
            f"{BASE_URL}/v1/analysis",
            json=analysis_data,
            headers=headers,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SKILL ANALYSIS SUCCESSFUL!")
            print(f"📊 Analysis ID: {result['analysisId']}")
            print("\n" + "=" * 60)
            print("📋 ANALYSIS RESULTS:")
            print("=" * 60)
            print(result['results'])
            print("=" * 60)
            
            # Test retrieval
            analysis_id = result['analysisId']
            response = requests.get(f"{BASE_URL}/v1/analysis/{analysis_id}", headers=headers)
            if response.status_code == 200:
                print("✅ Analysis retrieval successful")
            
        else:
            print(f"❌ Analysis failed: {response.status_code} - {response.text}")
    
    else:
        print("⚠️  No documents found for testing")
    
    print("\n🎉 TEST COMPLETED!")

if __name__ == "__main__":
    test_with_existing_documents()

