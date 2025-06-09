"""
Test the new JSON output format for skill analysis
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_json_output():
    """Test the new JSON-formatted analysis output"""
    print("🚀 TESTING NEW JSON OUTPUT FORMAT")
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
        jd_id = jds[-1]["jdId"]
        cv_id = cvs[-1]["cvId"]
        
        print(f"\n🧠 Testing JSON output with:")
        print(f"   JD: {jds[-1]['title']} (ID: {jd_id})")
        print(f"   CV: {cvs[-1]['fileName']} (ID: {cv_id})")
        
        # Perform analysis
        analysis_data = {
            "jdId": jd_id,
            "cvIds": [cv_id],
            "options": {"includeScores": True, "language": "en"}
        }
        
        print("⏳ Running analysis with new JSON output...")
        response = requests.post(
            f"{BASE_URL}/v1/analysis",
            json=analysis_data,
            headers=headers,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ANALYSIS WITH JSON OUTPUT SUCCESSFUL!")
            print(f"📊 Analysis ID: {result['analysisId']}")
            
            # Pretty print the JSON results
            print("\n" + "=" * 60)
            print("📋 NEW JSON FORMAT RESULTS:")
            print("=" * 60)
            
            analysis_results = result['results']
            print(json.dumps(analysis_results, indent=2))
            
            print("=" * 60)
            
            # Validate the JSON structure
            print("\n🔍 VALIDATING JSON STRUCTURE:")
            
            required_fields = ["analysis_id", "total_candidates", "jd_name", "jd_id", "overall_analysis_notes", "candidates"]
            for field in required_fields:
                if field in analysis_results:
                    print(f"✅ {field}: {type(analysis_results[field])}")
                else:
                    print(f"❌ Missing field: {field}")
            
            # Check candidate structure
            if "candidates" in analysis_results and analysis_results["candidates"]:
                candidate = analysis_results["candidates"][0]
                candidate_fields = ["candidate_name", "cv_id", "match_score", "skills_found", "missing_skills", "additional_skills", "experience_match", "detailed_reasoning"]
                
                print("\n🔍 VALIDATING CANDIDATE STRUCTURE:")
                for field in candidate_fields:
                    if field in candidate:
                        print(f"✅ {field}: {type(candidate[field])}")
                    else:
                        print(f"❌ Missing candidate field: {field}")
                
                print(f"\n📊 CANDIDATE SUMMARY:")
                print(f"   Name: {candidate.get('candidate_name', 'N/A')}")
                print(f"   Match Score: {candidate.get('match_score', 'N/A')}%")
                print(f"   CV ID: {candidate.get('cv_id', 'N/A')}")
            
            print("\n✅ JSON FORMAT VALIDATION COMPLETE!")
            
        else:
            print(f"❌ Analysis failed: {response.status_code} - {response.text}")
    
    else:
        print("⚠️  No documents found for testing")
    
    print("\n🎉 JSON OUTPUT TEST COMPLETED!")

if __name__ == "__main__":
    test_json_output()

