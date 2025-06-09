"""
Test the actual skill analysis results
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_analysis_results():
    """Test and display actual analysis results"""
    # Login
    login_data = {"email": "admin@system.com", "password": "admin123"}
    response = requests.post(f"{BASE_URL}/v1/auth/login", json=login_data)
    token = response.json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get the analysis ID from the previous test
    analysis_id = "52c632a2-30da-44bd-b5f4-0eae7f869a0a"
    
    response = requests.get(
        f"{BASE_URL}/v1/analysis/{analysis_id}",
        headers=headers
    )
    
    if response.status_code == 200:
        result = response.json()
        print("=== SKILL ANALYSIS RESULTS ===")
        print(f"Analysis ID: {result['analysisId']}")
        print(f"Timestamp: {result['timestamp']}")
        print("\n=== DETAILED ANALYSIS ===")
        print(result['results'])
    else:
        print(f"Failed to get analysis: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_analysis_results()

