"""
Debug test script for authentication
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def debug_auth():
    """Debug authentication flow"""
    print("=== DEBUGGING AUTHENTICATION ===")
    
    # Test login with existing user (admin)
    login_data = {
        "email": "admin@system.com",
        "password": "admin123"
    }
    
    print(f"Attempting login with: {login_data}")
    
    try:
        response = requests.post(f"{BASE_URL}/v1/auth/login", json=login_data)
        print(f"Login response status: {response.status_code}")
        print(f"Login response: {response.json()}")
        
        if response.status_code == 200:
            token_data = response.json()
            token = token_data["accessToken"]
            print(f"Generated token: {token}")
            
            # Test using the token
            headers = {"Authorization": f"Bearer {token}"}
            print(f"Using headers: {headers}")
            
            # Test JDs endpoint
            jd_response = requests.get(f"{BASE_URL}/v1/jds", headers=headers)
            print(f"JDs endpoint status: {jd_response.status_code}")
            print(f"JDs response: {jd_response.text}")
            
            return token
        
    except Exception as e:
        print(f"Error: {e}")
    
    return None

if __name__ == "__main__":
    debug_auth()

