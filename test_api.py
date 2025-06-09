"""
Test script for the Skill Analysis FastAPI application
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_auth_endpoints():
    """Test authentication endpoints"""
    print("Testing authentication endpoints...")
    
    # Test user registration
    register_data = {
        "fullName": "Test User",
        "email": "test@example.com",
        "password": "testpass123",
        "role": "user"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/auth/register", json=register_data)
        print(f"Register: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Register error: {e}")
    
    # Test user login
    login_data = {
        "email": "test@example.com",
        "password": "testpass123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/auth/login", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            print(f"Login: {response.status_code} - Success")
            return token_data["accessToken"]
        else:
            print(f"Login: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Login error: {e}")
    
    return None

def test_document_endpoints(token):
    """Test document management endpoints"""
    if not token:
        print("No token available for document tests")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    print("\nTesting document endpoints...")
    
    # Test listing JDs
    try:
        response = requests.get(f"{BASE_URL}/v1/jds", headers=headers)
        print(f"List JDs: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"List JDs error: {e}")
    
    # Test listing CVs
    try:
        response = requests.get(f"{BASE_URL}/v1/cvs", headers=headers)
        print(f"List CVs: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"List CVs error: {e}")

def test_health_check():
    """Test if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        return response.status_code == 200
    except:
        return False

def main():
    print("Starting FastAPI tests...")
    
    # Wait for server to start
    print("Waiting for server to start...")
    for i in range(10):
        if test_health_check():
            print("Server is running!")
            break
        time.sleep(2)
    else:
        print("Server not responding. Please start the server first.")
        return
    
    # Run tests
    token = test_auth_endpoints()
    test_document_endpoints(token)
    
    print("\nTests completed!")

if __name__ == "__main__":
    main()

