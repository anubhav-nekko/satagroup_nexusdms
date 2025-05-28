"""
Unit tests for authentication endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_register_user():
    """Test user registration endpoint."""
    response = client.post(
        "/v1/auth/register",
        json={
            "fullName": "Test User",
            "email": "test@example.com",
            "password": "password123",
            "role": "user"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert "userId" in data
    assert data["fullName"] == "Test User"
    assert data["email"] == "test@example.com"
    assert data["role"] == "user"

def test_register_user_invalid_role():
    """Test user registration with invalid role."""
    response = client.post(
        "/v1/auth/register",
        json={
            "fullName": "Test User",
            "email": "test@example.com",
            "password": "password123",
            "role": "invalid_role"
        }
    )
    assert response.status_code == 400

def test_login_user():
    """Test user login endpoint."""
    # Register a user first
    client.post(
        "/v1/auth/register",
        json={
            "fullName": "Login Test",
            "email": "login@example.com",
            "password": "password123",
            "role": "user"
        }
    )
    
    # Login with the registered user
    response = client.post(
        "/v1/auth/login",
        json={
            "email": "login@example.com",
            "password": "password123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "accessToken" in data
    assert "refreshToken" in data
    assert "user" in data
    assert data["user"]["email"] == "login@example.com"

def test_login_user_invalid_credentials():
    """Test user login with invalid credentials."""
    response = client.post(
        "/v1/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        }
    )
    assert response.status_code == 401

def test_forgot_password():
    """Test forgot password endpoint."""
    response = client.post(
        "/v1/auth/forgot-password",
        json={
            "email": "test@example.com"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

def test_reset_password():
    """Test reset password endpoint."""
    response = client.post(
        "/v1/auth/reset-password",
        json={
            "token": "reset_token",
            "newPassword": "newpassword123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
