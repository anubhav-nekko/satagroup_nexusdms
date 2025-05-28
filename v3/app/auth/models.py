"""
Authentication models for the API.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any

class UserRegister(BaseModel):
    """User registration request model."""
    fullName: str
    email: EmailStr
    password: str
    role: str = Field(..., description="User role (user|manager|admin)")

class UserLogin(BaseModel):
    """User login request model."""
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    """User response model."""
    userId: str
    fullName: str
    email: EmailStr
    role: str

class TokenResponse(BaseModel):
    """Token response model."""
    accessToken: str
    expiresIn: int
    refreshToken: str
    user: UserResponse

class ForgotPassword(BaseModel):
    """Forgot password request model."""
    email: EmailStr

class ResetPassword(BaseModel):
    """Reset password request model."""
    token: str
    newPassword: str

class MessageResponse(BaseModel):
    """Generic message response model."""
    message: str
