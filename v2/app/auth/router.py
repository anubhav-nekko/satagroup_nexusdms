"""
Authentication router for the API.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import timedelta
from typing import Dict, Any

from app.auth.models import (
    UserRegister,
    UserLogin,
    UserResponse,
    TokenResponse,
    ForgotPassword,
    ResetPassword,
    MessageResponse
)
from app.core.security import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_password_hash,
    MOCK_USERS,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.config import JWT_SECRET_KEY

router = APIRouter(prefix="/v1/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister) -> Dict[str, Any]:
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        
    Returns:
        Registered user data
    """
    # Check if email already exists
    if user_data.email in MOCK_USERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate role
    if user_data.role not in ["user", "manager", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be one of: user, manager, admin"
        )
    
    # Create new user
    user_id = f"user_{len(MOCK_USERS) + 1}"
    hashed_password = get_password_hash(user_data.password)
    
    # Store user in mock database
    MOCK_USERS[user_data.email] = {
        "userId": user_id,
        "fullName": user_data.fullName,
        "email": user_data.email,
        "password": hashed_password,
        "role": user_data.role
    }
    
    # Return user data
    return {
        "userId": user_id,
        "fullName": user_data.fullName,
        "email": user_data.email,
        "role": user_data.role
    }

@router.post("/login", response_model=TokenResponse)
async def login_user(user_data: UserLogin) -> Dict[str, Any]:
    """
    Authenticate user and return tokens.
    
    Args:
        user_data: User login data
        
    Returns:
        Access and refresh tokens with user data
    """
    # Authenticate user
    user = authenticate_user(user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(
        data={"sub": user["email"], "role": user["role"]}
    )
    
    # Return tokens and user data
    return {
        "accessToken": access_token,
        "expiresIn": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "refreshToken": refresh_token,
        "user": {
            "userId": user["userId"],
            "fullName": user["fullName"],
            "email": user["email"],
            "role": user["role"]
        }
    }

@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(data: ForgotPassword) -> Dict[str, str]:
    """
    Handle forgot password request.
    
    Args:
        data: Forgot password data
        
    Returns:
        Success message
    """
    # Check if email exists
    if data.email in MOCK_USERS:
        # In a real implementation, send password reset email
        return {"message": "If an account with that email exists, a password reset link has been sent."}
    
    # Return same message even if email doesn't exist for security
    return {"message": "If an account with that email exists, a password reset link has been sent."}

@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(data: ResetPassword) -> Dict[str, str]:
    """
    Reset user password.
    
    Args:
        data: Reset password data
        
    Returns:
        Success message
    """
    # In a real implementation, validate token and update password
    # For development, just return success
    return {"message": "Password has been reset successfully."}
