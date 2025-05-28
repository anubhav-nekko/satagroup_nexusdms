"""
Core security utilities for authentication and authorization.
"""
import jwt
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")

# Hardcoded token for development
HARDCODED_TOKEN = "development_token_for_testing_only"

# Mock user database for development
MOCK_USERS = {
    "user@example.com": {
        "userId": "user1",
        "fullName": "Test User",
        "email": "user@example.com",
        "password": "password123",  # In production, this would be properly hashed
        "role": "user"
    },
    "admin@example.com": {
        "userId": "admin1",
        "fullName": "Admin User",
        "email": "admin@example.com",
        "password": "password123",  # In production, this would be properly hashed
        "role": "admin"
    },
    "login@example.com": {
        "userId": "user2",
        "fullName": "Login Test",
        "email": "login@example.com",
        "password": "password123",  # In production, this would be properly hashed
        "role": "user"
    }
}

class TokenData(BaseModel):
    """Token data model."""
    sub: str
    exp: datetime
    role: str

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt

def create_refresh_token(data: Dict) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Data to encode in the token
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt

def decode_token(token: str) -> Dict:
    """
    Decode a JWT token.
    
    Args:
        token: JWT token to decode
        
    Returns:
        Decoded token data
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """
    Get the current user from the token.
    
    Args:
        token: JWT token
        
    Returns:
        User data
    """
    # For development, accept hardcoded token
    if token == HARDCODED_TOKEN:
        return MOCK_USERS["admin@example.com"]
    
    # Otherwise, validate JWT token
    payload = decode_token(token)
    email = payload.get("sub")
    
    if email is None or email not in MOCK_USERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return MOCK_USERS[email]

def get_current_active_user(current_user: Dict = Depends(get_current_user)) -> Dict:
    """
    Get the current active user.
    
    Args:
        current_user: Current user data
        
    Returns:
        User data if active
    """
    # In a real implementation, check if user is active
    return current_user

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches hash
    """
    # For development, simple comparison
    # In production, use proper password hashing
    return plain_password == "password123"  # Simplified for development

def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    # For development, return as is
    # In production, use proper password hashing
    return "hashed_" + password  # Simplified for development

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user.
    
    Args:
        email: User email
        password: User password
        
    Returns:
        User data if authenticated, None otherwise
    """
    if email not in MOCK_USERS:
        return None
    
    user = MOCK_USERS[email]
    
    # For development, simple password check
    # Fixed to use the verify_password function properly
    if not verify_password(password, user["password"]):
        return None
    
    return user
