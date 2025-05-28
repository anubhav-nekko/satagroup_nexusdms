"""
README file for the FastAPI application.
"""
# Skill Analysis API

A FastAPI application for skill analysis between job descriptions and CVs.

## Features

- User authentication with JWT tokens
- Job description upload and management
- CV upload and management with experience levels
- Skill analysis between JDs and CVs
- AWS integration for file storage and processing
- Swagger UI documentation

## Project Structure

```
/app
├── main.py                 # FastAPI application entry point
├── config.py               # Configuration settings
├── auth/                   # Authentication endpoints and utilities
├── jds/                    # Job description endpoints
├── cvs/                    # CV endpoints
├── analysis/               # Analysis endpoints
├── core/                   # Core utilities (AWS, security, embedding)
└── utils/                  # Utility functions
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   uvicorn app.main:app --reload --host=0.0.0.0
   ```

## API Documentation

Once the application is running, you can access the Swagger UI documentation at:
```
http://localhost:8000/docs
```

## Authentication

The API uses JWT token authentication. For development, you can use the hardcoded token:
```
Authorization: Bearer development_token_for_testing_only
```

## Endpoints

### Authentication
- `POST /v1/auth/register` - Register a new user
- `POST /v1/auth/login` - Login and get tokens
- `POST /v1/auth/forgot-password` - Request password reset
- `POST /v1/auth/reset-password` - Reset password

### Job Descriptions
- `GET /v1/jds` - List all job descriptions
- `POST /v1/jds` - Upload a new job description
- `GET /v1/jds/{jdId}` - Get job description details

### CVs
- `GET /v1/cvs` - List all CVs
- `POST /v1/cvs` - Upload a new CV with experience level

### Analysis
- `POST /v1/analysis` - Perform skill analysis
- `GET /v1/analysis/{analysisId}` - Get analysis results

## Testing

Run the tests with pytest:
```
pytest
```

## AWS Integration

The application integrates with AWS services:
- S3 for file storage
- Textract for OCR and text extraction
- Bedrock for LLM-based analysis

## Configuration

Update the configuration in `app/config.py` to match your environment.
