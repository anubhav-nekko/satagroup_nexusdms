# Skill Analysis FastAPI System

## Overview

The Skill Analysis FastAPI System is a comprehensive HR-tech solution designed to automate the process of skill extraction, gap analysis, and candidate matching between job descriptions (JDs) and resumes/CVs. Built upon the existing Streamlit architecture, this system leverages advanced machine learning techniques including FAISS vector indexing, AWS Textract for OCR, and AWS Bedrock's Claude LLM for intelligent skill analysis.

This system represents a significant advancement from prototype to production-ready API, maintaining compatibility with existing data structures while providing scalable, enterprise-grade endpoints for skill analysis workflows. The architecture preserves the proven document processing pipeline while introducing robust authentication, comprehensive error handling, and detailed analytics capabilities.

## Architecture Overview

The system is built on a modern FastAPI framework that integrates seamlessly with AWS services and maintains compatibility with the existing FAISS-based document indexing system. The architecture follows a microservices approach with clear separation of concerns between authentication, document management, and skill analysis functionalities.

### Core Components

**Document Processing Pipeline**: The system maintains the proven document processing approach from the original Streamlit implementation, utilizing PyMuPDF for PDF processing and AWS Textract for optical character recognition. Documents are processed page-wise, with each page converted to high-resolution images (300 DPI) before OCR processing to ensure maximum text extraction accuracy.

**FAISS Vector Indexing**: The system employs Facebook AI Similarity Search (FAISS) for efficient similarity search and clustering of document embeddings. Using the sentence-transformers/all-mpnet-base-v2 model, text chunks are converted to 768-dimensional embeddings that enable semantic search and similarity matching between job requirements and candidate skills.

**AWS Integration**: The system leverages multiple AWS services including S3 for document storage, Textract for OCR processing, and Bedrock for LLM-powered skill analysis. This integration ensures scalability, reliability, and access to state-of-the-art AI capabilities.

**Authentication System**: A simplified yet secure authentication system using SHA-256 token generation provides access control while maintaining compatibility with the existing users.json structure. The system supports role-based access control with user, manager, and admin roles.

## Installation and Setup

### Prerequisites

Before installing the Skill Analysis FastAPI System, ensure your environment meets the following requirements:

- Python 3.11 or higher
- AWS account with appropriate permissions for S3, Textract, and Bedrock services
- Sufficient disk space for model downloads (approximately 2GB for sentence transformers)
- Network connectivity for downloading pre-trained models and AWS service access

### Environment Setup

Create a dedicated virtual environment for the application to ensure dependency isolation:

```bash
python3.11 -m venv skill_analysis_env
source skill_analysis_env/bin/activate  # On Windows: skill_analysis_env\\Scripts\\activate
```

### Dependency Installation

Install all required dependencies using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

The system requires several key dependencies including FastAPI for the web framework, sentence-transformers for embedding generation, FAISS for vector search, PyMuPDF for PDF processing, and various AWS SDK components for cloud integration.

### AWS Configuration

Configure your AWS credentials either through environment variables or AWS credentials file. The system requires access to the following AWS services:

- **S3**: For document storage and metadata persistence
- **Textract**: For optical character recognition
- **Bedrock**: For LLM-powered skill analysis

Ensure your AWS account has the necessary permissions for these services and that the specified S3 bucket exists and is accessible.

### Initial Configuration

The system automatically initializes required files on first startup, including:

- `users.json`: User authentication database with default admin account
- `analysis_results.json`: Storage for skill analysis results
- FAISS index files: Vector search index and metadata storage

## API Documentation

### Authentication Endpoints

The authentication system provides secure access control with token-based authentication. All protected endpoints require a valid Bearer token in the Authorization header.

#### POST /v1/auth/register

Registers a new user in the system with the specified role and credentials.

**Request Body:**
```json
{
  "fullName": "string",
  "email": "user@example.com",
  "password": "string",
  "role": "user"
}
```

**Response:**
```json
{
  "userId": "uuid",
  "fullName": "string",
  "email": "user@example.com",
  "role": "user"
}
```

The registration endpoint validates email uniqueness and creates a new user entry in the users.json file. User roles include "user", "manager", and "admin", with different access levels for various system functionalities.

#### POST /v1/auth/login

Authenticates a user and returns access tokens for subsequent API calls.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "string"
}
```

**Response:**
```json
{
  "accessToken": "string",
  "expiresIn": 86400,
  "refreshToken": "string",
  "user": {
    "userId": "uuid",
    "fullName": "string",
    "email": "user@example.com",
    "role": "user"
  }
}
```

The login endpoint generates a SHA-256 hash token based on user email and ID, providing a simple yet effective authentication mechanism suitable for the current system requirements.

### Document Management Endpoints

The document management system handles job descriptions and CVs with comprehensive metadata tracking and access control.

#### GET /v1/jds

Retrieves a list of all job descriptions accessible to the authenticated user.

**Headers:**
```
Authorization: Bearer <accessToken>
```

**Response:**
```json
[
  {
    "jdId": "string",
    "title": "string",
    "uploadedAt": "2023-01-01T00:00:00Z"
  }
]
```

#### POST /v1/jds

Uploads a new job description document for processing and indexing.

**Headers:**
```
Authorization: Bearer <accessToken>
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: PDF or DOCX file
- `title` (optional): Custom title for the job description

**Response:**
```json
{
  "jdId": "string",
  "title": "string",
  "fileName": "string",
  "uploadedAt": "2023-01-01T00:00:00Z"
}
```

The upload process includes document validation, OCR processing, text extraction, embedding generation, and FAISS index updates. Each page of the document is processed individually to maintain granular search capabilities.

#### GET /v1/jds/{jdId}

Retrieves detailed content of a specific job description.

**Headers:**
```
Authorization: Bearer <accessToken>
```

**Response:**
```json
{
  "jdId": "string",
  "title": "string",
  "content": "string",
  "uploadedAt": "2023-01-01T00:00:00Z"
}
```

### CV Management Endpoints

The CV management system provides similar functionality to job descriptions with additional experience level tracking.

#### GET /v1/cvs

Lists all CVs accessible to the authenticated user with experience level information.

#### POST /v1/cvs

Uploads a CV with specified experience level (jr, mid, sr) for skill analysis.

**Form Data:**
- `file`: PDF or DOCX file
- `level`: Experience level (jr|mid|sr)

### Skill Analysis Endpoints

The core functionality of the system, providing AI-powered skill extraction and gap analysis.

#### POST /v1/analysis

Performs comprehensive skill analysis between a job description and multiple CVs.

**Request Body:**
```json
{
  "jdId": "string",
  "cvIds": ["string"],
  "options": {
    "includeScores": true,
    "language": "en"
  }
}
```

**Response:**
```json
{
  "analysisId": "string",
  "timestamp": "2023-01-01T00:00:00Z",
  "results": "string"
}
```

The analysis process involves multiple stages: skill extraction from both JD and CVs using LLM analysis, semantic matching between required and available skills, gap analysis identifying missing competencies, scoring based on skill overlap and experience alignment, and comprehensive report generation with recommendations.

#### GET /v1/analysis/{analysisId}

Retrieves previously generated analysis results for review and sharing.

## Skill Analysis Algorithm

The skill analysis system employs a sophisticated multi-stage approach to evaluate candidate-job fit through advanced natural language processing and machine learning techniques.

### Skill Extraction Process

The skill extraction process begins with comprehensive text analysis using AWS Bedrock's Claude LLM. The system processes both job descriptions and CVs to identify technical skills, soft skills, tools, technologies, frameworks, and implied competencies. The LLM is prompted to be comprehensive yet precise, avoiding duplicates while capturing both explicit mentions and contextual implications.

For job descriptions, the system focuses on identifying required skills, preferred qualifications, experience levels, and domain-specific competencies. The extraction process considers various formats and styles of job postings, from formal corporate descriptions to startup-style casual listings.

CV analysis involves identifying demonstrated skills, project experience, educational background, certifications, and career progression indicators. The system recognizes skills mentioned in different contexts, from technical project descriptions to achievement summaries.

### Matching and Scoring Algorithm

The matching algorithm employs a multi-dimensional approach to evaluate candidate-job fit:

**Skill Overlap Analysis**: Direct matching between required skills and candidate skills, with semantic similarity consideration for related technologies and competencies.

**Experience Level Assessment**: Evaluation of candidate experience against job requirements, considering both explicit years of experience and project complexity indicators.

**Gap Analysis**: Identification of critical missing skills and assessment of their importance to the role, with categorization of gaps as critical, important, or nice-to-have.

**Scoring Methodology**: A weighted scoring system that considers skill coverage (40%), experience alignment (30%), additional valuable skills (20%), and overall profile fit (10%).

### Report Generation

The system generates comprehensive reports that include executive summaries with key metrics, individual candidate analyses with detailed breakdowns, skill gap identification with recommendations, and comparative rankings with justifications.

Reports are structured to provide actionable insights for hiring managers, including specific recommendations for candidate interviews, areas for skill development, and potential role modifications based on available talent.

## Deployment Guide

### Local Development Deployment

For development and testing purposes, the system can be deployed locally with minimal configuration:

```bash
cd skill_analysis_api
python3.11 main.py
```

The application will start on `http://localhost:8000` with automatic API documentation available at `/docs`.

### Production Deployment Considerations

Production deployment requires several additional considerations for security, scalability, and reliability:

**Security Enhancements**: Implementation of proper JWT tokens with expiration, HTTPS enforcement, rate limiting, and input validation strengthening.

**Scalability Improvements**: Database migration from JSON files to proper database systems, horizontal scaling with load balancers, and caching implementation for frequently accessed data.

**Monitoring and Logging**: Comprehensive logging for audit trails, performance monitoring, and error tracking with alerting systems.

**Backup and Recovery**: Automated backup systems for user data and analysis results, disaster recovery procedures, and data retention policies.

### Cloud Deployment Options

The system is designed for cloud deployment with several recommended approaches:

**AWS ECS/Fargate**: Containerized deployment with automatic scaling and load balancing capabilities.

**AWS Lambda**: Serverless deployment for cost-effective scaling, though with considerations for cold start times and execution limits.

**Traditional EC2**: Full control deployment with custom configuration and optimization opportunities.

## Usage Examples

### Basic Authentication Flow

```python
import requests

# Register a new user
register_data = {
    "fullName": "John Doe",
    "email": "john@company.com",
    "password": "secure_password",
    "role": "user"
}
response = requests.post("http://localhost:8000/v1/auth/register", json=register_data)
print(response.json())

# Login to get access token
login_data = {
    "email": "john@company.com",
    "password": "secure_password"
}
response = requests.post("http://localhost:8000/v1/auth/login", json=login_data)
token = response.json()["accessToken"]

# Use token for authenticated requests
headers = {"Authorization": f"Bearer {token}"}
```

### Document Upload and Analysis

```python
# Upload a job description
with open("job_description.pdf", "rb") as f:
    files = {"file": f}
    data = {"title": "Senior Python Developer"}
    response = requests.post(
        "http://localhost:8000/v1/jds",
        files=files,
        data=data,
        headers=headers
    )
jd_id = response.json()["jdId"]

# Upload CVs
cv_ids = []
for cv_file in ["cv1.pdf", "cv2.pdf", "cv3.pdf"]:
    with open(cv_file, "rb") as f:
        files = {"file": f}
        data = {"level": "mid"}
        response = requests.post(
            "http://localhost:8000/v1/cvs",
            files=files,
            data=data,
            headers=headers
        )
    cv_ids.append(response.json()["cvId"])

# Perform skill analysis
analysis_data = {
    "jdId": jd_id,
    "cvIds": cv_ids,
    "options": {"includeScores": True, "language": "en"}
}
response = requests.post(
    "http://localhost:8000/v1/analysis",
    json=analysis_data,
    headers=headers
)
analysis_result = response.json()
print(analysis_result["results"])
```

## Troubleshooting

### Common Issues and Solutions

**Authentication Failures**: Verify that tokens are properly included in request headers and that user credentials are correct. Check that the users.json file is properly formatted and accessible.

**Document Upload Errors**: Ensure uploaded files are in supported formats (PDF, DOCX) and within size limits. Verify AWS credentials and S3 bucket permissions.

**Skill Analysis Timeouts**: Large documents or multiple CV analysis may require extended timeout settings. Consider processing smaller batches or implementing asynchronous processing.

**Model Download Issues**: Initial startup requires downloading sentence transformer models. Ensure stable internet connectivity and sufficient disk space.

### Performance Optimization

**Memory Management**: Monitor memory usage during large document processing and implement batch processing for multiple documents.

**FAISS Index Optimization**: Regular index rebuilding and optimization can improve search performance as the document corpus grows.

**Caching Strategies**: Implement caching for frequently accessed documents and analysis results to reduce processing overhead.

## Security Considerations

The current implementation provides a foundation for secure operations while maintaining simplicity for development and testing environments. Production deployments should implement additional security measures including proper JWT token management with expiration and refresh capabilities, input validation and sanitization for all endpoints, rate limiting to prevent abuse, HTTPS enforcement for all communications, and comprehensive audit logging for security monitoring.

Access control is implemented through role-based permissions, with users having access only to their own documents and analyses, managers having broader access within their organization, and administrators having system-wide access for management and monitoring purposes.

## Future Enhancements

The system architecture supports several potential enhancements for expanded functionality and improved performance:

**Advanced Skill Taxonomies**: Implementation of industry-specific skill frameworks and standardized competency models for more precise matching.

**Machine Learning Improvements**: Integration of custom-trained models for domain-specific skill extraction and improved matching algorithms.

**Real-time Processing**: Implementation of streaming processing for large-scale document analysis and real-time skill gap monitoring.

**Integration Capabilities**: API integrations with popular HR systems, applicant tracking systems, and learning management platforms.

**Analytics Dashboard**: Comprehensive analytics and reporting capabilities for HR teams to track hiring trends and skill gap patterns.

This comprehensive FastAPI system provides a robust foundation for enterprise-scale skill analysis while maintaining the proven effectiveness of the original Streamlit prototype. The modular architecture ensures scalability and extensibility for future enhancements while delivering immediate value through automated skill analysis and candidate matching capabilities.

