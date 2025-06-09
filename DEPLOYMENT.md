# Deployment Guide for Skill Analysis FastAPI System

## Quick Start Deployment

### Local Development Setup

1. **Clone or download the project files**
   ```bash
   # Ensure you have all project files in a directory
   cd skill_analysis_api
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   - The system uses hardcoded AWS credentials for simplicity
   - For production, use environment variables or AWS IAM roles
   - Ensure your AWS account has access to S3, Textract, and Bedrock services

4. **Start the application**
   ```bash
   python3.11 main.py
   ```

5. **Access the API**
   - API Server: http://localhost:8000
   - Interactive Documentation: http://localhost:8000/docs
   - OpenAPI Schema: http://localhost:8000/openapi.json

### Default Credentials

The system creates a default admin user on first startup:
- Email: admin@system.com
- Password: admin123
- Role: admin

## Production Deployment

### Environment Variables (Recommended)

For production deployment, replace hardcoded credentials with environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET_NAME="your-bucket-name"
```

### Docker Deployment

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t skill-analysis-api .
docker run -p 8000:8000 -e AWS_ACCESS_KEY_ID=your-key skill-analysis-api
```

### AWS ECS Deployment

1. **Create ECR repository**
   ```bash
   aws ecr create-repository --repository-name skill-analysis-api
   ```

2. **Build and push Docker image**
   ```bash
   docker build -t skill-analysis-api .
   docker tag skill-analysis-api:latest your-account.dkr.ecr.region.amazonaws.com/skill-analysis-api:latest
   docker push your-account.dkr.ecr.region.amazonaws.com/skill-analysis-api:latest
   ```

3. **Create ECS task definition and service**

### Load Balancer Configuration

For production traffic, configure an Application Load Balancer:
- Health check path: `/docs`
- Target port: 8000
- SSL termination recommended

## Security Considerations

### Production Security Checklist

- [ ] Replace hardcoded AWS credentials with IAM roles
- [ ] Implement proper JWT tokens with expiration
- [ ] Add rate limiting to prevent abuse
- [ ] Enable HTTPS/TLS encryption
- [ ] Implement input validation and sanitization
- [ ] Add comprehensive logging and monitoring
- [ ] Set up backup procedures for user data
- [ ] Configure firewall rules and security groups

### Authentication Improvements

For production use, consider implementing:
- JWT tokens with expiration and refresh
- OAuth2 integration
- Multi-factor authentication
- Password complexity requirements
- Account lockout policies

## Monitoring and Maintenance

### Health Checks

The system provides several endpoints for monitoring:
- `/docs` - API documentation (indicates server health)
- Authentication endpoints for user system health
- Document listing endpoints for database connectivity

### Log Monitoring

Monitor application logs for:
- Authentication failures
- Document processing errors
- AWS service connectivity issues
- Performance bottlenecks

### Performance Optimization

- Monitor memory usage during document processing
- Implement caching for frequently accessed data
- Consider database migration from JSON files for large user bases
- Optimize FAISS index for better search performance

## Backup and Recovery

### Data Backup Strategy

1. **User Data**: Regular backup of users.json to S3
2. **Analysis Results**: Backup of analysis_results.json
3. **FAISS Index**: Backup of index files and metadata
4. **Document Storage**: S3 provides built-in redundancy

### Recovery Procedures

1. **System Recovery**: Restore from backed-up configuration files
2. **Data Recovery**: Restore user data and analysis results from S3
3. **Index Rebuilding**: Rebuild FAISS index from stored documents if needed

## Scaling Considerations

### Horizontal Scaling

- Deploy multiple instances behind a load balancer
- Use shared storage for user data and analysis results
- Consider database migration for better concurrent access

### Vertical Scaling

- Increase memory for larger document processing
- Add CPU cores for faster embedding generation
- Optimize storage for FAISS index performance

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check Python version (3.11+ required)
   - Verify all dependencies installed
   - Check port 8000 availability

2. **AWS connectivity issues**
   - Verify AWS credentials
   - Check S3 bucket permissions
   - Ensure Textract and Bedrock service access

3. **Document upload failures**
   - Check file format (PDF/DOCX supported)
   - Verify file size limits
   - Check AWS Textract quotas

4. **Authentication problems**
   - Verify users.json file format
   - Check token generation and verification
   - Ensure proper header format

### Debug Mode

Enable debug logging by modifying the main.py file:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support and Maintenance

### Regular Maintenance Tasks

- Monitor disk space for model downloads and document storage
- Review and rotate AWS credentials periodically
- Update dependencies for security patches
- Monitor API usage and performance metrics

### Version Updates

When updating the system:
1. Backup current data and configuration
2. Test updates in staging environment
3. Plan for potential downtime during updates
4. Verify all functionality after deployment

This deployment guide provides a comprehensive approach to deploying the Skill Analysis FastAPI System in various environments, from local development to production cloud deployment.

