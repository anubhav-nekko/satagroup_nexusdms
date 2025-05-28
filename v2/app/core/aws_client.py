"""
Core AWS client integration for S3, Textract, and Bedrock services.
"""
import json
import boto3
from typing import Dict, Any, List

from app.config import (
    AWS_REGION, 
    S3_BUCKET,
    BEDROCK_MODEL_ID
)

class AWSClient:
    """AWS client for S3, Textract, and Bedrock services."""
    
    def __init__(self):
        """Initialize AWS clients."""
        self.s3 = boto3.client(
            "s3", 
            region_name=AWS_REGION
        )
        
        self.textract = boto3.client(
            "textract", 
            region_name=AWS_REGION
        )
        
        self.bedrock = boto3.client(
            "bedrock-runtime", 
            region_name=AWS_REGION
        )
    
    def upload_file(self, file_path: str, object_name: str = None) -> bool:
        """
        Upload a file to an S3 bucket.
        
        Args:
            file_path: Path to the file to upload
            object_name: S3 object name (if None, file_path is used)
            
        Returns:
            True if file was uploaded, else False
        """
        if object_name is None:
            object_name = file_path
            
        try:
            self.s3.upload_file(file_path, S3_BUCKET, object_name)
            return True
        except Exception as e:
            print(f"Error uploading file to S3: {e}")
            return False
    
    def upload_fileobj(self, file_obj, object_name: str) -> bool:
        """
        Upload a file-like object to an S3 bucket.
        
        Args:
            file_obj: File-like object to upload
            object_name: S3 object name
            
        Returns:
            True if file was uploaded, else False
        """
        try:
            self.s3.upload_fileobj(file_obj, S3_BUCKET, object_name)
            return True
        except Exception as e:
            print(f"Error uploading file object to S3: {e}")
            return False
    
    def download_file(self, object_name: str, file_path: str) -> bool:
        """
        Download a file from an S3 bucket.
        
        Args:
            object_name: S3 object name
            file_path: Path to save the downloaded file
            
        Returns:
            True if file was downloaded, else False
        """
        try:
            self.s3.download_file(S3_BUCKET, object_name, file_path)
            return True
        except Exception as e:
            print(f"Error downloading file from S3: {e}")
            return False
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """
        Extract text from an image using Amazon Textract.
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            Extracted text
        """
        try:
            response = self.textract.detect_document_text(
                Document={"Bytes": image_bytes}
            )
            return "\n".join(
                block["Text"] 
                for block in response.get("Blocks", [])
                if block["BlockType"] == "LINE"
            )
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    
    def invoke_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """
        Invoke Bedrock LLM for text generation.
        
        Args:
            prompt: Input prompt for the LLM
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            }
            
            response = self.bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]
        except Exception as e:
            print(f"Error invoking Bedrock LLM: {e}")
            return ""
    
    def analyze_skills(self, jd_text: str, cv_texts: List[str], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze skills match between JD and CVs.
        
        Args:
            jd_text: Job description text
            cv_texts: List of CV texts
            options: Analysis options
            
        Returns:
            Analysis results
        """
        if options is None:
            options = {}
            
        include_scores = options.get("includeScores", True)
        language = options.get("language", "en")
        
        results = []
        
        for i, cv_text in enumerate(cv_texts):
            prompt = f"""
            Analyze the following job description and CV to identify:
            1. Skills found in the CV that match the job description
            2. Skills required in the job description but missing from the CV
            3. Overall match score (0-100)
            
            Job Description:
            {jd_text}
            
            CV:
            {cv_text}
            
            Language: {language}
            
            Provide a structured JSON response with the following format:
            {{
                "matchScore": <score>,
                "skillsFound": ["skill1", "skill2", ...],
                "missingSkills": ["skill1", "skill2", ...]
            }}
            """
            
            try:
                response = self.invoke_llm(prompt)
                # Extract JSON from response
                import re
                json_match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                    if not include_scores and "matchScore" in analysis:
                        del analysis["matchScore"]
                    analysis["cvId"] = f"cv_{i}"  # Placeholder for actual CV ID
                    results.append(analysis)
                else:
                    results.append({
                        "cvId": f"cv_{i}",
                        "error": "Failed to parse analysis results"
                    })
            except Exception as e:
                results.append({
                    "cvId": f"cv_{i}",
                    "error": f"Analysis failed: {str(e)}"
                })
        
        return {"results": results}

# Create a singleton instance
aws_client = AWSClient()
