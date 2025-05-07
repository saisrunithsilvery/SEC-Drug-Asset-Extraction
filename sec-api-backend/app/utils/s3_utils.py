import os
import boto3
import io
import logging
from botocore.exceptions import ClientError
from typing import List, Optional, Dict, Any, BinaryIO, Union
import tempfile
import shutil

from app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BUCKET_NAME,
    AWS_REGION
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)


def upload_to_s3(content: Union[str, bytes, BinaryIO], key: str) -> str:
    """
    Upload content to S3 bucket with specified key.
    
    Args:
        content: Content to upload (string, bytes, or file-like object)
        key: S3 key for the uploaded file
        
    Returns:
        S3 URI for the uploaded file
    """
    try:
        # Convert string to bytes if needed
        if isinstance(content, str):
            content = content.encode('utf-8')
            
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=content
        )
        logger.info(f"Successfully uploaded to s3://{BUCKET_NAME}/{key}")
        return f"s3://{BUCKET_NAME}/{key}"
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        raise


def download_from_s3(key: str) -> bytes:
    """
    Download a file from S3 and return its contents.
    
    Args:
        key: S3 key of the file to download
        
    Returns:
        File contents as bytes
    """
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return response['Body'].read()
    except ClientError as e:
        logger.error(f"Error downloading {key} from S3: {str(e)}")
        raise


def list_s3_objects(prefix: str) -> List[str]:
    """
    List objects in S3 bucket with given prefix.
    
    Args:
        prefix: S3 prefix to list objects from
        
    Returns:
        List of S3 keys
    """
    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.warning(f"No files found in s3://{BUCKET_NAME}/{prefix}")
            return []
            
        keys = [item['Key'] for item in response['Contents']]
        logger.info(f"Found {len(keys)} objects in s3://{BUCKET_NAME}/{prefix}")
        return keys
    except ClientError as e:
        logger.error(f"Error listing S3 objects: {str(e)}")
        return []


def download_s3_folder(s3_prefix: str, local_path: str) -> bool:
    """
    Download an entire folder from S3 to local path.
    
    Args:
        s3_prefix: S3 prefix for the folder to download
        local_path: Local path to download to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the local directory exists
        os.makedirs(local_path, exist_ok=True)
        
        # List objects with the given prefix
        keys = list_s3_objects(s3_prefix)
        if not keys:
            return False
            
        # Download each object
        for s3_key in keys:
            # Create the relative path for the local file
            if not s3_key.startswith(s3_prefix):
                continue
                
            relative_path = s3_key[len(s3_prefix):].lstrip('/')
            if not relative_path:  # Skip the directory itself
                continue
                
            local_file_path = os.path.join(local_path, relative_path)
            
            # Create local directory structure if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download the file
            s3_client.download_file(BUCKET_NAME, s3_key, local_file_path)
            logger.info(f"Downloaded s3://{BUCKET_NAME}/{s3_key} to {local_file_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading folder from S3: {str(e)}")
        return False


def upload_folder_to_s3(local_path: str, s3_prefix: str) -> bool:
    """
    Upload an entire folder to S3.
    
    Args:
        local_path: Local path of the folder to upload
        s3_prefix: S3 prefix to upload to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                # Create S3 key by replacing the local path with S3 prefix
                relative_path = os.path.relpath(local_file_path, local_path)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
                
                # Upload file
                with open(local_file_path, 'rb') as f:
                    s3_client.upload_fileobj(f, BUCKET_NAME, s3_key)
                
                logger.info(f"Uploaded {local_file_path} to s3://{BUCKET_NAME}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading folder to S3: {str(e)}")
        return False


def delete_s3_object(key: str) -> bool:
    """
    Delete an object from S3.
    
    Args:
        key: S3 key of the object to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
        logger.info(f"Deleted s3://{BUCKET_NAME}/{key}")
        return True
    except ClientError as e:
        logger.error(f"Error deleting S3 object: {str(e)}")
        return False


def delete_s3_prefix(prefix: str) -> bool:
    """
    Delete all objects with a given prefix from S3.
    
    Args:
        prefix: S3 prefix to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        keys = list_s3_objects(prefix)
        if not keys:
            return True
            
        # Delete objects
        for key in keys:
            delete_s3_object(key)
            
        logger.info(f"Deleted {len(keys)} objects with prefix s3://{BUCKET_NAME}/{prefix}")
        return True
    except Exception as e:
        logger.error(f"Error deleting S3 prefix: {str(e)}")
        return False