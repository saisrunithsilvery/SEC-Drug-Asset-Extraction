import os
import argparse
import logging
import boto3
import tempfile
import shutil
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import io
import json
from dotenv import load_dotenv
from botocore.exceptions import ClientError

from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sec_pipeline.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# S3 configuration
BUCKET_NAME = "k-cap-hfund"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name='us-east-1'
)

# Initialize NLP tools - using OpenAI embeddings instead of HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1000)

def list_s3_html_files(ticker: str) -> list:
    """List all HTML files in the S3 bucket for a given ticker."""
    try:
        prefix = f"filings/{ticker}/"
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.warning(f"No files found in s3://{BUCKET_NAME}/{prefix}")
            return []
            
        file_keys = [
            item['Key'] for item in response['Contents']
            if item['Key'].endswith('.html')
        ]
        
        logger.info(f"Found {len(file_keys)} HTML files in s3://{BUCKET_NAME}/{prefix}")
        return file_keys
    except ClientError as e:
        logger.error(f"Error listing S3 objects: {str(e)}")
        return []

def download_from_s3(key: str) -> str:
    """Download a file from S3 and return its contents."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return response['Body'].read().decode('utf-8')
    except ClientError as e:
        logger.error(f"Error downloading {key} from S3: {str(e)}")
        raise

def parse_filing_from_s3(file_key: str) -> list:
    """Parse HTML filing from S3 and extract text chunks."""
    try:
        html_content = download_from_s3(file_key)
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        chunks = text_splitter.split_text(text)
        logger.info(f"Parsed {file_key} into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error parsing {file_key}: {str(e)}")
        return []

def upload_folder_to_s3(local_path: str, s3_prefix: str):
    """Upload an entire folder to S3."""
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
    except Exception as e:
        logger.error(f"Error uploading folder to S3: {str(e)}")
        raise

def build_vector_store(ticker: str):
    """Build FAISS index from all filings for a ticker in S3 bucket."""
    try:
        file_keys = list_s3_html_files(ticker)
        if not file_keys:
            logger.warning(f"No HTML files found in s3://{BUCKET_NAME}/filings/{ticker}/")
            return None
            
        all_chunks = []
        metadata = []
        
        for file_key in file_keys:
            chunks = parse_filing_from_s3(file_key)
            all_chunks.extend(chunks)
            metadata.extend([{"file_path": file_key, "chunk_id": i} for i in range(len(chunks))])
            
        if not all_chunks:
            logger.warning(f"No chunks extracted from filings for {ticker}")
            return None
            
        # Create a temporary directory to save the FAISS index
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = FAISS.from_texts(all_chunks, embeddings, metadatas=metadata)
            
            # Save FAISS index to the temporary directory
            local_index_path = os.path.join(temp_dir, "faiss_index")
            vector_store.save_local(local_index_path)
            logger.info(f"Built FAISS index for {ticker} with {len(all_chunks)} chunks")
            
            # Upload the index to S3
            s3_index_prefix = f"filings/{ticker}/faiss_index"
            upload_folder_to_s3(local_index_path, s3_index_prefix)
            logger.info(f"Uploaded FAISS index to s3://{BUCKET_NAME}/{s3_index_prefix}")
            
            return vector_store
    except Exception as e:
        logger.error(f"Error building vector store for {ticker}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Build vector store from existing SEC filings for a ticker.")
    parser.add_argument("--ticker", default="WVE", help="Stock ticker (e.g., WVE)")
    args = parser.parse_args()

    ticker = args.ticker
    try:
        logger.info(f"Starting pipeline for ticker {ticker}")
        vector_store = build_vector_store(ticker)
        if vector_store:
            logger.info(f"Pipeline completed for {ticker} - vector store created and uploaded to S3")
        else:
            logger.warning(f"Pipeline completed for {ticker} but no vector store was created")
    except Exception as e:
        logger.error(f"Pipeline failed for {ticker}: {str(e)}")
        raise

if __name__ == "__main__":
    main()