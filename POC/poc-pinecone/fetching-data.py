import os
import boto3
from sec_api import QueryApi, RenderApi
import pandas as pd
import logging
import json
import requests
import re
import time
import io
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sec_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API keys and configuration
SEC_API_KEY = os.getenv("SEC_API_KEY")  # Will be loaded from .env file
USER_AGENT = "saisrunith saisrunith54@gmail.com"  # Required for SEC API compliance
query_api = QueryApi(api_key=SEC_API_KEY)
render_api = RenderApi(api_key=SEC_API_KEY)

# S3 configuration
BUCKET_NAME = "k-cap-hfund"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")  # Get from environment variable
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")  # Get from environment variable

# Initialize S3 client with credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name='us-east-1'  # Specify the region to match your bucket
)

def get_cik_from_ticker(ticker: str) -> str:
    """Resolve ticker to CIK using SEC's company_tickers.json."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tickers = response.json()
        for entry in tickers.values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)  # Pad with leading zeros
                logger.info(f"Resolved ticker {ticker} to CIK {cik}")
                return cik
        logger.error(f"No CIK found for ticker {ticker}")
        raise ValueError(f"No CIK found for ticker {ticker}")
    except Exception as e:
        logger.error(f"Error resolving CIK for {ticker}: {str(e)}")
        raise

def fetch_filings(ticker: str, form_types: list = ["10-K", "8-K"]) -> list:
    """Fetch metadata for 10-K and 8-K filings for a given ticker."""
    try:
        cik = get_cik_from_ticker(ticker)
        filings = []
        for form_type in form_types:
            query = {
                "query": f'ticker:{ticker} AND formType:"{form_type}"',
                "from": "0",
                "size": "100",  # Max 100 per request
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            logger.info(f"Querying {form_type} filings for {ticker} (CIK: {cik})")
            while True:
                response = query_api.get_filings(query)
                filings.extend(response["filings"])
                total = response["total"]["value"]
                logger.info(f"Fetched {len(response['filings'])} of {total} {form_type} filings")
                if len(filings) >= total:
                    break
                query["from"] = str(len(filings))  # Paginate
        logger.info(f"Total filings retrieved for {ticker}: {len(filings)}")
        return filings
    except Exception as e:
        logger.error(f"Error fetching filings for {ticker}: {str(e)}")
        raise

def download_filing_content(filing: dict) -> dict:
    """Download the content of a single filing."""
    try:
        url = filing["linkToFilingDetails"]
        content = render_api.get_filing(url)
        logger.info(f"Downloaded content for filing {filing['accessionNo']}")
        return {"filing": filing, "content": content}
    except Exception as e:
        logger.error(f"Error downloading filing {filing['accessionNo']}: {str(e)}")
        return {"filing": filing, "content": None}

def sanitize_filename(filename: str) -> str:
    """Sanitize file names to remove or replace invalid characters."""
    # Replace invalid characters (e.g., /) with underscores
    return re.sub(r'[^\w\-\.]', '_', filename)    

def upload_to_s3(content, key):
    """Upload content to S3 bucket with specified key."""
    try:
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

def save_filings_to_s3(ticker: str, filings: list):
    """Save filings metadata and content to S3 bucket with the same folder structure."""
    try:
        # Define base path in S3 bucket
        base_path = f"filings/{ticker}"
        metadata = []
        
        for filing in filings:
            content_data = download_filing_content(filing)
            if content_data["content"]:
                # Sanitize form type for file naming (e.g., 10-K/A -> 10-K_A)
                form_type = sanitize_filename(filing["formType"])
                file_name = f"{filing['accessionNo']}_{form_type}.html"
                s3_key = f"{base_path}/{file_name}"
                
                # Upload content to S3
                try:
                    s3_path = upload_to_s3(content_data["content"], s3_key)
                    logger.info(f"Saved filing content to {s3_path}")
                except Exception as e:
                    logger.error(f"Failed to upload {s3_key}: {str(e)}")
                    continue
                
                # Store metadata
                metadata.append({
                    "ticker": ticker,
                    "formType": filing["formType"],
                    "filedAt": filing["filedAt"],
                    "accessionNo": filing["accessionNo"],
                    "url": filing["linkToFilingDetails"],
                    "s3Path": f"s3://{BUCKET_NAME}/{s3_key}"
                })
        
        if not metadata:
            logger.warning(f"No valid filings saved for {ticker}")
            return
        
        # Save metadata to S3
        metadata_df = pd.DataFrame(metadata)
        csv_buffer = io.StringIO()
        metadata_df.to_csv(csv_buffer, index=False)
        metadata_key = f"{base_path}/metadata.csv"
        
        upload_to_s3(csv_buffer.getvalue(), metadata_key)
        logger.info(f"Saved metadata for {len(metadata)} filings to s3://{BUCKET_NAME}/{metadata_key}")
    except Exception as e:
        logger.error(f"Error saving filings for {ticker} to S3: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    ticker = "WVE"
    try:
        filings = fetch_filings(ticker)
        save_filings_to_s3(ticker, filings)
    except Exception as e:
        logger.error(f"Pipeline failed for {ticker}: {str(e)}")