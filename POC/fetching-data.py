import os
from sec_api import QueryApi, RenderApi
import pandas as pd
import logging
import json
import requests
import re
import time

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
SEC_API_KEY = os.getenv("SEC_API_KEY", "")  # Replace with your key
USER_AGENT = "saisrunith saisrunith12@gmail.com"  # Required for SEC API compliance
query_api = QueryApi(api_key=SEC_API_KEY)
render_api = RenderApi(api_key=SEC_API_KEY)

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sec_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """Sanitize file names to remove or replace invalid characters."""
    # Replace invalid characters (e.g., /) with underscores
    return re.sub(r'[^\w\-\.]', '_', filename)    

def save_filings(ticker: str, filings: list):
    """Save filings metadata and content to disk with robust directory handling."""
    try:
        # Ensure the directory exists
        output_dir = f"filings/{ticker}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory {output_dir}")
        else:
            logger.info(f"Directory {output_dir} already exists")

        metadata = []
        for filing in filings:
            content_data = download_filing_content(filing)
            if content_data["content"]:
                # Sanitize form type for file naming (e.g., 10-K/A -> 10-K_A)
                form_type = sanitize_filename(filing["formType"])
                file_name = f"{filing['accessionNo']}_{form_type}.html"
                file_path = os.path.join(output_dir, file_name)
                
                # Write content to file
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content_data["content"])
                    logger.info(f"Saved filing content to {file_path}")
                except Exception as e:
                    logger.error(f"Failed to write {file_path}: {str(e)}")
                    continue
                
                # Store metadata
                metadata.append({
                    "ticker": ticker,
                    "formType": filing["formType"],
                    "filedAt": filing["filedAt"],
                    "accessionNo": filing["accessionNo"],
                    "url": filing["linkToFilingDetails"],
                    "filePath": file_path
                })
        
        if not metadata:
            logger.warning(f"No valid filings saved for {ticker}")
            return
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_path = os.path.join(output_dir, "metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata for {len(metadata)} filings to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving filings for {ticker}: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    ticker = "WVE"
    try:
        filings = fetch_filings(ticker)
        save_filings(ticker, filings)
    except Exception as e:
        logger.error(f"Pipeline failed for {ticker}: {str(e)}")