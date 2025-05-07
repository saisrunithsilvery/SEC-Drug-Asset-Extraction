import re
import requests
import logging
import json
import time
from typing import List, Dict, Any, Optional
from sec_api import QueryApi, RenderApi

from app.config import (
    SEC_API_KEY, 
    USER_AGENT,
    DEFAULT_FILING_TYPES,
    DEFAULT_FILING_COUNT
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize SEC API clients
query_api = QueryApi(api_key=SEC_API_KEY)
render_api = RenderApi(api_key=SEC_API_KEY)


def get_cik_from_ticker(ticker: str) -> str:
    """
    Resolve ticker to CIK using SEC's company_tickers.json.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'WVE')
        
    Returns:
        CIK number (padded with leading zeros)
    """
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


def fetch_filings(ticker: str, form_types: Optional[List[str]] = None, max_filings: int = DEFAULT_FILING_COUNT) -> List[Dict[str, Any]]:
    """
    Fetch metadata for SEC filings for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'WVE')
        form_types: List of form types to fetch (e.g., ['10-K', '8-K'])
        max_filings: Maximum number of filings to fetch per form type
        
    Returns:
        List of filing metadata dictionaries
    """
    try:
        if form_types is None:
            form_types = DEFAULT_FILING_TYPES
            
        cik = get_cik_from_ticker(ticker)
        all_filings = []
        
        for form_type in form_types:
            query = {
                "query": f'ticker:{ticker} AND formType:"{form_type}"',
                "from": "0",
                "size": str(max_filings),
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            
            logger.info(f"Querying {form_type} filings for {ticker} (CIK: {cik})")
            response = query_api.get_filings(query)
            
            if 'filings' in response:
                filings = response["filings"]
                all_filings.extend(filings)
                total = response["total"]["value"]
                logger.info(f"Fetched {len(filings)} of {total} {form_type} filings")
            else:
                logger.warning(f"No {form_type} filings found for {ticker}")
                
        logger.info(f"Total filings retrieved for {ticker}: {len(all_filings)}")
        return all_filings
    except Exception as e:
        logger.error(f"Error fetching filings for {ticker}: {str(e)}")
        raise


def download_filing_content(filing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download the content of a single filing.
    
    Args:
        filing: Filing metadata dictionary
        
    Returns:
        Dictionary with filing metadata and content
    """
    try:
        url = filing["linkToFilingDetails"]
        content = render_api.get_filing(url)
        logger.info(f"Downloaded content for filing {filing['accessionNo']}")
        return {"filing": filing, "content": content}
    except Exception as e:
        logger.error(f"Error downloading filing {filing['accessionNo']}: {str(e)}")
        return {"filing": filing, "content": None}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize file names to remove or replace invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters (e.g., /) with underscores
    return re.sub(r'[^\w\-\.]', '_', filename)