import os
import io
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import uuid

from app.utils.sec_utils import (
    fetch_filings,
    download_filing_content,
    sanitize_filename
)
from app.utils.s3_utils import upload_to_s3

from app.config import BUCKET_NAME

# Configure logging
logger = logging.getLogger(__name__)


class FilingService:
    """Service for fetching and saving SEC filings"""
    
    @staticmethod
    def fetch_and_save_filings(ticker: str, form_types: Optional[List[str]] = None, max_filings: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Fetch filings for a given ticker and save them to S3.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'WVE')
            form_types: List of form types to fetch (e.g., ['10-K', '8-K'])
            max_filings: Maximum number of filings to fetch per form type
            
        Returns:
            Tuple containing job_id and list of filing metadata
        """
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        try:
            # Fetch filings
            filings = fetch_filings(ticker, form_types, max_filings or 100)
            if not filings:
                logger.warning(f"No filings found for {ticker}")
                return job_id, []
            
            # Define base path in S3 bucket
            base_path = f"filings/{ticker}"
            metadata = []
            
            for filing in filings:
                # Download filing content
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
                return job_id, []
            
            # Save metadata to S3
            metadata_df = pd.DataFrame(metadata)
            csv_buffer = io.StringIO()
            metadata_df.to_csv(csv_buffer, index=False)
            metadata_key = f"{base_path}/metadata.csv"
            
            upload_to_s3(csv_buffer.getvalue(), metadata_key)
            logger.info(f"Saved metadata for {len(metadata)} filings to s3://{BUCKET_NAME}/{metadata_key}")
            
            return job_id, metadata
        except Exception as e:
            logger.error(f"Error in fetch_and_save_filings for {ticker}: {str(e)}")
            raise