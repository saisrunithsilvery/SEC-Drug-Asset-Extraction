import os
import logging
import uuid
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.s3_utils import (
    list_s3_objects,
    download_from_s3,
    upload_folder_to_s3
)
from app.config import EMBEDDINGS_MODEL

# Configure logging
logger = logging.getLogger(__name__)

# Initialize NLP tools
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)


class VectorStoreService:
    """Service for building and managing vector stores for SEC filings"""
    
    @staticmethod
    def build_vector_store(ticker: str) -> Tuple[str, bool]:
        """
        Build FAISS index from all filings for a ticker in S3 bucket.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'WVE')
            
        Returns:
            Tuple containing job_id and success status
        """
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        try:
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
            
            # Get list of HTML files for the ticker
            file_keys = VectorStoreService._list_html_files(ticker)
            if not file_keys:
                logger.warning(f"No HTML files found for {ticker}")
                return job_id, False
            
            # Process files and extract chunks
            all_chunks = []
            metadata = []
            
            for file_key in file_keys:
                chunks = VectorStoreService._parse_filing_from_s3(file_key)
                all_chunks.extend(chunks)
                metadata.extend([{"file_path": file_key, "chunk_id": i} for i in range(len(chunks))])
            
            if not all_chunks:
                logger.warning(f"No chunks extracted from filings for {ticker}")
                return job_id, False
            
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
                logger.info(f"Uploaded FAISS index to s3://{s3_index_prefix}")
                
                return job_id, True
        except Exception as e:
            logger.error(f"Error building vector store for {ticker}: {str(e)}")
            raise
    
    @staticmethod
    def _list_html_files(ticker: str) -> List[str]:
        """
        List all HTML files in the S3 bucket for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'WVE')
            
        Returns:
            List of S3 keys
        """
        prefix = f"filings/{ticker}/"
        all_files = list_s3_objects(prefix)
        html_files = [key for key in all_files if key.endswith('.html')]
        logger.info(f"Found {len(html_files)} HTML files for {ticker}")
        return html_files
    
    @staticmethod
    def _parse_filing_from_s3(file_key: str) -> List[str]:
        """
        Parse HTML filing from S3 and extract text chunks.
        
        Args:
            file_key: S3 key of the filing HTML file
            
        Returns:
            List of text chunks
        """
        try:
            html_content = download_from_s3(file_key).decode('utf-8')
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            chunks = text_splitter.split_text(text)
            logger.info(f"Parsed {file_key} into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error parsing {file_key}: {str(e)}")
            return []