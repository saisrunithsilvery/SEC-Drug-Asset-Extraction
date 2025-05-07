import os
import argparse
import logging
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import glob
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sec_pipeline.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize NLP tools
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1000)

def parse_filing(file_path: str) -> list:
    """Parse HTML filing and extract text chunks."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        chunks = text_splitter.split_text(text)
        logger.info(f"Parsed {file_path} into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {str(e)}")
        return []

def build_vector_store(ticker: str):
    """Build FAISS index from all filings for a ticker."""
    try:
        file_paths = glob.glob(f"filings/{ticker}/*.html")
        if not file_paths:
            logger.warning(f"No HTML files found in filings/{ticker}/")
            return None
        logger.info(f"Found {len(file_paths)} HTML files in filings/{ticker}/")
        all_chunks = []
        metadata = []
        for file_path in file_paths:
            chunks = parse_filing(file_path)
            all_chunks.extend(chunks)
            metadata.extend([{"file_path": file_path, "chunk_id": i} for i in range(len(chunks))])
        if not all_chunks:
            logger.warning(f"No chunks extracted from filings for {ticker}")
            return None
        vector_store = FAISS.from_texts(all_chunks, embeddings, metadatas=metadata)
        vector_store.save_local(f"filings/{ticker}/faiss_index")
        logger.info(f"Built FAISS index for {ticker} with {len(all_chunks)} chunks")
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
            logger.info(f"Pipeline completed for {ticker}")
        else:
            logger.warning(f"Pipeline completed for {ticker} but no vector store was created")
    except Exception as e:
        logger.error(f"Pipeline failed for {ticker}: {str(e)}")
        raise

if __name__ == "__main__":
    main()