import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
API_TITLE = "SEC API Backend"
API_DESCRIPTION = "API for SEC filing retrieval, analysis and drug asset extraction"
API_VERSION = "0.1.0"

# SEC API Configuration
SEC_API_KEY = os.getenv("SEC_API_KEY")
USER_AGENT = os.getenv("USER_AGENT", "Company-Email user@example.com")

# S3 Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME", "k-cap-hfund")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hugging Face Embeddings model
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")

# File paths
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")
LOG_FILE = os.getenv("LOG_FILE", "sec_pipeline.log")

# Default parameters
DEFAULT_FILING_TYPES = ["10-K", "8-K"]
DEFAULT_FILING_COUNT = 100