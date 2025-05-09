import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.routes.filing_routes import router as filing_router
from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    LOG_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(filing_router)


# Exception handler for uncaught exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Uncaught exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error", "details": str(exc)}
    )


# Root endpoint
@app.get("/")
async def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "documentation": "/docs"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    # Run the application
    uvicorn.run("app.main:app", host="0.0.0.0", port=8005, reload=True)