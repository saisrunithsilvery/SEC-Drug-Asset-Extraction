from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List

from app.models.schemas import (
    FilingRequest,
    ProcessingStatus,
    VectorStoreRequest,
    DrugAnalysisRequest,
    DrugAnalysisResponse,
    ErrorResponse
)
from app.controllers.filing_controller import FilingController

# Create router
router = APIRouter(
    prefix="/api/v1",
    tags=["SEC Filings"],
    responses={
        404: {"model": ErrorResponse, "description": "Not found"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)


@router.post("/filings/fetch", response_model=ProcessingStatus, status_code=202)
async def fetch_filings(request: FilingRequest):
    """
    Fetch SEC filings for a given ticker and save to S3.
    """
    return await FilingController.process_filings(request)


@router.post("/filings/vectorize", response_model=ProcessingStatus, status_code=202)
async def build_vector_store(request: VectorStoreRequest):
    """
    Build vector store from existing SEC filings.
    """
    return await FilingController.build_vector_store(request)


@router.post("/filings/analyze", response_model=DrugAnalysisResponse)
async def analyze_drugs(request: DrugAnalysisRequest):
    """
    Analyze SEC filings to extract drug, program, and platform information.
    """
    return await FilingController.analyze_drugs(request)


@router.post("/filings/pipeline", response_model=DrugAnalysisResponse)
async def run_complete_pipeline(request: FilingRequest):
    """
    Run the complete pipeline: fetch filings, build vector store, and analyze drugs.
    """
    return await FilingController.process_complete_pipeline(request)