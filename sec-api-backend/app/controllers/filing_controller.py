import logging
from typing import Dict, Any, Optional, List
from fastapi import HTTPException

from app.models.schemas import (
    FilingRequest,
    ProcessingStatus,
    VectorStoreRequest,
    DrugAnalysisRequest,
    DrugAnalysisResponse
)
from app.services.filing_service import FilingService
from app.services.vector_store_service import VectorStoreService
from app.services.analysis_service import AnalysisService

# Configure logging
logger = logging.getLogger(__name__)


class FilingController:
    """Controller for SEC filing operations"""
    
    @staticmethod
    async def process_filings(request: FilingRequest) -> ProcessingStatus:
        """
        Process SEC filings for a given ticker.
        
        Args:
            request: Filing request model
            
        Returns:
            Processing status
        """
        try:
            logger.info(f"Processing filings for ticker: {request.ticker}")
            
            # Use the filing service to fetch and save filings
            job_id, metadata = FilingService.fetch_and_save_filings(
                ticker=request.ticker,
                form_types=request.form_types,
                max_filings=request.max_filings
            )
            
            # Return status
            return ProcessingStatus(
                status="success",
                message=f"Successfully processed {len(metadata)} filings for {request.ticker}",
                job_id=job_id,
                ticker=request.ticker
            )
        except ValueError as e:
            # Handle known value errors (e.g., invalid ticker)
            logger.error(f"Value error in processing filings: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Handle any other exceptions
            logger.error(f"Error processing filings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    
    @staticmethod
    async def build_vector_store(request: VectorStoreRequest) -> ProcessingStatus:
        """
        Build a vector store from existing filings.
        
        Args:
            request: Vector store request model
            
        Returns:
            Processing status
        """
        try:
            logger.info(f"Building vector store for ticker: {request.ticker}")
            
            # Use the vector store service to build the index
            job_id, success = VectorStoreService.build_vector_store(
                ticker=request.ticker
            )
            
            if not success:
                return ProcessingStatus(
                    status="warning",
                    message=f"No filings found or processed for {request.ticker}",
                    job_id=job_id,
                    ticker=request.ticker
                )
            
            # Return status
            return ProcessingStatus(
                status="success",
                message=f"Successfully built vector store for {request.ticker}",
                job_id=job_id,
                ticker=request.ticker
            )
        except FileNotFoundError as e:
            # Handle missing files
            logger.error(f"File not found: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            # Handle any other exceptions
            logger.error(f"Error building vector store: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    
    @staticmethod
    async def analyze_drugs(request: DrugAnalysisRequest) -> DrugAnalysisResponse:
        """
        Analyze filings to extract drug/program information.
        
        Args:
            request: Drug analysis request model
            
        Returns:
            Drug analysis response
        """
        try:
            logger.info(f"Analyzing drugs for ticker: {request.ticker}")
            
            # Create analysis service
            analysis_service = AnalysisService(ticker=request.ticker)
            
            # Run analysis
            results = analysis_service.analyze()
            
            # Return response
            return DrugAnalysisResponse(
                ticker=request.ticker,
                assets=results["assets"],
                s3_paths=results["s3_paths"]
            )
        except FileNotFoundError as e:
            # Handle missing files
            logger.error(f"File not found: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            # Handle value errors (e.g., missing API key)
            logger.error(f"Value error in drug analysis: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Handle any other exceptions
            logger.error(f"Error analyzing drugs: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    
    @staticmethod
    async def process_complete_pipeline(request: FilingRequest) -> DrugAnalysisResponse:
        """
        Run the complete pipeline: fetch filings, build vector store, and analyze drugs.
        
        Args:
            request: Filing request model
            
        Returns:
            Drug analysis response
        """
        try:
            logger.info(f"Running complete pipeline for ticker: {request.ticker}")
            
            # Step 1: Fetch filings
            job_id, metadata = FilingService.fetch_and_save_filings(
                ticker=request.ticker,
                form_types=request.form_types,
                max_filings=request.max_filings
            )
            
            if not metadata:
                raise ValueError(f"No filings found for {request.ticker}")
            
            # Step 2: Build vector store
            vector_job_id, success = VectorStoreService.build_vector_store(
                ticker=request.ticker
            )
            
            if not success:
                raise ValueError(f"Failed to build vector store for {request.ticker}")
            
            # Step 3: Analyze drugs
            analysis_service = AnalysisService(ticker=request.ticker)
            results = analysis_service.analyze()
            
            # Return response
            return DrugAnalysisResponse(
                ticker=request.ticker,
                assets=results["assets"],
                s3_paths=results["s3_paths"]
            )
        except ValueError as e:
            # Handle known value errors
            logger.error(f"Value error in pipeline: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Handle any other exceptions
            logger.error(f"Error in complete pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")