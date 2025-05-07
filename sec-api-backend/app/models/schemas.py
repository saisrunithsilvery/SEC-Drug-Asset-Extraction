from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


class FilingRequest(BaseModel):
    """Request model for fetching SEC filings"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'WVE')")
    form_types: Optional[List[str]] = Field(None, description="List of form types to fetch (e.g., ['10-K', '8-K'])")
    max_filings: Optional[int] = Field(None, description="Maximum number of filings to fetch per form type")


class ProcessingStatus(BaseModel):
    """Response model for processing status"""
    status: str
    message: str
    job_id: str
    ticker: str


class VectorStoreRequest(BaseModel):
    """Request model for building vector store from filings"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'WVE')")


class DrugAnalysisRequest(BaseModel):
    """Request model for drug/program analysis"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'WVE')")


class DrugAsset(BaseModel):
    """Model for a drug asset"""
    name: str = Field(..., alias="Name/Number")
    mechanism: str = Field(..., alias="Mechanism of Action")
    targets: str = Field(..., alias="Target(s)")
    indication: str = Field(..., alias="Indication")
    preclinical_data: str = Field(..., alias="Animal Models/Preclinical Data")
    clinical_trials: str = Field(..., alias="Clinical Trials")
    upcoming_milestones: str = Field(..., alias="Upcoming Milestones")
    references: str = Field(..., alias="References")


class DrugAnalysisResponse(BaseModel):
    """Response model for drug/program analysis"""
    ticker: str
    assets: List[DrugAsset]
    summary: Optional[str] = None
    s3_paths: Dict[str, str]


class ErrorResponse(BaseModel):
    """Response model for errors"""
    status: str = "error"
    message: str
    details: Optional[Any] = None