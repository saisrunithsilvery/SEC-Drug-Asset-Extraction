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


class AnimalModel(BaseModel):
    """Model for animal model/preclinical data"""
    Model: str
    Key_Results: str = Field(..., alias="Key Results")
    Year: str


class TrialResults(BaseModel):
    """Model for clinical trial results"""
    Safety: str
    Efficacy: str


class ClinicalTrial(BaseModel):
    """Model for clinical trial data"""
    Phase: str
    N: str
    Duration: str
    Results: TrialResults
    Dates: str


class DrugAsset(BaseModel):
    """Model for a drug asset"""
    name: str = Field(..., alias="Name/Number")
    mechanism: str = Field(..., alias="Mechanism_of_Action")
    targets: str = Field(..., alias="Target")
    indication: str = Field(..., alias="Indication")
    preclinical_data: List[AnimalModel] = Field(..., alias="Animal_Models_Preclinical_Data")
    clinical_trials: List[ClinicalTrial] = Field(..., alias="Clinical_Trials")
    upcoming_milestones: List[str] = Field(..., alias="Upcoming_Milestones")
    references: List[str] = Field(..., alias="References")


class DrugAnalysisResponse(BaseModel):
    """Response model for drug/program analysis"""
    ticker: str
    assets: List[DrugAsset]
    s3_paths: Dict[str, str]


class ErrorResponse(BaseModel):
    """Response model for errors"""
    status: str = "error"
    message: str
    details: Optional[Any] = None