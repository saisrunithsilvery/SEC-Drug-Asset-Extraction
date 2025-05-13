import os
import logging
import tempfile
import shutil
import json
import uuid
import traceback
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from tabulate import tabulate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.retrieval_qa.base import RetrievalQA
import io  # Add this import at the top with other imports

from app.utils.s3_utils import (
    download_s3_folder,
    upload_to_s3
)
from app.config import (
    OPENAI_API_KEY,
    EMBEDDINGS_MODEL
)

# Configure logging
logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for analyzing SEC filings and extracting drug information"""
    
    def __init__(self, ticker: str):
        """
        Initialize the analysis service.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'WVE')
        """
        self.ticker = ticker
        self.temp_dir = None
        self.vector_store = None
        self.embeddings = None
        self.job_id = str(uuid.uuid4())
    
    def _validate_api_key(self) -> str:
        """
        Check if a valid OpenAI API key is set.
        
        Returns:
            OpenAI API key
        """
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_api_key"):
            raise ValueError("Invalid or missing OPENAI_API_KEY. Please set a valid API key.")
        return OPENAI_API_KEY
    
    def _load_vector_store(self) -> None:
        """
        Load FAISS index from S3 bucket for the ticker.
        """
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
            
            # Create a temporary directory to download the FAISS index
            self.temp_dir = tempfile.mkdtemp()
            
            # Define S3 path for the FAISS index
            s3_prefix = f"filings/{self.ticker}/faiss_index"
            local_index_path = os.path.join(self.temp_dir, "faiss_index")
            
            # Download the index from S3
            success = download_s3_folder(s3_prefix, local_index_path)
            
            if not success:
                logger.error(f"Failed to download FAISS index for {self.ticker} from S3")
                raise FileNotFoundError(f"No FAISS index found for {self.ticker} in S3")
            
            # Load the index
            self.vector_store = FAISS.load_local(
                local_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded FAISS index for {self.ticker} from S3")
        except Exception as e:
            logger.error(f"Failed to load FAISS index for {self.ticker}: {str(e)}")
            # Clean up temporary directory if loading failed
            self._clean_up()
            raise
    
    def _clean_up(self) -> None:
        """
        Clean up temporary directory.
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            self.temp_dir = None
    
    def identify_assets(self) -> List[str]:
        """
        Identify drugs, programs, and platforms using a targeted prompt.
        
        Returns:
            List of identified assets
        """
        try:
            if not self.vector_store:
                self._load_vector_store()
            
            # Validate OpenAI API key
            self._validate_api_key()
            
            # Simplified and strict prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """
        You are an expert in biotech SEC filings analysis. From the provided context, identify all drugs the company is developing. Be thorough and comprehensive in your search.

        Return a flat JSON list of unique drug names/identifiers using this format: ["Drug1", "Drug2"].

        Include:
        - Drug names and numbers (e.g., WVE-4, WVE-N531)

        Do NOT include:
        - Program names (e.g., PRECISION-HD, PRISM, RNA editing program)
        - Platform technologies (e.g., PN backbone chemistry, GalNAc-AIMer)
        - Specific technologies mentioned as company assets (e.g., allele-selective targeting)
        - Disease names or indications alone
        - General technical terms not specific to company programs
        - Target molecules (unless they are also program names)

        Ensure the output is valid JSON. If no assets are found, return an empty list [].
        When examining the context, look for patterns like:
        - Names followed by mechanism descriptions
        - Assets mentioned in clinical trial updates

        Be thorough and pick up all potential assets across the entire context.
        """),

                ("human", "Context: {context}\nReturn a flat JSON list of all drug names.")
            ])
            
            # Create RAG chain
            llm = ChatOpenAI(
                model_name="gpt-4.1-mini",
                openai_api_key=OPENAI_API_KEY
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 40}),
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            # Run query
            query = f"Identify all drug names for {self.ticker}"
            response = qa_chain.invoke({"query": query})
            raw_result = response.get("result", "[]")
            
            # Clean response
            raw_result = self._clean_json_response(raw_result)
            logger.info(f"Cleaned response for {self.ticker}: {raw_result}")
            
            # Parse JSON
            try:
                assets = json.loads(raw_result)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for {self.ticker}: {str(e)}")
                logger.error(f"Problematic response: {raw_result}")
                return []
            
            # Flatten if necessary
            flattened_assets = []
            if isinstance(assets, dict):
                logger.warning(f"Received grouped assets for {self.ticker}: {assets}. Flattening to list.")
                for category, items in assets.items():
                    if isinstance(items, list):
                        flattened_assets.extend(items)
            else:
                flattened_assets = assets
            
            # Ensure uniqueness
            flattened_assets = list(set(flattened_assets))
            logger.info(f"Identified {len(flattened_assets)} unique assets for {self.ticker}: {flattened_assets}")
            return flattened_assets
        except Exception as e:
            logger.error(f"Error identifying assets for {self.ticker}: {str(e)}")
            raise
    
    def _normalize_complex_value(self, value: Any) -> Any:
        """
        Process complex data types to ensure they match the expected structured format.
        
        Args:
            value: Value to normalize
            
        Returns:
            Properly structured value (may be a complex object, not just a string)
        """
        if value is None:
            return "Not specified"
        
        if isinstance(value, str):
            # If the value is a string but should be a complex structure, try to parse it
            if value != "Not specified":
                try:
                    # If it looks like JSON, try to parse it
                    if (value.strip().startswith('[') and value.strip().endswith(']')) or \
                       (value.strip().startswith('{') and value.strip().endswith('}')):
                        parsed = json.loads(value)
                        return parsed
                except json.JSONDecodeError:
                    # If parsing fails, return the original string
                    return value
            return value
        
        # Keep complex objects as-is
        if isinstance(value, (dict, list)):
            return value
        
        # Convert other types to string
        return str(value)
    
    def extract_asset_details(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Extract detailed information for a specific drug/program.

        Args:
            asset: Asset name or identifier

        Returns:
            Dictionary of asset details
        """
        try:
            if not self.vector_store:
                self._load_vector_store()

            # Validate OpenAI API key
            self._validate_api_key()

            # IMPORTANT: The original approach using LangChain chains and retrievers has an issue
            # The error is: 'dict' object has no attribute 'replace' when trying to invoke the chain
            # This happens because we're passing an empty dictionary as input
            # Let's use a more direct approach that avoids this issue

            # We'll use the vector store directly to get relevant document chunks
            logger.info(f"Fetching relevant documents for {asset} from vector store")
            docs = self.vector_store.similarity_search(asset, k=40)
            
            # Combine the document chunks into a context
            context = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Retrieved {len(docs)} document chunks for context")
            
            # Create a system prompt that doesn't use any 'Model' or 'Phase' words that could
            # be confused as template variables
            system_prompt = (
                "You are an expert in biotech SEC filings and drug development. "
                "Extract all available information about the specified drug from the provided context. "
                f"I need details about: {asset}\n\n"
                "Format your response as a JSON object with these exact fields:\n"
                "{\n"
                '  "Name/Number": "The drug identifier",\n'
                '  "Mechanism_of_Action": "How the drug works only the ans like antisense oligonucleotide (ASO) ",\n'
                '  "Target": "The biological target",\n'
                '  "Indication": "The disease being treated",\n'
                '  "Animal_Models_Preclinical_Data": [\n'
                '    {\n'
                '      "Model": "The animal model used",\n'
                '      "Key Results": "Main findings",\n'
                '      "Year": "When conducted"\n'
                '    }\n'
                '  ],\n'
                '  "Clinical_Trials": [\n'
                '    {\n'
                '      "Phase": "Trial phase",\n'
                '      "N": "Number of patients",\n'
                '      "Duration": "Trial duration",\n'
                '      "Results": {\n'
                '        "Safety": "Safety data",\n'
                '        "Efficacy": "Efficacy data"\n'
                '      },\n'
                '      "Dates": "When conducted"\n'
                '    }\n'
                '  ],\n'
                '  "Upcoming_Milestones": ["Future event 1", "Future event 2"],\n'
                '  "References": ["Reference 1", "Reference 2"]\n'
                '}\n\n'
                "If information is unavailable for a field, use \"Not specified\" for text fields "
                "or provide empty arrays with default placeholder items for the array fields. "
                "Be thorough - include all information from the context about this drug."
            )

            # Call OpenAI directly rather than using LangChain chains
            logger.info(f"Calling OpenAI to extract details for {asset}")
            openai_client = ChatOpenAI(
                model_name="gpt-4.1-mini",
                openai_api_key=OPENAI_API_KEY
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nDrug to extract details for: {asset}"}
            ]
            response = openai_client.invoke(messages)
            raw_result = response.content
            logger.info(f"Received response from OpenAI for {asset}")

            # Clean and parse JSON
            raw_result = self._clean_json_response(raw_result)
            logger.info(f"Cleaned JSON response for {asset}")

            try:
                details = json.loads(raw_result.strip())
                
                # Ensure Name/Number is consistent
                if details.get("Name/Number") != asset and asset != "":
                    logger.warning(f"Name mismatch: Asset={asset}, Returned Name={details.get('Name/Number')}")
                    details["Name/Number"] = asset

                # Ensure all fields have proper structure
                logger.info(f"Normalizing field structures for {asset}")
                self._ensure_proper_field_structure(details)

                logger.info(f"Successfully extracted details for {asset}")
                return details
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {asset}: {str(e)}")
                logger.error(f"Problematic JSON: {raw_result}")

                # Return a fallback object
                return self._create_fallback_details(asset)
        except Exception as e:
            logger.error(f"Error extracting details for {asset} in {self.ticker}: {str(e)}")
            import traceback
            logger.error(f"Detailed traceback: {traceback.format_exc()}")
            return self._create_fallback_details(asset)

    def _ensure_proper_field_structure(self, details: Dict[str, Any]) -> None:
        """
        Ensures all fields have the expected structure.
        
        Args:
            details: The details dictionary to normalize
        """
        # Ensure basic fields exist
        for field in ["Name/Number", "Mechanism_of_Action", "Target", "Indication"]:
            if field not in details or not details[field]:
                details[field] = "Not specified"
        
        # Normalize Animal_Models_Preclinical_Data
        if "Animal_Models_Preclinical_Data" not in details or not details["Animal_Models_Preclinical_Data"]:
            details["Animal_Models_Preclinical_Data"] = [
                {
                    "Model": "Not specified",
                    "Key Results": "Not specified",
                    "Year": "Not specified"
                }
            ]
        elif not isinstance(details["Animal_Models_Preclinical_Data"], list):
            # Try to parse if it's a string
            try:
                if isinstance(details["Animal_Models_Preclinical_Data"], str):
                    parsed = json.loads(details["Animal_Models_Preclinical_Data"])
                    if isinstance(parsed, list):
                        details["Animal_Models_Preclinical_Data"] = parsed
                    else:
                        details["Animal_Models_Preclinical_Data"] = [
                            {
                                "Model": "Not specified",
                                "Key Results": "Not specified",
                                "Year": "Not specified"
                            }
                        ]
            except:
                details["Animal_Models_Preclinical_Data"] = [
                    {
                        "Model": "Not specified",
                        "Key Results": "Not specified",
                        "Year": "Not specified"
                    }
                ]
        
        # Ensure each animal model item has required fields
        for i, item in enumerate(details["Animal_Models_Preclinical_Data"]):
            if not isinstance(item, dict):
                details["Animal_Models_Preclinical_Data"][i] = {
                    "Model": str(item),
                    "Key Results": "Not specified",
                    "Year": "Not specified"
                }
            else:
                if "Model" not in item:
                    item["Model"] = "Not specified"
                if "Key Results" not in item:
                    item["Key Results"] = "Not specified"
                if "Year" not in item:
                    item["Year"] = "Not specified"
        
        # Normalize Clinical_Trials
        if "Clinical_Trials" not in details or not details["Clinical_Trials"]:
            details["Clinical_Trials"] = [
                {
                    "Phase": "Not specified",
                    "N": "Not specified",
                    "Duration": "Not specified",
                    "Results": {
                        "Safety": "Not specified",
                        "Efficacy": "Not specified"
                    },
                    "Dates": "Not specified"
                }
            ]
        elif not isinstance(details["Clinical_Trials"], list):
            # Try to parse if it's a string
            try:
                if isinstance(details["Clinical_Trials"], str):
                    parsed = json.loads(details["Clinical_Trials"])
                    if isinstance(parsed, list):
                        details["Clinical_Trials"] = parsed
                    else:
                        details["Clinical_Trials"] = [
                            {
                                "Phase": "Not specified",
                                "N": "Not specified",
                                "Duration": "Not specified",
                                "Results": {
                                    "Safety": "Not specified",
                                    "Efficacy": "Not specified"
                                },
                                "Dates": "Not specified"
                            }
                        ]
            except:
                details["Clinical_Trials"] = [
                    {
                        "Phase": "Not specified",
                        "N": "Not specified",
                        "Duration": "Not specified",
                        "Results": {
                            "Safety": "Not specified",
                            "Efficacy": "Not specified"
                        },
                        "Dates": "Not specified"
                    }
                ]
        
        # Ensure each clinical trial item has required fields
        for i, item in enumerate(details["Clinical_Trials"]):
            if not isinstance(item, dict):
                details["Clinical_Trials"][i] = {
                    "Phase": str(item),
                    "N": "Not specified",
                    "Duration": "Not specified",
                    "Results": {
                        "Safety": "Not specified",
                        "Efficacy": "Not specified"
                    },
                    "Dates": "Not specified"
                }
            else:
                if "Phase" not in item:
                    item["Phase"] = "Not specified"
                if "N" not in item:
                    item["N"] = "Not specified"
                if "Duration" not in item:
                    item["Duration"] = "Not specified"
                if "Dates" not in item:
                    item["Dates"] = "Not specified"
                
                # Ensure Results has proper structure
                if "Results" not in item or not item["Results"]:
                    item["Results"] = {
                        "Safety": "Not specified",
                        "Efficacy": "Not specified"
                    }
                elif not isinstance(item["Results"], dict):
                    item["Results"] = {
                        "Safety": "Not specified",
                        "Efficacy": "Not specified"
                    }
                else:
                    if "Safety" not in item["Results"]:
                        item["Results"]["Safety"] = "Not specified"
                    if "Efficacy" not in item["Results"]:
                        item["Results"]["Efficacy"] = "Not specified"
        
        # Normalize Upcoming_Milestones
        if "Upcoming_Milestones" not in details or not details["Upcoming_Milestones"]:
            details["Upcoming_Milestones"] = ["Not specified"]
        elif not isinstance(details["Upcoming_Milestones"], list):
            if details["Upcoming_Milestones"] == "Not specified":
                details["Upcoming_Milestones"] = ["Not specified"]
            else:
                details["Upcoming_Milestones"] = [str(details["Upcoming_Milestones"])]
        elif len(details["Upcoming_Milestones"]) == 0:
            details["Upcoming_Milestones"] = ["Not specified"]
        
        # Normalize References
        if "References" not in details or not details["References"]:
            details["References"] = ["Not specified"]
        elif not isinstance(details["References"], list):
            if details["References"] == "Not specified":
                details["References"] = ["Not specified"]
            else:
                details["References"] = [str(details["References"])]
        elif len(details["References"]) == 0:
            details["References"] = ["Not specified"]

    def _create_fallback_details(self, asset: str) -> Dict[str, Any]:
        """
        Creates a fallback details object with proper structure.
        
        Args:
            asset: The asset name/identifier
            
        Returns:
            A properly structured fallback details dictionary
        """
        return {
            "Name/Number": asset,
            "Mechanism_of_Action": "Not specified",
            "Target": "Not specified",
            "Indication": "Not specified",
            "Animal_Models_Preclinical_Data": [
                {
                    "Model": "Not specified", 
                    "Key Results": "Not specified", 
                    "Year": "Not specified"
                }
            ],
            "Clinical_Trials": [
                {
                    "Phase": "Not specified",
                    "N": "Not specified",
                    "Duration": "Not specified",
                    "Results": {
                        "Safety": "Not specified",
                        "Efficacy": "Not specified"
                    },
                    "Dates": "Not specified"
                }
            ],
            "Upcoming_Milestones": ["Not specified"],
            "References": ["Not specified"]
        }
            
    def _normalize_field_structures(self, details: Dict[str, Any]) -> None:
        """
        Ensures all fields have the expected structure.
        """
        logger.info("Normalizing Animal_Models_Preclinical_Data field")
        # Normalize Animal_Models_Preclinical_Data
        try:
            if not isinstance(details.get("Animal_Models_Preclinical_Data"), list):
                logger.info("Converting Animal_Models_Preclinical_Data to list")
                if details.get("Animal_Models_Preclinical_Data") == "Not specified":
                    details["Animal_Models_Preclinical_Data"] = [
                        {
                            "Model": "Not specified",
                            "Key Results": "Not specified",
                            "Year": "Not specified"
                        }
                    ]
                else:
                    # Try to parse if it's a JSON string
                    try:
                        if isinstance(details.get("Animal_Models_Preclinical_Data"), str):
                            parsed = json.loads(details.get("Animal_Models_Preclinical_Data"))
                            if isinstance(parsed, list):
                                details["Animal_Models_Preclinical_Data"] = parsed
                            else:
                                details["Animal_Models_Preclinical_Data"] = [
                                    {
                                        "Model": "Not specified",
                                        "Key Results": "Not specified",
                                        "Year": "Not specified"
                                    }
                                ]
                    except:
                        details["Animal_Models_Preclinical_Data"] = [
                            {
                                "Model": "Not specified",
                                "Key Results": "Not specified",
                                "Year": "Not specified"
                            }
                        ]
        except Exception as e:
            logger.error(f"Error normalizing Animal_Models_Preclinical_Data: {str(e)}")
            details["Animal_Models_Preclinical_Data"] = [
                {
                    "Model": "Not specified",
                    "Key Results": "Not specified",
                    "Year": "Not specified"
                }
            ]
            
        logger.info("Normalizing Clinical_Trials field")
        # Normalize Clinical_Trials
        try:
            if not isinstance(details.get("Clinical_Trials"), list):
                logger.info("Converting Clinical_Trials to list")
                if details.get("Clinical_Trials") == "Not specified":
                    details["Clinical_Trials"] = [
                        {
                            "Phase": "Not specified",
                            "N": "Not specified",
                            "Duration": "Not specified",
                            "Results": {
                                "Safety": "Not specified",
                                "Efficacy": "Not specified"
                            },
                            "Dates": "Not specified"
                        }
                ]
                else:
                    # Try to parse if it's a JSON string
                    try:
                        if isinstance(details.get("Clinical_Trials"), str):
                            parsed = json.loads(details.get("Clinical_Trials"))
                            if isinstance(parsed, list):
                                details["Clinical_Trials"] = parsed
                            else:
                                details["Clinical_Trials"] = [
                                    {
                                        "Phase": "Not specified",
                                        "N": "Not specified",
                                        "Duration": "Not specified",
                                        "Results": {
                                            "Safety": "Not specified",
                                            "Efficacy": "Not specified"
                                        },
                                        "Dates": "Not specified"
                                    }
                                ]
                    except:
                        details["Clinical_Trials"] = [
                            {
                                "Phase": "Not specified",
                                "N": "Not specified",
                                "Duration": "Not specified",
                                "Results": {
                                    "Safety": "Not specified",
                                    "Efficacy": "Not specified"
                                },
                                "Dates": "Not specified"
                            }
                        ]
        except Exception as e:
            logger.error(f"Error normalizing Clinical_Trials: {str(e)}")
            details["Clinical_Trials"] = [
                {
                    "Phase": "Not specified",
                    "N": "Not specified",
                    "Duration": "Not specified",
                    "Results": {
                        "Safety": "Not specified",
                        "Efficacy": "Not specified"
                    },
                    "Dates": "Not specified"
                }
            ]
        
        logger.info("Normalizing Upcoming_Milestones field")
        # Normalize Upcoming_Milestones
        try:
            if not isinstance(details.get("Upcoming_Milestones"), list):
                logger.info("Converting Upcoming_Milestones to list")
                if details.get("Upcoming_Milestones") == "Not specified":
                    details["Upcoming_Milestones"] = ["Not specified"]
                elif details.get("Upcoming_Milestones") is None:
                    details["Upcoming_Milestones"] = ["Not specified"]
                else:
                    # Try to parse if it's a JSON string
                    try:
                        if isinstance(details.get("Upcoming_Milestones"), str):
                            parsed = json.loads(details.get("Upcoming_Milestones"))
                            if isinstance(parsed, list):
                                details["Upcoming_Milestones"] = parsed
                            else:
                                details["Upcoming_Milestones"] = [str(details.get("Upcoming_Milestones"))]
                    except:
                        details["Upcoming_Milestones"] = [str(details.get("Upcoming_Milestones"))]
        except Exception as e:
            logger.error(f"Error normalizing Upcoming_Milestones: {str(e)}")
            details["Upcoming_Milestones"] = ["Not specified"]
            
        logger.info("Normalizing References field")
        # Normalize References
        try:
            if not isinstance(details.get("References"), list):
                logger.info("Converting References to list")
                if details.get("References") == "Not specified":
                    details["References"] = ["Not specified"]
                elif details.get("References") is None:
                    details["References"] = ["Not specified"]
                else:
                    # Try to parse if it's a JSON string
                    try:
                        if isinstance(details.get("References"), str):
                            parsed = json.loads(details.get("References"))
                            if isinstance(parsed, list):
                                details["References"] = parsed
                            else:
                                details["References"] = [str(details.get("References"))]
                    except:
                        details["References"] = [str(details.get("References"))]
        except Exception as e:
            logger.error(f"Error normalizing References: {str(e)}")
            details["References"] = ["Not specified"]
    
    def consolidate_data(self, asset_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge data for each asset, resolving conflicts and combining references.
        
        Args:
            asset_details: List of asset detail dictionaries
            
        Returns:
            Consolidated list of asset details
        """
        try:
            # Initialize merged data structure with properly structured fields
            merged_data = defaultdict(lambda: {
                "Name/Number": None,
                "Mechanism_of_Action": "Not specified",
                "Target": "Not specified",
                "Indication": "Not specified",
                "Animal_Models_Preclinical_Data": [
                    {
                        "Model": "Not specified",
                        "Key Results": "Not specified",
                        "Year": "Not specified"
                    }
                ],
                "Clinical_Trials": [
                    {
                        "Phase": "Not specified",
                        "N": "Not specified",
                        "Duration": "Not specified",
                        "Results": {
                            "Safety": "Not specified",
                            "Efficacy": "Not specified"
                        },
                        "Dates": "Not specified"
                    }
                ],
                "Upcoming_Milestones": ["Not specified"],
                "References": []
            })
            
            # Track asset names for debugging
            asset_names = [details.get("Name/Number", "Unknown") for details in asset_details if details]
            logger.info(f"Consolidating assets: {asset_names}")
            
            # Process each asset detail
            for details in asset_details:
                if not details:
                    continue
                
                name = details.get("Name/Number", "Unknown")
                logger.info(f"Processing asset: {name}")
                
                # Process complex values to ensure they maintain the right structure
                processed_details = details.copy()
                for key in processed_details.keys():
                    processed_details[key] = self._normalize_complex_value(processed_details[key])
                
                # Update fields with meaningful values (skip "Not specified")
                if processed_details.get("Mechanism_of_Action") != "Not specified":
                    merged_data[name]["Mechanism_of_Action"] = processed_details.get("Mechanism_of_Action")
                    
                if processed_details.get("Target") != "Not specified":
                    merged_data[name]["Target"] = processed_details.get("Target")
                    
                if processed_details.get("Indication") != "Not specified":
                    merged_data[name]["Indication"] = processed_details.get("Indication")
                
                # Handle structured data fields
                animal_models = processed_details.get("Animal_Models_Preclinical_Data")
                if animal_models and animal_models != "Not specified":
                    if isinstance(animal_models, list) and len(animal_models) > 0 and animal_models[0].get("Model") != "Not specified":
                        merged_data[name]["Animal_Models_Preclinical_Data"] = animal_models
                
                clinical_trials = processed_details.get("Clinical_Trials")
                if clinical_trials and clinical_trials != "Not specified":
                    if isinstance(clinical_trials, list) and len(clinical_trials) > 0 and clinical_trials[0].get("Phase") != "Not specified":
                        merged_data[name]["Clinical_Trials"] = clinical_trials
                
                milestones = processed_details.get("Upcoming_Milestones")
                if milestones and milestones != "Not specified":
                    if isinstance(milestones, list) and len(milestones) > 0 and milestones[0] != "Not specified":
                        merged_data[name]["Upcoming_Milestones"] = milestones
                
                # Handle references properly
                ref = processed_details.get("References", "")
                if isinstance(ref, list):
                    merged_data[name]["References"].extend(ref)
                elif ref and ref != "Not specified":
                    merged_data[name]["References"].append(ref)
            
            # Clean up references (remove duplicates)
            for name, data in merged_data.items():
                if isinstance(data["References"], list):
                    # Convert any non-string references to strings
                    cleaned_refs = []
                    for ref in data["References"]:
                        if not isinstance(ref, str):
                            ref = str(ref)
                        if ref and ref != "Not specified":
                            cleaned_refs.append(ref)
                    
                    # Remove duplicates
                    data["References"] = list(set(cleaned_refs))
                    # If no references, set a default
                    if not data["References"]:
                        data["References"] = ["Not specified"]
                else:
                    # Make sure References is always a list
                    data["References"] = ["Not specified"]
            
            # Create final data structure for output, ensuring we keep complex structure
            final_data = []
            for name, data in merged_data.items():
                final_data.append({
                    "Name/Number": name,
                    "Mechanism_of_Action": data["Mechanism_of_Action"],
                    "Target": data["Target"],
                    "Indication": data["Indication"],
                    "Animal_Models_Preclinical_Data": data["Animal_Models_Preclinical_Data"],
                    "Clinical_Trials": data["Clinical_Trials"],
                    "Upcoming_Milestones": data["Upcoming_Milestones"],
                    "References": data["References"]
                })
            
            logger.info(f"Consolidated data for {len(final_data)} unique assets")
            return final_data
        except Exception as e:
            logger.error(f"Error consolidating data: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def df_to_markdown_with_delimiters(self, df):
        # Get column names and data
        columns = df.columns.tolist()
        data = df.values.tolist()
        
        # Create header row
        markdown = "| " + " | ".join([str(col).replace('_', ' ') for col in columns]) + " |\n"
        
        # Create separator row
        markdown += "| " + " | ".join(["---" for _ in columns]) + " |\n"
        
        # Create data rows
        for row in data:
            # Convert all values to strings, handling complex objects
            row_str = []
            for cell in row:
                if isinstance(cell, (dict, list)):
                    cell_str = json.dumps(cell, indent=2).replace('\n', ' ').replace('\r', '')
                else:
                    cell_str = str(cell).replace('\n', ' ').replace('\r', '')
                row_str.append(cell_str)
                
            markdown += "| " + " | ".join(row_str) + " |\n"
    
        return markdown
       
    def save_output(self, drug_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Save drug data as CSV and Markdown with additional summaries to S3.
        
        Args:
            drug_data: List of drug data dictionaries
            
        Returns:
            Dictionary of S3 paths for output files
        """
        try:
            # Log the data before saving
            logger.info(f"Saving {len(drug_data)} assets to CSV and Markdown in S3")
            
            # S3 output directory
            s3_output_dir = f"filings/{self.ticker}"
            
            # Deep copy the data to avoid modifying the original
            json_data = json.loads(json.dumps(drug_data))
            
            # Create DataFrame - we need to stringify complex fields for CSV
            df_data = []
            for drug in json_data:
                drug_row = drug.copy()
                # Convert complex fields to JSON strings for CSV
                for key, value in drug_row.items():
                    if isinstance(value, (dict, list)):
                        drug_row[key] = json.dumps(value)
                df_data.append(drug_row)
                
            df = pd.DataFrame(df_data)
            
            # Save CSV to S3
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_key = f"{s3_output_dir}/drug_summary.csv"
            upload_to_s3(csv_buffer.getvalue(), csv_key)
            logger.info(f"Saved CSV to S3 for {self.ticker}")
            
            # Create Markdown content
            markdown_table = self.df_to_markdown_with_delimiters(df)
            markdown_content = f"# {self.ticker.upper()} Drug Development Pipeline\n\n{markdown_table}\n\n"
            
            # Add disclaimer
            markdown_content += "*Note: Data may be incomplete due to limitations in filings or processing.*"
            
            # Save Markdown to S3
            md_key = f"{s3_output_dir}/drug_summary.md"
            upload_to_s3(markdown_content, md_key)
            logger.info(f"Saved Markdown to S3 for {self.ticker}")
            
            # Return paths to output files
            return {
                "csv": f"s3://{s3_output_dir}/drug_summary.csv",
                "markdown": f"s3://{s3_output_dir}/drug_summary.md"
            }
        except Exception as e:
            logger.error(f"Error saving output to S3 for {self.ticker}: {str(e)}")
            raise
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run the full analysis pipeline for the ticker.
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Step 1: Identify assets
            assets = self.identify_assets()
            
            if not assets:
                logger.warning(f"No assets identified for {self.ticker}")
                return {
                    "ticker": self.ticker,
                    "job_id": self.job_id,
                    "assets": [],
                    "s3_paths": {}
                }
            
            # Step 2: Extract details for each asset
            asset_details = []
            for asset in assets:
                details = self.extract_asset_details(asset)
                if details:
                    asset_details.append(details)
            
            if not asset_details:
                logger.warning(f"No details extracted for {self.ticker}")
                return {
                    "ticker": self.ticker,
                    "job_id": self.job_id,
                    "assets": [],
                    "s3_paths": {}
                }
            
            # Step 3: Consolidate data
            consolidated_data = self.consolidate_data(asset_details)
            
            # Step 4: Save output to S3
            s3_paths = self.save_output(consolidated_data)
            
            # Return results
            return {
                "ticker": self.ticker,
                "job_id": self.job_id,
                "assets": consolidated_data,
                "s3_paths": s3_paths
            }
        except Exception as e:
            logger.error(f"Analysis failed for {self.ticker}: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            self._clean_up()
    
    @staticmethod
    def _clean_json_response(text: str) -> str:
        """
        Clean a JSON response to make it parseable.
        
        Args:
            text: Raw JSON response text
            
        Returns:
            Cleaned JSON string
        """
        # Remove code block markers if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Clean invalid control characters
        text = text.replace('\r', '').replace('\n', ' ').replace('\t', ' ')
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch == '\n')
        
        # Try to extract JSON from non-JSON text
        if not (text.strip().startswith("[") or text.strip().startswith("{")):
            import re
            json_match = re.search(r'\[.*?\]|\{.*?\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
            else:
                logger.error(f"No valid JSON in response: {text}")
                return "[]"
        
        return text.strip()
    def consolidate_data(self, asset_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge data for each asset, resolving conflicts and combining references.
        
        Args:
            asset_details: List of asset detail dictionaries
            
        Returns:
            Consolidated list of asset details
        """
        try:
            # Initialize merged data structure with properly structured fields
            merged_data = defaultdict(lambda: {
                "Name/Number": None,
                "Mechanism_of_Action": "Not specified",
                "Target": "Not specified",
                "Indication": "Not specified",
                "Animal_Models_Preclinical_Data": [
                    {
                        "Model": "Not specified",
                        "Key Results": "Not specified",
                        "Year": "Not specified"
                    }
                ],
                "Clinical_Trials": [
                    {
                        "Phase": "Not specified",
                        "N": "Not specified",
                        "Duration": "Not specified",
                        "Results": {
                            "Safety": "Not specified",
                            "Efficacy": "Not specified"
                        },
                        "Dates": "Not specified"
                    }
                ],
                "Upcoming_Milestones": ["Not specified"],
                "References": []
            })
            
            # Track asset names for debugging
            asset_names = [details.get("Name/Number", "Unknown") for details in asset_details if details]
            logger.info(f"Consolidating assets: {asset_names}")
            
            # Process each asset detail
            for details in asset_details:
                if not details:
                    continue
                
                name = details.get("Name/Number", "Unknown")
                logger.info(f"Processing asset: {name}")
                
                # Process complex values to ensure they maintain the right structure
                processed_details = details.copy()
                for key in processed_details.keys():
                    processed_details[key] = self._normalize_complex_value(processed_details[key])
                
                # Update fields with meaningful values (skip "Not specified")
                if processed_details.get("Mechanism_of_Action") != "Not specified":
                    merged_data[name]["Mechanism_of_Action"] = processed_details.get("Mechanism_of_Action")
                    
                if processed_details.get("Target") != "Not specified":
                    merged_data[name]["Target"] = processed_details.get("Target")
                    
                if processed_details.get("Indication") != "Not specified":
                    merged_data[name]["Indication"] = processed_details.get("Indication")
                
                # Handle structured data fields
                animal_models = processed_details.get("Animal_Models_Preclinical_Data")
                if animal_models and animal_models != "Not specified":
                    if isinstance(animal_models, list) and len(animal_models) > 0 and animal_models[0].get("Model") != "Not specified":
                        merged_data[name]["Animal_Models_Preclinical_Data"] = animal_models
                
                clinical_trials = processed_details.get("Clinical_Trials")
                if clinical_trials and clinical_trials != "Not specified":
                    if isinstance(clinical_trials, list) and len(clinical_trials) > 0 and clinical_trials[0].get("Phase") != "Not specified":
                        merged_data[name]["Clinical_Trials"] = clinical_trials
                
                milestones = processed_details.get("Upcoming_Milestones")
                if milestones and milestones != "Not specified":
                    if isinstance(milestones, list) and len(milestones) > 0 and milestones[0] != "Not specified":
                        merged_data[name]["Upcoming_Milestones"] = milestones
                
                # Handle references properly
                ref = processed_details.get("References", "")
                if isinstance(ref, list):
                    merged_data[name]["References"].extend(ref)
                elif ref and ref != "Not specified":
                    merged_data[name]["References"].append(ref)
            
            # Clean up references (remove duplicates)
            for name, data in merged_data.items():
                if isinstance(data["References"], list):
                    # Convert any non-string references to strings
                    cleaned_refs = []
                    for ref in data["References"]:
                        if not isinstance(ref, str):
                            ref = str(ref)
                        if ref and ref != "Not specified":
                            cleaned_refs.append(ref)
                    
                    # Remove duplicates
                    data["References"] = list(set(cleaned_refs))
                    # If no references, set a default
                    if not data["References"]:
                        data["References"] = ["Not specified"]
                else:
                    # Make sure References is always a list
                    data["References"] = ["Not specified"]
            
            # Create final data structure for output, ensuring we keep complex structure
            final_data = []
            for name, data in merged_data.items():
                final_data.append({
                    "Name/Number": name,
                    "Mechanism_of_Action": data["Mechanism_of_Action"],
                    "Target": data["Target"],
                    "Indication": data["Indication"],
                    "Animal_Models_Preclinical_Data": data["Animal_Models_Preclinical_Data"],
                    "Clinical_Trials": data["Clinical_Trials"],
                    "Upcoming_Milestones": data["Upcoming_Milestones"],
                    "References": data["References"]
                })
            
            logger.info(f"Consolidated data for {len(final_data)} unique assets")
            return final_data
        except Exception as e:
            logger.error(f"Error consolidating data: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def df_to_markdown_with_delimiters(self, df):
        # Get column names and data
        columns = df.columns.tolist()
        data = df.values.tolist()
        
        # Create header row
        markdown = "| " + " | ".join([str(col).replace('_', ' ') for col in columns]) + " |\n"
        
        # Create separator row
        markdown += "| " + " | ".join(["---" for _ in columns]) + " |\n"
        
        # Create data rows
        for row in data:
            # Convert all values to strings, handling complex objects
            row_str = []
            for cell in row:
                if isinstance(cell, (dict, list)):
                    cell_str = json.dumps(cell, indent=2).replace('\n', ' ').replace('\r', '')
                else:
                    cell_str = str(cell).replace('\n', ' ').replace('\r', '')
                row_str.append(cell_str)
                
            markdown += "| " + " | ".join(row_str) + " |\n"
    
        return markdown
       
    def save_output(self, drug_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Save drug data as CSV and Markdown with additional summaries to S3.
        
        Args:
            drug_data: List of drug data dictionaries
            
        Returns:
            Dictionary of S3 paths for output files
        """
        try:
            # Log the data before saving
            logger.info(f"Saving {len(drug_data)} assets to CSV and Markdown in S3")
            
            # S3 output directory
            s3_output_dir = f"filings/{self.ticker}"
            
            # Deep copy the data to avoid modifying the original
            json_data = json.loads(json.dumps(drug_data))
            
            # Create DataFrame - we need to stringify complex fields for CSV
            df_data = []
            for drug in json_data:
                drug_row = drug.copy()
                # Convert complex fields to JSON strings for CSV
                for key, value in drug_row.items():
                    if isinstance(value, (dict, list)):
                        drug_row[key] = json.dumps(value)
                df_data.append(drug_row)
                
            df = pd.DataFrame(df_data)
            
            # Save CSV to S3
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_key = f"{s3_output_dir}/drug_summary.csv"
            upload_to_s3(csv_buffer.getvalue(), csv_key)
            logger.info(f"Saved CSV to S3 for {self.ticker}")
            
            # Create Markdown content
            markdown_table = self.df_to_markdown_with_delimiters(df)
            markdown_content = f"# {self.ticker.upper()} Drug Development Pipeline\n\n{markdown_table}\n\n"
            
            # Add disclaimer
            markdown_content += "*Note: Data may be incomplete due to limitations in filings or processing.*"
            
            # Save Markdown to S3
            md_key = f"{s3_output_dir}/drug_summary.md"
            upload_to_s3(markdown_content, md_key)
            logger.info(f"Saved Markdown to S3 for {self.ticker}")
            
            # Return paths to output files
            return {
                "csv": f"s3://{s3_output_dir}/drug_summary.csv",
                "markdown": f"s3://{s3_output_dir}/drug_summary.md"
            }
        except Exception as e:
            logger.error(f"Error saving output to S3 for {self.ticker}: {str(e)}")
            raise
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run the full analysis pipeline for the ticker.
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Step 1: Identify assets
            assets = self.identify_assets()
            
            if not assets:
                logger.warning(f"No assets identified for {self.ticker}")
                return {
                    "ticker": self.ticker,
                    "job_id": self.job_id,
                    "assets": [],
                    "s3_paths": {}
                }
            
            # Step 2: Extract details for each asset
            asset_details = []
            for asset in assets:
                details = self.extract_asset_details(asset)
                if details:
                    asset_details.append(details)
            
            if not asset_details:
                logger.warning(f"No details extracted for {self.ticker}")
                return {
                    "ticker": self.ticker,
                    "job_id": self.job_id,
                    "assets": [],
                    "s3_paths": {}
                }
            
            # Step 3: Consolidate data
            consolidated_data = self.consolidate_data(asset_details)
            
            # Step 4: Save output to S3
            s3_paths = self.save_output(consolidated_data)
            
            # Return results
            return {
                "ticker": self.ticker,
                "job_id": self.job_id,
                "assets": consolidated_data,
                "s3_paths": s3_paths
            }
        except Exception as e:
            logger.error(f"Analysis failed for {self.ticker}: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            self._clean_up()
    
    @staticmethod
    def _clean_json_response(text: str) -> str:
        """
        Clean a JSON response to make it parseable.
        
        Args:
            text: Raw JSON response text
            
        Returns:
            Cleaned JSON string
        """
        # Remove code block markers if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Clean invalid control characters
        text = text.replace('\r', '').replace('\n', ' ').replace('\t', ' ')
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch == '\n')
        
        # Try to extract JSON from non-JSON text
        if not (text.strip().startswith("[") or text.strip().startswith("{")):
            import re
            json_match = re.search(r'\[.*?\]|\{.*?\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
            else:
                logger.error(f"No valid JSON in response: {text}")
                return "[]"
        
        return text.strip()