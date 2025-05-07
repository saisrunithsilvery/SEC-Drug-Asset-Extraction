import os
import argparse
import logging
import json
import traceback
from uuid import uuid4
from collections import defaultdict
import pandas as pd
from tabulate import tabulate
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Union, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sec_drug_pipeline.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Validate OpenAI API key
def validate_api_key():
    """Check if a valid OpenAI API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-") is False:
        logger.error("Invalid or missing OPENAI_API_KEY. Set it using 'export OPENAI_API_KEY=your_key'.")
        raise ValueError("Invalid or missing OPENAI_API_KEY. Please set a valid API key.")
    return api_key

# Helper function to ensure string format for fields
def ensure_string(value):
    """Convert various data types to string format."""
    if isinstance(value, list):
        # Join list elements with newlines
        return "\n".join(str(item) for item in value)
    elif isinstance(value, dict):
        # For dictionaries, create a formatted string representation
        try:
            # First try to convert to JSON string
            return json.dumps(value, indent=2)
        except:
            # If that fails, use a simple string representation
            return str(value)
    elif value is None:
        return "Not specified"
    else:
        return str(value)

# Pydantic models for data validation
class AssetList(BaseModel):
    """Model for validating list of assets from the DrugIdentifierAgent"""
    assets: List[str] = Field(default_factory=list)

    @field_validator('assets')
    @classmethod
    def unique_assets(cls, v):
        """Ensure all assets are unique"""
        return list(set(v))

class DrugDetails(BaseModel):
    """Model for drug/program details"""
    name_number: str = Field(..., validation_alias="Name/Number")
    mechanism_of_action: str = Field("Not specified", validation_alias="Mechanism of Action")
    targets: str = Field("Not specified", validation_alias="Target(s)")
    indication: str = Field("Not specified", validation_alias="Indication")
    animal_models_preclinical_data: str = Field("Not specified", validation_alias="Animal Models/Preclinical Data")
    clinical_trials: str = Field("Not specified", validation_alias="Clinical Trials")
    upcoming_milestones: str = Field("Not specified", validation_alias="Upcoming Milestones") 
    references: Union[str, List[str]] = Field("Not specified", validation_alias="References")
    
    model_config = {
        "validate_by_name": True,
        "populate_by_name": True,
    }
    
    @field_validator('name_number')
    @classmethod
    def name_not_empty(cls, v):
        """Validate that name is not empty"""
        if not v or v.strip() == "":
            raise ValueError("Name/Number cannot be empty")
        return v
    
    @field_validator('animal_models_preclinical_data', 'clinical_trials', 'upcoming_milestones')
    @classmethod
    def ensure_string_format(cls, v):
        """Ensure these fields are in string format"""
        return ensure_string(v)
    
    @model_validator(mode='before')
    @classmethod
    def set_defaults(cls, data):
        """Set default values for missing fields"""
        if not isinstance(data, dict):
            return data
            
        fields = [
            "Name/Number", "Mechanism of Action", "Target(s)", "Indication", 
            "Animal Models/Preclinical Data", "Clinical Trials", 
            "Upcoming Milestones", "References"
        ]
        
        for field in fields:
            if field not in data or not data[field]:
                data[field] = "Not specified"
            elif field in ["Animal Models/Preclinical Data", "Clinical Trials", "Upcoming Milestones"]:
                # Ensure these fields are in string format
                data[field] = ensure_string(data[field])
                
        return data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with aliased field names"""
        return {
            "Name/Number": self.name_number,
            "Mechanism of Action": self.mechanism_of_action,
            "Target(s)": self.targets,
            "Indication": self.indication,
            "Animal Models/Preclinical Data": self.animal_models_preclinical_data,
            "Clinical Trials": self.clinical_trials,
            "Upcoming Milestones": self.upcoming_milestones,
            "References": self.references
        }

class DrugIdentifierAgent:
    """Agent to identify all drugs, programs, and platforms from SEC filings."""
    def __init__(self, ticker: str, embeddings):
        self.ticker = ticker
        self.embeddings = embeddings
        self.vector_store = None

    def load_vector_store(self):
        """Load FAISS index for the ticker."""
        try:
            self.vector_store = FAISS.load_local(
                f"filings/{self.ticker}/faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded FAISS index for {self.ticker}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index for {self.ticker}: {str(e)}")
            raise

    def identify_assets(self) -> List[str]:
        """Identify drugs, programs, and platforms using a targeted prompt."""
        try:
            if not self.vector_store:
                self.load_vector_store()

            # Enhanced prompt for more accurate extraction
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """
                    You are an expert in biotech SEC filings with deep domain knowledge in pharmaceutical and biotechnology
                    regulatory documents. Your specialized task is to extract and identify all drugs, programs, and platform 
                    technologies mentioned in the provided SEC filing context.
                    
                    ## IMPORTANT EXTRACTION RULES:
                    - Focus ONLY on extracting unique identifiers for:
                      * Drug candidates (e.g., WVE-N531, ABC-123, eteplirsen)
                      * Development programs (e.g., AATD program, INHBE program)
                      * Platform technologies (e.g., PRISM technology, ADAR editing platform)
                    - For drugs, extract both code names (e.g., WVE-N531) and generic/trade names if available
                    - For platforms, include the core platform name without excess descriptive text
                    - Maintain precise spelling and formatting (e.g., case sensitivity, hyphens)
                    - De-duplicate the list to ensure each asset appears exactly once
                    - Exclude general therapeutic areas, targets, or disease names that aren't specific programs
                    - DO NOT categorize or group assets - provide only a flat list
                    
                    ## OUTPUT FORMAT REQUIREMENTS:
                    - Return VALID JSON as a flat array of strings
                    - Format: ["Asset1", "Asset2", "Asset3", ...]
                    - If no assets are found, return an empty array: []
                    - Do not include any explanatory text, commentary, or non-JSON elements
                    - Ensure the output is properly escaped and parseable as JSON
                    
                    ## EXAMPLE OUTPUT:
                    ```json
                    ["WVE-N531", "WVE-003", "PN-chemistry", "PRISM", "ADAR editing", "C9orf72", "AATD program"]
                    ```
                """),
                ("human", "Context: {context}\nReturn a flat JSON list of all drug, program, and platform names.")
            ])

            # Create RAG chain
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=validate_api_key(),
                temperature=0.1  # Lower temperature for more consistent outputs
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 10}),
                chain_type_kwargs={"prompt": prompt_template}
            )

            # Run query
            query = f"Identify all drug, program, and platform names for {self.ticker}"
            response = qa_chain.invoke({"query": query})
            raw_result = response.get("result", "")

            # Clean response
            def clean_response(text: str) -> str:
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if not (text.startswith("[") or text.startswith("{")):
                    import re
                    json_match = re.search(r'\[.*?\]', text, re.DOTALL)
                    if json_match:
                        text = json_match.group(0)
                    else:
                        logger.error(f"No valid JSON in response: {text}")
                        return "[]"
                return text

            raw_result = clean_response(raw_result)
            logger.info(f"Cleaned response for {self.ticker}: {raw_result}")

            # Parse JSON with Pydantic
            try:
                # First try to parse as a simple list
                assets = json.loads(raw_result)
                
                # Use Pydantic to validate
                if isinstance(assets, dict):
                    # If we got a dict instead of a list, flatten it
                    flattened_assets = []
                    for category, items in assets.items():
                        if isinstance(items, list):
                            flattened_assets.extend(items)
                    
                    asset_list = AssetList(assets=flattened_assets)
                    assets = asset_list.assets
                else:
                    # Handle as normal list
                    asset_list = AssetList(assets=assets)
                    assets = asset_list.assets
                    
                logger.info(f"Identified {len(assets)} unique assets for {self.ticker}: {assets}")
                return assets
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for {self.ticker}: {str(e)}")
                logger.error(f"Problematic response: {raw_result}")
                return []
            except Exception as e:
                logger.error(f"Validation error: {str(e)}")
                # Try to salvage any assets if validation fails
                try:
                    assets = json.loads(raw_result)
                    if isinstance(assets, list):
                        # Return the list even if validation failed
                        return list(set(assets))
                    elif isinstance(assets, dict):
                        # Try to flatten
                        flattened = []
                        for _, items in assets.items():
                            if isinstance(items, list):
                                flattened.extend(items)
                        return list(set(flattened))
                except:
                    return []
                
        except Exception as e:
            logger.error(f"Error identifying assets for {self.ticker}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

class DataExtractorAgent:
    """Agent to extract detailed information for each drug/program."""
    def __init__(self, ticker: str, vector_store):
        self.ticker = ticker
        self.vector_store = vector_store

    def extract_details(self, asset: str) -> Optional[DrugDetails]:
        """Extract detailed information for a specific drug/program."""
        try:
            # Enhanced prompt with properly escaped JSON keys and improved structure
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """
                    You are an expert in biotech SEC filings analysis with deep domain knowledge in pharmaceuticals 
                    and biotechnology. Your task is to extract comprehensive, structured information about the specified 
                    drug, program, or platform from the provided context of SEC filings.
                    
                    ## IMPORTANT GUIDELINES:
                    - SEC filings often use technical language and may not explicitly use terms like 'Mechanism of Action' 
                      or 'Indication' - you must infer these from technical descriptions and context
                    - Be precise and specific in your extraction - avoid vague statements
                    - If information truly isn't available, use 'Not specified' rather than making assumptions
                    - Look for the most recent information available in the context
                    - Extract detailed timelines and specific results when available
                    - IMPORTANT: All field values must be STRING format, not nested objects or arrays
                    
                    ## REQUIRED INFORMATION TO EXTRACT:
                    1. Name/Number: Use the provided asset name exactly (REQUIRED)
                    
                    2. Mechanism of Action: How the drug/program works at a molecular or cellular level
                       - Look for phrases like: "mechanism", "therapeutic approach", "modality", "technology platform"
                       - Examples: "small molecule inhibitor of JAK2", "antisense oligonucleotide promoting exon skipping"
                    
                    3. Target(s): Specific biological molecule(s) or pathway(s) the drug acts upon
                       - Look for gene names, protein targets, receptors, enzymes
                       - Examples: "EGFR mutations", "dystrophin pre-mRNA", "TNF-alpha"
                    
                    4. Indication: Disease(s) or condition(s) the drug/program aims to treat
                       - Look for specific disease names or therapeutic areas
                       - Examples: "metastatic non-small cell lung cancer", "Duchenne Muscular Dystrophy"
                    
                    5. Animal Models/Preclinical Data: Experimental results from laboratory and animal studies
                       - Include: model type, key results, efficacy markers, comparative data
                       - Format as LINE-SEPARATED TEXT with bullet points (not a JSON array)
                       - Example: "- Mouse model: Increased dystrophin expression\\n- NHP study: Well-tolerated"
                    
                    6. Clinical Trials: Human study information
                       - Include: phase, patient numbers (N=X), duration, key endpoints, efficacy results, safety data
                       - Format as LINE-SEPARATED TEXT with bullet points (not a JSON array)
                       - Example: "- Phase 1/2 (DYSTANCE-51), N=10, ongoing\\n- No serious adverse events reported"
                    
                    7. Upcoming Milestones: Future events or data releases mentioned in the most recent filings
                       - Include: expected timing, study phase transitions, data readouts, regulatory filings
                       - Format as LINE-SEPARATED TEXT with bullet points (not a JSON array)
                       - Example: "- Phase 2 data expected Q4 2023\\n- Potential IND submission for EU Q1 2024"
                    
                    8. References: Source information for the extracted data
                       - Include: SEC filing accession number, form type (e.g., 10-K), filing date
                       - Format as SIMPLE STRING, not an array
                       - Example: "0001621227-22-000123 (10-K, 2022-03-15), 0001621227-23-000052 (10-Q, 2023-05-10)"
                    
                    ## OUTPUT FORMAT:
                    Return your analysis as a properly structured JSON dictionary exactly matching this format:
                    
                    ```json
                    {{
                        "Name/Number": "WVE-N531",
                        "Mechanism of Action": "Splicing oligonucleotide promoting exon 53 skipping",
                        "Target(s)": "Dystrophin pre-mRNA (exon 53)",
                        "Indication": "Duchenne Muscular Dystrophy",
                        "Animal Models/Preclinical Data": "- Mouse model: Increased dystrophin expression by 53% (p<0.001)\\n- NHP study: Well-tolerated at therapeutic doses\\n- Published in Nature Medicine, 2022",
                        "Clinical Trials": "- Phase 1/2 (DYSTANCE-51), N=10, ongoing, started 2022\\n- Interim data showed 74% exon skipping (p<0.01)\\n- No serious adverse events reported",
                        "Upcoming Milestones": "- Phase 2 data expected Q4 2023\\n- Potential IND submission for EU Q1 2024",
                        "References": "0001621227-22-000123 (10-K, 2022-03-15), 0001621227-23-000052 (10-Q, 2023-05-10)"
                    }}
                    ```
                    
                    IMPORTANT: All values MUST be strings (text), even for multi-line data. Do not use actual JSON arrays or objects within the response - use formatted strings with line breaks instead.
                """),
                ("human", "Context: {context}\nExtract details for asset: {asset}")
            ])

            # Create RAG chain with custom input mapping
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=validate_api_key(),
                temperature=0.1,  # Lower temperature for more consistent outputs
            )
            
            # Reduce k to avoid context length issues
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
            chain = (
                {"context": retriever, "asset": RunnablePassthrough()}
                | prompt_template
                | llm
                | {"result": lambda x: x.content}
            )

            # Run query
            raw_result = chain.invoke(asset).get("result", "")
            raw_result = raw_result.strip()
            
            # Log raw result for debugging
            logger.info(f"Raw result for {asset}: {raw_result[:200]}...")

            # Clean and parse JSON
            if raw_result.startswith("```json"):
                raw_result = raw_result[7:]
            if raw_result.endswith("```"):
                raw_result = raw_result[:-3]
                
            # Clean invalid control characters
            raw_result = raw_result.replace('\r', '').replace('\t', ' ')
            raw_result = ''.join(ch for ch in raw_result if ord(ch) >= 32 or ch == '\n')
            
            try:
                # Parse JSON 
                json_data = json.loads(raw_result.strip())
                
                # Ensure Name/Number is consistent with asset parameter
                if json_data.get("Name/Number") != asset and asset != "":
                    logger.warning(f"Name mismatch: Asset={asset}, Returned Name={json_data.get('Name/Number')}")
                    json_data["Name/Number"] = asset
                
                # Pre-process fields that might be arrays or objects to ensure they're strings
                for field in ["Animal Models/Preclinical Data", "Clinical Trials", "Upcoming Milestones"]:
                    if field in json_data and not isinstance(json_data[field], str):
                        json_data[field] = ensure_string(json_data[field])
                
                # Validate through Pydantic model
                details = DrugDetails.model_validate(json_data)
                logger.info(f"Successfully extracted details for {asset} in {self.ticker}")
                return details
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {asset}: {str(e)}")
                logger.error(f"Problematic JSON: {raw_result}")
                
                # Create a minimal valid record using Pydantic
                return DrugDetails(
                    name_number=asset,
                    mechanism_of_action="Not specified",
                    targets="Not specified", 
                    indication="Not specified",
                    animal_models_preclinical_data="Not specified",
                    clinical_trials="Not specified",
                    upcoming_milestones="Not specified",
                    references="JSON parsing error"
                )
                
            except Exception as e:
                logger.error(f"Pydantic validation error for {asset}: {str(e)}")
                
                # Try to salvage by creating a minimal valid record
                try:
                    # Force create with just the name
                    return DrugDetails(
                        name_number=asset,
                        mechanism_of_action="Not specified",
                        targets="Not specified", 
                        indication="Not specified",
                        animal_models_preclinical_data="Not specified",
                        clinical_trials="Not specified",
                        upcoming_milestones="Not specified",
                        references="Validation error"
                    )
                except Exception as inner_e:
                    logger.error(f"Failed to create fallback record for {asset}: {str(inner_e)}")
                    return None
                
        except Exception as e:
            logger.error(f"Error extracting details for {asset} in {self.ticker}: {str(e)}")
            return None
            
    def to_dict(self, details: DrugDetails) -> dict:
        """Convert DrugDetails model to a dictionary"""
        if details is None:
            return None
        return details.to_dict()

class DataConsolidatorAgent:
    """Agent to merge and reconcile data across filings."""
    def __init__(self):
        self.merged_data = defaultdict(lambda: {
            "Name/Number": None,
            "Mechanism of Action": "Not specified",
            "Target(s)": "Not specified",
            "Indication": "Not specified",
            "Animal Models/Preclinical Data": "Not specified",
            "Clinical Trials": "Not specified",
            "Upcoming Milestones": "Not specified",
            "References": []
        })

    def consolidate(self, asset_details: List[Union[DrugDetails, dict]]) -> List[dict]:
        """Merge data for each asset, resolving conflicts and combining references."""
        try:
            # Add debugging to track all incoming assets
            asset_names = []
            for details in asset_details:
                if not details:
                    continue
                if isinstance(details, DrugDetails):
                    asset_names.append(details.name_number)
                else:
                    asset_names.append(details.get("Name/Number", "Unknown"))
                    
            logger.info(f"Consolidating assets: {asset_names}")
            
            for details in asset_details:
                if not details:
                    continue
                
                # Handle both Pydantic models and dictionaries
                if isinstance(details, DrugDetails):
                    details_dict = details.to_dict()
                else:
                    details_dict = details
                
                name = details_dict.get("Name/Number", "Unknown")
                logger.info(f"Processing asset: {name}")
                
                # Update fields
                self.merged_data[name].update({
                    "Name/Number": name,
                    "Mechanism of Action": details_dict.get("Mechanism of Action", "Not specified"),
                    "Target(s)": details_dict.get("Target(s)", "Not specified"),
                    "Indication": details_dict.get("Indication", "Not specified"),
                    "Animal Models/Preclinical Data": details_dict.get("Animal Models/Preclinical Data", "Not specified"),
                    "Clinical Trials": details_dict.get("Clinical Trials", "Not specified"),
                    "Upcoming Milestones": details_dict.get("Upcoming Milestones", "Not specified")
                })
                
                # Handle references properly
                refs = details_dict.get("References", [])
                if isinstance(refs, list):
                    self.merged_data[name]["References"].extend(refs)
                elif refs and refs != "Not specified":
                    self.merged_data[name]["References"].append(refs)

            # Clean up references (remove duplicates)
            for name, data in self.merged_data.items():
                logger.info(f"Cleaning references for {name}")
                if isinstance(data["References"], list):
                    # Convert any dict references to strings
                    cleaned_refs = []
                    for ref in data["References"]:
                        if isinstance(ref, dict):
                            # Convert dict to string representation
                            try:
                                ref_str = json.dumps(ref)
                                cleaned_refs.append(ref_str)
                            except:
                                # If conversion fails, use a generic reference
                                str_ref = str(ref)
                                cleaned_refs.append(str_ref)
                        elif ref and ref != "Not specified":
                            cleaned_refs.append(str(ref))
                    
                    # Now we can safely create a set
                    data["References"] = list(set(cleaned_refs))
                else:
                    # Make sure References is always a list
                    data["References"] = []

            # Create final data structure for output
            final_data = []
            for name, data in self.merged_data.items():
                logger.info(f"Adding {name} to final output")
                # Ensure Animal Models field doesn't contain "Clinical Trials" text
                animal_models = data["Animal Models/Preclinical Data"]
                if isinstance(animal_models, str) and "Clinical Trials" in animal_models:
                    animal_models = animal_models.replace("Clinical Trials", "Animal Models")
                
                final_data.append({
                    "Name/Number": name,
                    "Mechanism of Action": data["Mechanism of Action"],
                    "Target(s)": data["Target(s)"],
                    "Indication": data["Indication"],
                    "Animal Models/Preclinical Data": animal_models,
                    "Clinical Trials": data["Clinical Trials"],
                    "Upcoming Milestones": data["Upcoming Milestones"],
                    "References": ", ".join(data["References"]) if data["References"] else "Not specified"
                })

            logger.info(f"Consolidated data for {len(final_data)} unique assets")
            logger.info(f"Final assets: {[item.get('Name/Number') for item in final_data]}")
            return final_data
        except Exception as e:
            logger.error(f"Error consolidating data: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

class OutputFormatterAgent:
    """Agent to save output as CSV and Markdown."""
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.output_dir = f"filings/{ticker}"
        os.makedirs(self.output_dir, exist_ok=True)

    def save_output(self, drug_data: list):
        """Save drug data as CSV and Markdown with additional summaries."""
        try:
            # Log the data before saving
            logger.info(f"Saving {len(drug_data)} assets to CSV and Markdown")
            
            # Create DataFrame
            df = pd.DataFrame(drug_data)

            # Save CSV
            df.to_csv(f"{self.output_dir}/drug_summary.csv", index=False)
            logger.info(f"Saved CSV for {self.ticker}")

            # Save Markdown
            markdown_table = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
            markdown_content = f"# {self.ticker.upper()} Drug Asset Summary\n\n{markdown_table}\n\n"

            # Add additional sections
            platforms = [d["Name/Number"] for d in drug_data if "platform" in str(d.get("Mechanism of Action", "")).lower() or "technology" in str(d.get("Mechanism of Action", "")).lower()]
            if platforms:
                markdown_content += "**Platform Technologies:**\n"
                markdown_content += "\n".join([f"- {plat}" for plat in platforms]) + "\n\n"

            tech_areas = set()
            for d in drug_data:
                notes = str(d.get("Mechanism of Action", "")).lower() + str(d.get("Target(s)", "")).lower()
                if "rna" in notes or "oligonucleotide" in notes:
                    tech_areas.add("RNA editing/targeting technologies")
                if "snp" in notes:
                    tech_areas.add("SNP-targeting for specific genetic diseases")
            if tech_areas:
                markdown_content += "**Key Technology Areas:**\n"
                markdown_content += "\n".join([f"- {area}" for area in tech_areas]) + "\n\n"

            recent_refs = df["References"].dropna().unique()
            if recent_refs.size > 0:
                markdown_content += "**Recent Updates:**\n"
                markdown_content += f"- Most recent filings referenced: {', '.join(recent_refs[:2])}\n\n"

            markdown_content += "*Note: Data may be incomplete due to limitations in filings or processing.*"

            with open(f"{self.output_dir}/drug_summary.md", "w") as f:
                f.write(markdown_content)
            logger.info(f"Saved Markdown for {self.ticker}")

            return df
        except Exception as e:
            logger.error(f"Error saving output for {self.ticker}: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Extract drug data from SEC filings using a multi-agent system.")
    parser.add_argument("--ticker", default="WVE", help="Stock ticker (e.g., WVE)")
    args = parser.parse_args()

    ticker = args.ticker
    try:
        logger.info(f"Starting multi-agent pipeline for ticker {ticker}")

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Agent 1: Identify drugs, programs, and platforms
        identifier = DrugIdentifierAgent(ticker, embeddings)
        assets = identifier.identify_assets()

        if not assets:
            logger.warning(f"No assets identified for {ticker}")
            return

        # Agent 2: Extract details for each asset
        extractor = DataExtractorAgent(ticker, identifier.vector_store)
        asset_details = []
        for asset in assets:
            details = extractor.extract_details(asset)
            if details:
                asset_details.append(details)

        if not asset_details:
            logger.warning(f"No details extracted for {ticker}")
            return

        # Agent 3: Consolidate data
        consolidator = DataConsolidatorAgent()
        consolidated_data = consolidator.consolidate(asset_details)

        # Agent 4: Format and save output
        formatter = OutputFormatterAgent(ticker)
        df = formatter.save_output(consolidated_data)

        logger.info(f"Pipeline completed for {ticker}. Output table shape: {df.shape}")
    except Exception as e:
        logger.error(f"Pipeline failed for {ticker}: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()