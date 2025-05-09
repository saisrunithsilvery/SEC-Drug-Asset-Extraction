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
                    You are an expert in biotech SEC filings. From the provided context, identify all drugs, programs, and platform technologies 
                    mentioned. Return a flat JSON list of unique names/identifiers (e.g., ["PE-1", "SM", "AA"]). 
                    - Include drug names/number.
                    - Include program names the company has developed.
                    - Do NOT include diseases names or indications.
                    - Do NOT group by category.
                    - Ensure the output is valid JSON.
                    - If no assets are found, return an empty list [].
                """),
                ("human", "Context: {context}\nReturn a flat JSON list of all drug, program, and platform names.")
            ])
            
            # Create RAG chain
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=OPENAI_API_KEY
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
    
    def _normalize_complex_value(self, value: Any) -> str:
        """
        Convert complex data types (lists, dicts) to formatted strings.
        
        Args:
            value: Value to normalize
            
        Returns:
            Normalized string value
        """
        if value is None:
            return "Not specified"
        
        if isinstance(value, str):
            return value
        
        # Convert complex objects to formatted strings
        try:
            if isinstance(value, (dict, list)):
                # For pretty formatting, use indent
                return json.dumps(value, indent=2)
            else:
                return str(value)
        except Exception as e:
            logger.warning(f"Error formatting complex value: {str(e)}")
            # Return a simple string representation as fallback
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
            
            # Enhanced prompt with properly escaped JSON keys
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """
                    You are an expert in biotech SEC filings. Extract detailed information for the specified drug, program, or platform 
                    from the provided context. SEC filings may not use terms like 'Mechanism of Action' or 'Indication' directly, so infer 
                    these from descriptions, technical details, or context. Provide:
                    - Name/Number: Use the provided asset name (compulsory).
                    - Mechanism of Action: How the drug/program works (e.g., 'promotes exon skipping' or 'RNA editing'). Look for terms like 
                      'therapeutic approach,' 'targets,' or functional descriptions. Default to 'Not specified' if unclear.
                    - Target(s): Biological molecule/pathway (e.g., 'dystrophin pre-mRNA'). Look for mentions of genes, proteins, or pathways. 
                      Default to 'Not specified' if unclear.
                    - Indication: Target disease (e.g., 'Duchenne Muscular Dystrophy'). Look for disease names or treatment goals. Default to 
                      'Not specified' if unclear.
                    - Animal Models/Preclinical Data: Bullet points summarizing experiments (model type, results, year/reference). Look for 
                      animal studies or lab results. Default to 'Not specified' if missing.
                    - Clinical Trials: Bullet points with phase, N, duration, results, adverse events (AEs), and dates. Look for trial updates or 
                      clinical study mentions. Default to 'Not specified' if missing.
                    - Upcoming Milestones: Expected or Future catalysts/events that is planned or mentioned. Look for future plans or timelines from the last updated file. 
                      Default to 'Not specified' if missing.
                    - References: Filing accession number, form type (e.g., 10-K), and filed date from metadata. Default to 'Not specified' if missing.
                    Merge data from multiple chunks if applicable. Return a JSON dictionary.
                    
                    IMPORTANT: The response must be formatted as a valid JSON object. Return only a single plain JSON object.
                """),
                ("human", "Context: {context}\nExtract details for asset: {asset}")
            ])
            
            # Create RAG chain with custom input mapping
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=OPENAI_API_KEY
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
            raw_result = self._clean_json_response(raw_result)
            
            try:
                details = json.loads(raw_result.strip())
                # Ensure Name/Number is consistent
                if details.get("Name/Number") != asset and asset != "":
                    logger.warning(f"Name mismatch: Asset={asset}, Returned Name={details.get('Name/Number')}")
                    details["Name/Number"] = asset
                
                # Normalize complex values to strings
                for key in details.keys():
                    if not isinstance(details[key], str):
                        details[key] = self._normalize_complex_value(details[key])
                
                logger.info(f"Extracted details for {asset} in {self.ticker}")
                return details
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {asset}: {str(e)}")
                logger.error(f"Problematic JSON: {raw_result}")
                
                # Try to salvage by creating a minimal valid record
                return {
                    "Name/Number": asset,
                    "Mechanism of Action": "Not specified",
                    "Target(s)": "Not specified", 
                    "Indication": "Not specified",
                    "Animal Models/Preclinical Data": "Not specified",
                    "Clinical Trials": "Not specified",
                    "Upcoming Milestones": "Not specified",
                    "References": "JSON parsing error"
                }
        except Exception as e:
            logger.error(f"Error extracting details for {asset} in {self.ticker}: {str(e)}")
            return None
    
    def consolidate_data(self, asset_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge data for each asset, resolving conflicts and combining references.
        
        Args:
            asset_details: List of asset detail dictionaries
            
        Returns:
            Consolidated list of asset details
        """
        try:
            # Initialize merged data structure
            merged_data = defaultdict(lambda: {
                "Name/Number": None,
                "Mechanism of Action": "Not specified",
                "Target(s)": "Not specified",
                "Indication": "Not specified",
                "Animal Models/Preclinical Data": "Not specified",
                "Clinical Trials": "Not specified",
                "Upcoming Milestones": "Not specified",
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
                
                # Process and normalize complex values
                processed_details = details.copy()
                for key in processed_details.keys():
                    if not isinstance(processed_details[key], str):
                        processed_details[key] = self._normalize_complex_value(processed_details[key])
                
                # Update fields with non-empty values
                merged_data[name].update({
                    "Name/Number": name,
                    "Mechanism of Action": processed_details.get("Mechanism of Action", "Not specified"),
                    "Target(s)": processed_details.get("Target(s)", "Not specified"),
                    "Indication": processed_details.get("Indication", "Not specified"),
                    "Animal Models/Preclinical Data": processed_details.get("Animal Models/Preclinical Data", "Not specified"),
                    "Clinical Trials": processed_details.get("Clinical Trials", "Not specified"),
                    "Upcoming Milestones": processed_details.get("Upcoming Milestones", "Not specified")
                })
                
                # Handle references properly
                ref = details.get("References", "")
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
                else:
                    # Make sure References is always a list
                    data["References"] = []
            
            # Create final data structure for output
            final_data = []
            for name, data in merged_data.items():
                # Ensure all values are strings for Pydantic validation
                for key in data.keys():
                    if key != "References" and not isinstance(data[key], str):
                        data[key] = self._normalize_complex_value(data[key])
                
                final_data.append({
                    "Name/Number": name,
                    "Mechanism of Action": data["Mechanism of Action"],
                    "Target(s)": data["Target(s)"],
                    "Indication": data["Indication"],
                    "Animal Models/Preclinical Data": data["Animal Models/Preclinical Data"],
                    "Clinical Trials": data["Clinical Trials"],
                    "Upcoming Milestones": data["Upcoming Milestones"],
                    "References": ", ".join(data["References"]) if data["References"] else "Not specified"
                })
            
            logger.info(f"Consolidated data for {len(final_data)} unique assets")
            return final_data
        except Exception as e:
            logger.error(f"Error consolidating data: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    def df_to_markdown_with_delimiters(self,df):
        
        # Get column names and data
        columns = df.columns.tolist()
        data = df.values.tolist()
        
        # Create header row
        markdown = "| " + " | ".join([str(col) for col in columns]) + " |\n"
        
        # Create separator row
        markdown += "| " + " | ".join(["---" for _ in columns]) + " |\n"
        
        # Create data rows
        for row in data:
            # Convert all values to strings
            row_str = [str(cell).replace('\n', ' ').replace('\r', '') for cell in row]
            markdown += "| " + " | ".join(row_str) + " |\n"
    
        with open('testing_ds.md', 'w') as f:
            f.write(markdown)
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
            
            # Create DataFrame
            df = pd.DataFrame(drug_data)
            
            # Save CSV to S3
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_key = f"{s3_output_dir}/drug_summary.csv"
            upload_to_s3(csv_buffer.getvalue(), csv_key)
            logger.info(f"Saved CSV to S3 for {self.ticker}")
            
            # Create Markdown content
            # markdown_table = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
            markdown_table = self.df_to_markdown_with_delimiters(df)
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