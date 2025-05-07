import os
import argparse
import logging
import json
import traceback
from uuid import uuid4
from collections import defaultdict
import pandas as pd
from tabulate import tabulate
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
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key == "your_api_key_" or not api_key:
        logger.error("Invalid or missing OPENAI_API_KEY. Set it using 'export OPENAI_API_KEY=your_key'.")
        raise ValueError("Invalid or missing OPENAI_API_KEY. Please set a valid API key.")
    return api_key

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

    def identify_assets(self):
        """Identify drugs, programs, and platforms using a targeted prompt."""
        try:
            if not self.vector_store:
                self.load_vector_store()

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
                openai_api_key=validate_api_key()
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
            raw_result = response["result"]

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

class DataExtractorAgent:
    """Agent to extract detailed information for each drug/program."""
    def __init__(self, ticker: str, vector_store):
        self.ticker = ticker
        self.vector_store = vector_store

    def extract_details(self, asset: str):
        """Extract detailed information for a specific drug/program."""
        try:
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

                    Example:
                    ```json
                    {{
                        "Name/Number": "WVE-N531",
                        "Mechanism of Action": "Splicing oligonucleotide promoting exon 53 skipping",
                        "Target(s)": "Dystrophin pre-mRNA (exon 53)",
                        "Indication": "Duchenne Muscular Dystrophy",
                        "Animal Models/Preclinical Data": "- Mouse model: Efficacy shown\n- Published 2022",
                        "Clinical Trials": "- Phase 1/2, N=10, ongoing, started 2022",
                        "Upcoming Milestones": "Phase 2 data Q4 2023",
                        "References": "0001621227-22-000123 (10-K, 2022-03-15)"
                    }}
                    ```
                """),
                ("human", "Context: {context}\nExtract details for asset: {asset}")
            ])

            # Create RAG chain with custom input mapping
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=validate_api_key()
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
            raw_result = raw_result.replace('\r', '').replace('\n', ' ').replace('\t', ' ')
            raw_result = ''.join(ch for ch in raw_result if ord(ch) >= 32 or ch == '\n')
            
            try:
                details = json.loads(raw_result.strip())
                # Ensure Name/Number is consistent
                if details.get("Name/Number") != asset and asset != "":
                    logger.warning(f"Name mismatch: Asset={asset}, Returned Name={details.get('Name/Number')}")
                    details["Name/Number"] = asset
                
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

    def consolidate(self, asset_details: list):
        """Merge data for each asset, resolving conflicts and combining references."""
        try:
            # Add debugging to track all incoming assets
            asset_names = [details.get("Name/Number", "Unknown") for details in asset_details if details]
            logger.info(f"Consolidating assets: {asset_names}")
            
            for details in asset_details:
                if not details:
                    continue
                
                name = details.get("Name/Number", "Unknown")
                logger.info(f"Processing asset: {name}")
                
                self.merged_data[name].update({
                    "Name/Number": name,
                    "Mechanism of Action": details.get("Mechanism of Action", "Not specified"),
                    "Target(s)": details.get("Target(s)", "Not specified"),
                    "Indication": details.get("Indication", "Not specified"),
                    "Animal Models/Preclinical Data": details.get("Animal Models/Preclinical Data", "Not specified"),
                    "Clinical Trials": details.get("Clinical Trials", "Not specified"),
                    "Upcoming Milestones": details.get("Upcoming Milestones", "Not specified")
                })
                
                # Handle references properly
                if isinstance(details.get("References", []), list):
                    self.merged_data[name]["References"].extend(details.get("References", []))
                    logger.info(f"Added list references for {name}")
                else:
                    # If it's a string, add it as a single item
                    ref = details.get("References", "")
                    if ref and ref != "Not specified":
                        self.merged_data[name]["References"].append(ref)
                        logger.info(f"Added string reference for {name}: {ref}")

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
                                logger.info(f"Converted dict reference for {name}")
                            except:
                                # If conversion fails, use a generic reference
                                str_ref = str(ref)
                                cleaned_refs.append(str_ref)
                                logger.info(f"Converted dict reference (fallback) for {name}")
                        elif ref and ref != "Not specified":
                            cleaned_refs.append(str(ref))
                    
                    # Now we can safely create a set
                    data["References"] = list(set(cleaned_refs))
                else:
                    # Make sure References is always a list
                    data["References"] = []

            # Log all assets after processing
            logger.info(f"Assets after processing: {list(self.merged_data.keys())}")

            # Create final data structure for output
            final_data = []
            for name, data in self.merged_data.items():
                logger.info(f"Adding {name} to final output")
                final_data.append({
                    "Name/Number": name,
                    "Mechanism of Action": data["Mechanism of Action"],
                    "Target(s)": data["Target(s)"],
                    "Indication": data["Indication"],
                    "Animal Models/Preclinical Data": data["Animal Models/Preclinical Data"].replace("Clinical Trials", "Animal Models") if isinstance(data["Animal Models/Preclinical Data"], str) else data["Animal Models/Preclinical Data"],
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

            with open(f"{self.output_dir}/drug_summary_old.md", "w") as f:
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