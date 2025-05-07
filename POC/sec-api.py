import pandas as pd
import os
import time
from bs4 import BeautifulSoup
from sec_api import QueryApi, RenderApi

# Initialize with your API key (requires registration)
api_key = "7e95421a9c39c390d941cb3ac698086705bf8474ed69f79fdf7c34c06958c2b7"  # Replace with your actual API key
query_api = QueryApi(api_key=api_key)
render_api = RenderApi(api_key=api_key)

def get_filings_by_ticker(ticker, form_types=['10-K', '8-K'], start_date='2010-01-01', end_date='2025-05-05', limit=100):
    """Get SEC filings for a given ticker and form types"""
    # Construct the query string
    form_type_query = ' OR '.join([f'formType:"{form_type}"' for form_type in form_types])
    query = f'ticker:{ticker} AND ({form_type_query}) AND filedAt:[{start_date} TO {end_date}]'
    
    # Set up the search parameters
    search_params = {
        "query": query,
        "from": "0",
        "size": str(limit),
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    
    # Execute the query
    filings = query_api.get_filings(search_params)
    
    return filings

def download_filing(url):
    """Download a filing using the Render API"""
    filing_content = render_api.get_filing(url)
    return filing_content

def save_filing_to_file(ticker, accession_number, form_type, content):
    """Save the filing content to a file on disk"""
    # Create directory for the ticker if it doesn't exist
    directory = os.path.join("sec-filings", ticker, form_type)
    os.makedirs(directory, exist_ok=True)
    
    # Create a filename based on the accession number
    filename = f"{accession_number.replace('-', '')}.html"
    filepath = os.path.join(directory, filename)
    
    # Write the content to the file
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)
    
    return filepath

def extract_tables_from_filing(filing_content):
    """Extract tables from a filing"""
    soup = BeautifulSoup(filing_content, 'html.parser')
    tables = soup.find_all('table')
    
    return [table for table in tables]

def extract_text_from_filing(filing_content):
    """Extract all text from a filing"""
    soup = BeautifulSoup(filing_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text
    text = soup.get_text()
    
    # Clean up the text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def save_text_to_file(ticker, accession_number, form_type, text_content):
    """Save the extracted text to a file"""
    directory = os.path.join("sec-filings-text", ticker, form_type)
    os.makedirs(directory, exist_ok=True)
    
    filename = f"{accession_number.replace('-', '')}.txt"
    filepath = os.path.join(directory, filename)
    
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text_content)
    
    return filepath

def main():
    # Example usage
    ticker = "WVE"  # Wave Life Sciences
    
    print(f"Retrieving filings for {ticker}...")
    filings = get_filings_by_ticker(ticker, form_types=['10-K', '8-K'], limit=10)
    
    print(f"Found {filings['total']['value']} filings for {ticker}")
    
    # Create a DataFrame with the filing metadata
    filing_data = []
    for filing in filings['filings']:
        filing_data.append({
            'formType': filing.get('formType'),
            'filedAt': filing.get('filedAt'),
            'reportDate': filing.get('periodOfReport'),
            'accessionNumber': filing.get('accessionNo'),
            'fileUrl': filing.get('linkToFilingDetails'),
            'companyName': filing.get('companyName'),
            'cik': filing.get('cik')
        })
    
    filings_df = pd.DataFrame(filing_data)
    print("\nFirst few filings:")
    print(filings_df.head())
    
    # Save DataFrame to CSV for reference
    filings_df.to_csv(f"{ticker}_filings.csv", index=False)
    print(f"Saved filing metadata to {ticker}_filings.csv")
    
    # Download and save all filings
    print("\nDownloading and saving filings...")
    for index, row in filings_df.iterrows():
        try:
            print(f"Processing {row['formType']} from {row['filedAt']} (Accession: {row['accessionNumber']})...")
            
            # Download filing
            filing_content = download_filing(row['fileUrl'])
            
            # Save raw HTML
            html_filepath = save_filing_to_file(
                ticker, 
                row['accessionNumber'], 
                row['formType'], 
                filing_content
            )
            print(f"  - HTML saved to: {html_filepath}")
            
            # Extract and save text
            text_content = extract_text_from_filing(filing_content)
            text_filepath = save_text_to_file(
                ticker, 
                row['accessionNumber'], 
                row['formType'], 
                text_content
            )
            print(f"  - Text saved to: {text_filepath}")
            
            # Extract tables for further analysis if needed
            tables = extract_tables_from_filing(filing_content)
            print(f"  - Extracted {len(tables)} tables from the filing")
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing filing {row['accessionNumber']}: {str(e)}")
    
    print("\nAll filings downloaded and processed successfully.")
    print("Files are stored in the sec-filings and sec-filings-text directories.")

if __name__ == "__main__":
    main()