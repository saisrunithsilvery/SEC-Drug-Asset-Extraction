import streamlit as st
import pandas as pd
import requests
import json
import base64
from io import StringIO
import os
from dotenv import load_dotenv
import time
import uuid

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="K-Cap Funding SEC Drug Asset Visualization",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache clearing mechanism
# 1. Generate a unique session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
    # Clear all cached data on new session
    if 'data' in st.session_state:
        del st.session_state.data
    if 'df' in st.session_state:
        del st.session_state.df
    if 'filtered_df' in st.session_state:
        del st.session_state.filtered_df
    if 'ticker' in st.session_state:
        del st.session_state.ticker
    if 'has_run' in st.session_state:
        del st.session_state.has_run
    
    # Also clear Streamlit's internal cache
    st.cache_data.clear()

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: white;
    }
    .css-1y4p8pa {
        margin-top: -4rem;
    }
    .stDataFrame {
        background-color: #1e1e1e;
    }
    th {
        background-color: #2c2c2c;
        color: white !important;
        font-weight: bold !important;
    }
    td {
        color: white !important;
    }
    .css-145kmo2 {
        font-size: 2.5rem !important;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1kyxreq {
        justify-content: center;
    }
    .st-bc {
        background-color: #2c2c2c;
    }
    .css-1v3fvcr {
        background-color: #121212;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = f"{os.environ.get('BACKEND_URL', 'http://localhost:8000')}/api/v1/filings/pipeline"

# Initialize session state for storing data between reruns
if 'data' not in st.session_state:
    st.session_state.data = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = ""
if 'has_run' not in st.session_state:
    st.session_state.has_run = False

# Title
st.title("Drug Asset Analysis Dashboard")
st.markdown("Analyze SEC filings to extract drug, program, and platform information")

# Add a manual cache clear button in the sidebar
def clear_cache():
    """Clear all cached data"""
    # Clear session state
    for key in list(st.session_state.keys()):
        if key != 'session_id':  # Keep session ID
            del st.session_state[key]
    
    # Reinitialize essential state
    st.session_state.data = None
    st.session_state.df = None
    st.session_state.filtered_df = None
    st.session_state.ticker = ""
    st.session_state.has_run = False
    
    # Clear Streamlit's cache
    st.cache_data.clear()
    
    # Force a rerun to refresh the UI
    st.rerun()

# Function to create a download link
def get_download_link(data, filename, link_text):
    """Generate a download link for the data"""
    if filename.endswith('.csv'):
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    else:  # Markdown file
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to normalize JSON string fields
def normalize_json_fields(data, fields_to_normalize=['Animal Models/Preclinical Data', 'Clinical Trials', 'Upcoming Milestones', 'References']):
    """Normalize JSON string fields for better display"""
    for asset in data.get('assets', []):
        for field in fields_to_normalize:
            if field in asset and isinstance(asset[field], str) and (asset[field].startswith('{') or asset[field].startswith('[')):
                try:
                    # Try to parse the JSON
                    parsed = json.loads(asset[field])
                    # Convert back to a more readable format
                    asset[field] = json.dumps(parsed, indent=2)
                except:
                    # If parsing fails, keep as is
                    pass
    return data

def df_to_markdown_with_delimiters(df):
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

    return markdown

# Function to fetch data from API - use session_id in cache key to refresh on reload
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def fetch_drug_data(ticker, session_id):
    """Fetch drug data from the API"""
    try:
        response = requests.post(
            API_URL,
            json={"ticker": ticker},
            headers={"Content-Type": "application/json"}
        )
        print(response.json())
        
        if response.status_code == 200:
            data = response.json()
            # Normalize JSON string fields
            data = normalize_json_fields(data)
            return data
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to handle the submit button click
# Function to handle the submit button click
# Replace the current transformation logic in handle_submit() with this improved version
# For better handling of complex nested data
def handle_submit():
    st.session_state.has_run = True
    ticker = st.session_state.ticker_input
    st.session_state.ticker = ticker

    # Show loading spinner
    with st.spinner(f"Analyzing SEC filings for {ticker}..."):
        # Fetch data using session_id to ensure fresh cache on page reload
        st.session_state.data = fetch_drug_data(ticker, st.session_state.session_id)

    # If data is loaded successfully
    if st.session_state.data:
        # Transform field names to match frontend expectations
        transformed_assets = []
        for asset in st.session_state.data.get('assets', []):
            # Process Animal_Models_Preclinical_Data
            animal_data = asset.get("Animal_Models_Preclinical_Data", [])
            if isinstance(animal_data, list) and animal_data:
                # Convert nested structure to string representation
                animal_data_str = json.dumps(animal_data, indent=2)
            elif isinstance(animal_data, str):
                # Keep as is if it's already a string
                animal_data_str = animal_data
            else:
                animal_data_str = "Not available"
            
            # Process Clinical_Trials
            clinical_trials = asset.get("Clinical_Trials", [])
            if isinstance(clinical_trials, list) and clinical_trials:
                # Convert nested structure to string representation
                trials_str = json.dumps(clinical_trials, indent=2)
            elif isinstance(clinical_trials, str):
                # Keep as is if it's already a string
                trials_str = clinical_trials
            else:
                trials_str = "Not available"
            
            # Process Upcoming_Milestones
            milestones = asset.get("Upcoming_Milestones", [])
            if isinstance(milestones, list) and milestones:
                milestones_str = json.dumps(milestones, indent=2)
            elif isinstance(milestones, str):
                milestones_str = milestones
            else:
                milestones_str = "Not available"
            
            # Process References
            references = asset.get("References", [])
            if isinstance(references, list) and references:
                references_str = json.dumps(references, indent=2)
            elif isinstance(references, str):
                references_str = references
            else:
                references_str = "Not available"
            
            transformed_asset = {
                "Name/Number": asset.get("Name/Number", "Not available"),
                "Mechanism of Action": asset.get("Mechanism_of_Action", "Not available"),
                "Target(s)": asset.get("Target", "Not available"),
                "Indication": asset.get("Indication", "Not available"),
                "Animal Models/Preclinical Data": animal_data_str,
                "Clinical Trials": trials_str,
                "Upcoming Milestones": milestones_str,
                "References": references_str
            }
            transformed_assets.append(transformed_asset)
        
        # Create a DataFrame from the transformed assets
        df = pd.DataFrame(transformed_assets)
        
        # Ensure all expected columns are visible
        st.session_state.df = ensure_columns_visible(df)
        
        # Set the filtered dataframe to be the same as the main dataframe initially
        st.session_state.filtered_df = st.session_state.df.copy()
def ensure_columns_visible(df):
    """
    Make sure all important columns are visible in the dataframe display
    """
    # Get the list of columns we want to ensure are displayed
    all_expected_columns = [
        "Name/Number", 
        "Mechanism of Action", 
        "Target(s)", 
        "Indication", 
        "Animal Models/Preclinical Data",
        "Clinical Trials", 
        "Upcoming Milestones", 
        "References"
    ]
    
    # Check which columns are missing and add them with placeholder content
    for column in all_expected_columns:
        if column not in df.columns:
            df[column] = "Not available"
    
    # Ensure columns are in the expected order
    return df[all_expected_columns]  
# Function to handle indication filter change
def filter_by_indication():
    selected_indication = st.session_state.selected_indication
    
    if selected_indication == 'All':
        st.session_state.filtered_df = st.session_state.df.copy()
    else:
        st.session_state.filtered_df = st.session_state.df[st.session_state.df['Indication'] == selected_indication]

# Sidebar for user input
with st.sidebar:
    st.header("Search Parameters")
    ticker_input = st.text_input("Enter Ticker Symbol", value=st.session_state.ticker if st.session_state.ticker else "WVE", key="ticker_input").upper()
    
    # Add cache clear button
    st.button("Clear Cache", on_click=clear_cache, help="Clear all cached data and reset the application")
    
    st.subheader("Filtering Options")
    
    # Only show filters if data is loaded
    if st.session_state.df is not None:
        # Get unique values for filtering
        indications = ['All'] + sorted(st.session_state.df['Indication'].unique().tolist())
        
        # Create the selectbox with a key and on_change callback
        st.selectbox(
            "Filter by Indication", 
            indications, 
            key="selected_indication",
            on_change=filter_by_indication,
            index=0
        )
        
        # Show asset count
        if st.session_state.filtered_df is not None:
            st.metric("Total Assets", len(st.session_state.filtered_df))
        
        # Download options
        st.subheader("Download Options")
        
        if st.session_state.filtered_df is not None:
            # Create CSV for download
            csv = st.session_state.filtered_df.to_csv(index=False)
            st.markdown(
                get_download_link(csv, f"{st.session_state.ticker}_drug_assets.csv", "Download CSV"),
                unsafe_allow_html=True
            )
            
            # Create Markdown for download
            md_content = f"# {st.session_state.ticker} Drug Asset Summary\n\n"
            md_content += df_to_markdown_with_delimiters(st.session_state.filtered_df)
            
            # Add additional sections
            filtered_df = st.session_state.filtered_df
            try:
                platforms = [d for d in filtered_df["Name/Number"] if "platform" in str(filtered_df.loc[filtered_df["Name/Number"] == d, "Mechanism of Action"].values[0]).lower()]
                if platforms:
                    md_content += "**Platform Technologies:**\n"
                    md_content += "\n".join([f"- {plat}" for plat in platforms]) + "\n\n"
            except:
                pass
            
            # Add disclaimer
            md_content += "*Note: Data extracted from SEC filings. May be incomplete due to limitations in filings or processing.*"
            
            st.markdown(
                get_download_link(md_content, f"{st.session_state.ticker}_drug_assets.md", "Download Markdown"),
                unsafe_allow_html=True
            )
    else:
        st.markdown("Filter options will appear after data is loaded")
    
    submitted = st.button("Analyze SEC Filings", type="primary", on_click=handle_submit)

# Main content area
if st.session_state.has_run and st.session_state.data:
    st.success(f"Successfully retrieved data for {st.session_state.ticker}")
    
    # Display the table in a clean format
    st.subheader(f"{st.session_state.ticker} Drug Asset Summary")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Standard View", "Expanded View"])
    
    with tab1:
    # Standard view with all essential columns
        display_columns = [
            "Name/Number", 
            "Mechanism of Action", 
            "Target(s)", 
            "Indication", 
            "Animal Models/Preclinical Data",
            "Clinical Trials", 
            "Upcoming Milestones"
        ]
        available_columns = [col for col in display_columns if col in st.session_state.filtered_df.columns]
    
    # Create a styled dataframe with proper column configurations
        st.dataframe(
            st.session_state.filtered_df[available_columns], 
            use_container_width=True, 
            height=500,
            column_config={
                "Animal Models/Preclinical Data": st.column_config.TextColumn(
                    "Animal Models/Preclinical Data",
                    width="large"
                ),
                "Clinical Trials": st.column_config.TextColumn(
                    "Clinical Trials",
                    width="large"
                ),
                "Upcoming Milestones": st.column_config.TextColumn(
                    "Upcoming Milestones",
                    width="medium"
                )
            }
        )

    with tab2:
        # Expanded view with all columns
        st.dataframe(
            st.session_state.filtered_df, 
            use_container_width=True, 
            height=600,
            column_config={
                "Animal Models/Preclinical Data": st.column_config.TextColumn(
                    "Animal Models/Preclinical Data",
                    width="large"
                ),
                "Clinical Trials": st.column_config.TextColumn(
                    "Clinical Trials",
                    width="large"
                ),
                "Upcoming Milestones": st.column_config.TextColumn(
                    "Upcoming Milestones",
                    width="medium"
                ),
                "References": st.column_config.TextColumn(
                    "References",
                    width="medium"
                )
            }
        )