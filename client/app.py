import streamlit as st
import pandas as pd
import requests
import json
import base64
from io import StringIO
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="K-Cap Funding SEC Drug Asset Visualization",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Function to fetch data from API - cached to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_drug_data(ticker):
    """Fetch drug data from the API"""
    try:
        response = requests.post(
            API_URL,
            json={"ticker": ticker},
            headers={"Content-Type": "application/json"}
        )
        
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
def handle_submit():
    st.session_state.has_run = True
    ticker = st.session_state.ticker_input
    st.session_state.ticker = ticker
    
    # Show loading spinner
    with st.spinner(f"Analyzing SEC filings for {ticker}..."):
        # Fetch data
        st.session_state.data = fetch_drug_data(ticker)
    
    # If data is loaded successfully
    if st.session_state.data:
        # Create a DataFrame from the assets
        st.session_state.df = pd.DataFrame(st.session_state.data.get('assets', []))
        
        # Rename columns for display
        column_mapping = {
            "Name/Number": "Name/Number",
            "Mechanism of Action": "Mechanism of Action",
            "Target(s)": "Target(s)",
            "Indication": "Indication",
            "Animal Models/Preclinical Data": "Animal Models/Preclinical Data",
            "Clinical Trials": "Clinical Trials",
            "Upcoming Milestones": "Upcoming Milestones",
            "References": "References"
        }
        st.session_state.df = st.session_state.df.rename(columns=column_mapping)
        
        # Set the filtered dataframe to be the same as the main dataframe initially
        st.session_state.filtered_df = st.session_state.df.copy()

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
        # Standard view with limited columns
        display_columns = ["Name/Number", "Mechanism of Action", "Target(s)", "Indication", "Clinical Trials", "Upcoming Milestones"]
        available_columns = [col for col in display_columns if col in st.session_state.filtered_df.columns]
        st.dataframe(st.session_state.filtered_df[available_columns], use_container_width=True, height=500)
    
    with tab2:
        # Expanded view with all columns
        st.dataframe(st.session_state.filtered_df, use_container_width=True, height=600)
    
    # Display additional analyses
    st.subheader("Asset Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count by indication
        st.subheader("Assets by Indication")
        indication_counts = st.session_state.df['Indication'].value_counts()
        st.bar_chart(indication_counts)
    
    with col2:
        # Clinical trial phases distribution
        st.subheader("Clinical Trial Phases")
        
        # Extract phases from Clinical Trials field (this is simplified)
        if 'Clinical Trials' in st.session_state.df.columns:
            phases = []
            for trial_info in st.session_state.df['Clinical Trials']:
                if 'Phase 1' in str(trial_info):
                    phases.append('Phase 1')
                elif 'Phase 2' in str(trial_info):
                    phases.append('Phase 2')
                elif 'Phase 3' in str(trial_info):
                    phases.append('Phase 3')
                elif 'Phase' in str(trial_info):
                    phases.append('Other Phase')
                else:
                    phases.append('Not Specified')
            
            phase_counts = pd.Series(phases).value_counts()
            st.bar_chart(phase_counts)
    
    # Display S3 paths if available
    if 's3_paths' in st.session_state.data:
        st.subheader("S3 Storage Locations")
        for file_type, path in st.session_state.data['s3_paths'].items():
            st.code(f"{file_type.upper()}: {path}")

# Initial state - before search
elif not st.session_state.has_run:
    st.info("Enter a ticker symbol in the sidebar and click 'Analyze SEC Filings' to start.")
    
    # Show example of what the output will look like
    st.subheader("Example Output")
    
    example_data = {
        "Name/Number": ["Drug-A", "Drug-B", "Platform-X"],
        "Mechanism of Action": ["RNA interference", "Antisense oligonucleotide", "Novel delivery system"],
        "Target(s)": ["Gene X", "Protein Y", "Multiple targets"],
        "Indication": ["Rare Disease A", "Cancer", "Platform technology"],
        "Clinical Trials": ["Phase 2", "Phase 1", "Preclinical"]
    }
    
    st.dataframe(pd.DataFrame(example_data), use_container_width=True)