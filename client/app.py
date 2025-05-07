import streamlit as st
import pandas as pd
import requests
import json
import base64
from io import StringIO

# Set page config
st.set_page_config(
    page_title="SEC Drug Asset Visualization",
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
API_URL = "http://0.0.0.0:8000/api/v1/filings/pipeline"

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
            if field in asset and asset[field].startswith('{') or asset[field].startswith('['):
                try:
                    # Try to parse the JSON
                    parsed = json.loads(asset[field])
                    # Convert back to a more readable format
                    asset[field] = json.dumps(parsed, indent=2)
                except:
                    # If parsing fails, keep as is
                    pass
    return data

# Function to fetch data from API
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

# Sidebar for user input
with st.sidebar:
    st.header("Search Parameters")
    ticker = st.text_input("Enter Ticker Symbol", "WVE").upper()
    
    st.subheader("Filtering Options")
    st.markdown("Filter options will appear after data is loaded")
    
    submitted = st.button("Analyze SEC Filings", type="primary")

# Main content area
if submitted:
    # Show loading spinner
    with st.spinner(f"Analyzing SEC filings for {ticker}..."):
        # Fetch data
        data = fetch_drug_data(ticker)
        
    # If data is loaded successfully
    if data:
        st.success(f"Successfully retrieved data for {ticker}")
        
        # Create a DataFrame from the assets
        df = pd.DataFrame(data.get('assets', []))
        
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
        df = df.rename(columns=column_mapping)
        
        # Add filtering options to sidebar now that we have data
        with st.sidebar:
            st.subheader("Filter Data")
            
            # Get unique values for filtering
            indications = ['All'] + sorted(df['Indication'].unique().tolist())
            selected_indication = st.selectbox("Filter by Indication", indications)
            
            # Apply filters
            filtered_df = df.copy()
            if selected_indication != 'All':
                filtered_df = filtered_df[filtered_df['Indication'] == selected_indication]
            
            # Show asset count
            st.metric("Total Assets", len(filtered_df))
            
            # Download options
            st.subheader("Download Options")
            
            # Create CSV for download
            csv = filtered_df.to_csv(index=False)
            st.markdown(
                get_download_link(csv, f"{ticker}_drug_assets.csv", "Download CSV"),
                unsafe_allow_html=True
            )
            
            # Create Markdown for download
            from tabulate import tabulate
            md_content = f"# {ticker} Drug Asset Summary\n\n"
            md_content += tabulate(filtered_df, headers="keys", tablefmt="pipe", showindex=False)
            md_content += "\n\n"
            
            # Add additional sections
            platforms = [d for d in filtered_df["Name/Number"] if "platform" in str(filtered_df.loc[filtered_df["Name/Number"] == d, "Mechanism of Action"].values[0]).lower()]
            if platforms:
                md_content += "**Platform Technologies:**\n"
                md_content += "\n".join([f"- {plat}" for plat in platforms]) + "\n\n"
            
            # Add disclaimer
            md_content += "*Note: Data extracted from SEC filings. May be incomplete due to limitations in filings or processing.*"
            
            st.markdown(
                get_download_link(md_content, f"{ticker}_drug_assets.md", "Download Markdown"),
                unsafe_allow_html=True
            )
        
        # Display the table in a clean format
        st.subheader(f"{ticker} Drug Asset Summary")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Standard View", "Expanded View"])
        
        with tab1:
            # Standard view with limited columns
            display_columns = ["Name/Number", "Mechanism of Action", "Target(s)", "Indication", "Clinical Trials", "Upcoming Milestones"]
            st.dataframe(filtered_df[display_columns], use_container_width=True, height=500)
        
        with tab2:
            # Expanded view with all columns
            st.dataframe(filtered_df, use_container_width=True, height=600)
        
        # Display additional analyses
        st.subheader("Asset Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Count by indication
            st.subheader("Assets by Indication")
            indication_counts = df['Indication'].value_counts()
            st.bar_chart(indication_counts)
        
        with col2:
            # Clinical trial phases distribution
            st.subheader("Clinical Trial Phases")
            
            # Extract phases from Clinical Trials field (this is simplified)
            phases = []
            for trial_info in df['Clinical Trials']:
                if 'Phase 1' in trial_info:
                    phases.append('Phase 1')
                elif 'Phase 2' in trial_info:
                    phases.append('Phase 2')
                elif 'Phase 3' in trial_info:
                    phases.append('Phase 3')
                elif 'Phase' in trial_info:
                    phases.append('Other Phase')
                else:
                    phases.append('Not Specified')
            
            phase_counts = pd.Series(phases).value_counts()
            st.bar_chart(phase_counts)
        
        # Display S3 paths if available
        if 's3_paths' in data:
            st.subheader("S3 Storage Locations")
            for file_type, path in data['s3_paths'].items():
                st.code(f"{file_type.upper()}: {path}")
    
    else:
        st.error(f"Failed to retrieve data for ticker {ticker}. Please check if the API is running and the ticker symbol is valid.")

# Initial state - before search
else:
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