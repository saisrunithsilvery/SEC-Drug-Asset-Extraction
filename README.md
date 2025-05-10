# SEC Filing Analysis

The SEC Drug Asset Extraction Pipeline is an intelligent data processing system that automatically extracts, organizes, and summarizes drug development programs from biotech companies' SEC filings (10-K and 8-K reports). This tool transforms scattered, technical filings into structured, chronological summaries of each drug asset, complete with mechanisms of action, targets, indications, and development history.

#code labs link - https://codelabs-preview.appspot.com/?file_id=1J2qgr5sLVbDpQWztI2nhbNEzX7CLqYung32U6MnvQDk#0

# SEC Drug Asset Extraction Pipeline



## Key Features

- **Automated Extraction**: Processes SEC filings to identify and extract drug asset information
- **Comprehensive Drug Profiles**: Generates structured tables with key drug information fields
- **Historical Reconciliation**: Tracks drug development progress across multiple filings
- **Vector Search**: Employs semantic search to find relevant information within large documents
- **Multiple Output Formats**: Generates outputs in CSV and Markdown formats

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- AWS account (for cloud deployment option)



## Architecture

The pipeline consists of the following key components:

1. **Ingestion Layer**: Retrieves SEC filings via the EDGAR API
2. **Processing Layer**: Cleans documents and identifies relevant sections
3. **Vector Database**: Creates searchable embeddings of document chunks
4. **Analysis Layer**: Extracts drug information using NLP techniques
5. **Reconciliation Layer**: Consolidates information across multiple filings
6. **Output Layer**: Generates structured tables in CSV and Markdown formats


## Data Processing

The system uses several techniques to process SEC filings:

1. **Document Chunking**: Breaks large filings into manageable segments
2. **Vector Embeddings**: Converts text chunks into semantic vectors for similarity search
3. **Entity Recognition**: Identifies drug names, targets, and indications
4. **Information Extraction**: Pulls out structured data from unstructured text
5. **Chronological Reconciliation**: Resolves information across multiple time periods

## Output Format

The pipeline produces a structured table with the following columns:

| Column Name | Description |
|-------------|-------------|
| Name/Number | Drug name or identifier (e.g., WVE-N531) |
| Mechanism of Action | How the drug works (e.g., siRNA to silence gene expression) |
| Target | Biological molecule/pathway targeted (e.g., dystrophin pre-mRNA) |
| Indication | Target disease (e.g., Duchenne muscular dystrophy) |
| Animal Models / Preclinical Data | Bullet-point summaries of experiments with model type, results, and year |
| Clinical Trials | Grouped bullets for each trial (phase, N, duration, results, AEs, dates) |
| Upcoming Milestones | Key expected catalysts or events |
| References | Section headers, document types (10K/8K), and filing dates used as sources |

## Cloud Deployment 

For processing large volumes of filings, the system can be deployed to AWS:

1. Set up AWS credentials and configure the AWS CLI
2. Run the deployment script:
```bash
python deploy_cloud.py
```

This creates:
- Lambda functions for document processing
- S3 buckets for document storage
- API Gateway endpoints for interaction

## Troubleshooting

### Common Issues

- **SEC API Rate Limits**: The pipeline implements backoff strategies, but you may need to adjust timing
- **Memory Usage**: Processing large companies with many filings requires sufficient memory
- **Missing Information**: Some drug programs may have limited information in SEC filings

### Logging

Enable verbose logging to troubleshoot issues:

```bash
python extract_drugs.py --ticker MRNA --verbose
```

Logs are stored in `./logs` by default.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request






## System Architecture

The system is built using a modular architecture:

- **Controllers**: Handle request processing and orchestrate service calls
- **Services**: Implement core business logic for filing retrieval, vectorization, and analysis
- **Routes**: Define API endpoints for different operations
- **Models**: Provide Pydantic schemas for request/response validation
- **Utils**: Offer reusable functions for S3, SEC API, and other operations

## Prerequisites

- Python 3.8+
- AWS S3 bucket with proper permissions
- SEC API key (https://sec-api.io/)
- OpenAI API key for GPT models
- Hugging Face model access for embeddings

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sec-api-backend.git
   cd sec-api-backend
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`:
   ```
   cp .env.example .env
   ```

5. Update the `.env` file with your credentials and settings

## Usage

### Running the server

Start the FastAPI server:

```
python -m app.main
```

The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs

### API Endpoints

#### 1. Fetch SEC Filings

```
POST /api/v1/filings/fetch
```

Request body:
```json
{
  "ticker": "WVE",
  "form_types": ["10-K", "8-K"],
  "max_filings": 100
}
```

#### 2. Build Vector Store

```
POST /api/v1/filings/vectorize
```

Request body:
```json
{
  "ticker": "WVE"
}
```

#### 3. Analyze Drug Assets

```
POST /api/v1/filings/analyze
```

Request body:
```json
{
  "ticker": "WVE"
}
```

#### 4. Complete Pipeline

```
POST /api/v1/filings/pipeline
```

Request body:
```json
{
  "ticker": "WVE",
  "form_types": ["10-K", "8-K"],
  "max_filings": 100
}
```

## Example Workflow

1. First, fetch the SEC filings for a ticker:
   ```
   curl -X POST http://localhost:8000/api/v1/filings/fetch \
     -H "Content-Type: application/json" \
     -d '{"ticker": "WVE"}'
   ```

2. Then, build the vector store from the filings:
   ```
   curl -X POST http://localhost:8000/api/v1/filings/vectorize \
     -H "Content-Type: application/json" \
     -d '{"ticker": "WVE"}'
   ```

3. Finally, analyze the filings to extract drug information:
   ```
   curl -X POST http://localhost:8000/api/v1/filings/analyze \
     -H "Content-Type: application/json" \
     -d '{"ticker": "WVE"}'
   ```

4. Or, run the complete pipeline in one step:
   ```
   curl -X POST http://localhost:8000/api/v1/filings/pipeline \
     -H "Content-Type: application/json" \
     -d '{"ticker": "WVE"}'
   ```

## Output Files

The system generates the following output files in your S3 bucket:

- HTML filings: `s3://your-bucket/filings/{ticker}/{accessionNo}_{formType}.html`
- Metadata CSV: `s3://your-bucket/filings/{ticker}/metadata.csv`
- FAISS index: `s3://your-bucket/filings/{ticker}/faiss_index/`
- Analysis results:
  - CSV: `s3://your-bucket/filings/{ticker}/drug_summary.csv`
  - Markdown: `s3://your-bucket/filings/{ticker}/drug_summary.md`

