# SEC Filing Analysis Backend

This FastAPI backend provides a comprehensive solution for fetching, processing, and analyzing SEC filings, with a specific focus on extracting drug asset information for biotech companies.

## Features

- **SEC Filing Retrieval**: Fetches 10-K, 8-K, and other filing types for any ticker symbol
- **Document Processing**: Chunks SEC filings and builds vector embeddings for efficient retrieval
- **Drug Asset Analysis**: Extracts detailed information about drugs, programs, and platforms
- **Complete Pipeline**: Offers step-by-step or end-to-end processing options

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

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.