
---

# Financial RAG LLM

Retrieval-Augmented Generation (RAG) pipeline for financial documents and macroeconomic data.
This project enables natural-language question answering over SEC filings (10-K, 10-Q, etc.) and macroeconomic time series (FRED), combining structured and unstructured data with a vector database and local LLM inference via Ollama.

---

## Features

* **Financial Document Question Answering**

  * Ingests and parses SEC EDGAR 10-K filings.
  * Chunks and embeds text using SentenceTransformers with FAISS vector search.
  * Retrieves context and generates answers with citations.

* **Macroeconomic Data Integration**

  * Fetches CPI, unemployment, and other indicators from the FRED API.
  * Computes latest values and year-over-year changes.

* **Local LLM Integration**

  * Utilizes Ollama for local LLM inference (e.g., Qwen2.5).
  * Produces structured outputs including sentences, bullet points, and numbered lists.

* **Robust Query Handling**

  * Typo-tolerant corrections for user queries.
  * Pre-defined replacements for common misspellings.

* **Extensible Architecture**

  * Easily extendable to additional filings, macroeconomic series, or alternative LLM backends (AWS Bedrock, Azure OpenAI).

---

## Project Structure

```
LLM_pipeline/
├── app/                 # Core application code
│   ├── ingest/          # SEC ingestion: fetch, parse, index
│   ├── qa_cli.py        # Interactive Q&A CLI
│   ├── rag_prompt.py    # Builds LLM prompts
│   ├── ollama_client.py # Wrapper for local Ollama API
│   ├── text_utils.py    # Query autocorrection and vocab utilities
│   ├── formatting.py    # Answer formatting helpers
│   └── ...
├── data/                # Raw, interim, cache (ignored in git)
│   ├── raw/sec/         # Raw filings
│   ├── interim/sec/     # Parsed text
│   └── macro/           # Macroeconomic CSVs
├── processed/           # Embeddings and FAISS index (ignored in git)
├── requirements.txt     # Python dependencies
├── .env                 # API keys and configuration (ignored in git)
├── .gitignore           # Ignore rules for sensitive/large files
└── README.md            # Project documentation
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/brucewayneoptimusprime/Financial_RAG_LLM.git
cd Financial_RAG_LLM
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
# FRED API
FRED_API_KEY=your_fred_api_key_here

# SEC EDGAR
SEC_USER_AGENT=your_email@example.com
SEC_REQUEST_DELAY=0.4
```

### 5. Install and configure Ollama

* Install Ollama from: [https://ollama.ai](https://ollama.ai)
* Verify installation:

  ```bash
  ollama --version
  ```
* Pull the model (example: Qwen2.5 3B):

  ```bash
  ollama pull qwen2.5:3b-instruct
  ```

---

## Usage

### Interactive Q\&A

```bash
python app/qa_cli.py
```

Example:

```
You: What are Apple's main risk factors?
Assistant: The main risk factors include ...
Sources: [1] Apple_10K_2024.html p.1–1
```

### Fetch macroeconomic data

```bash
python app/macro_fetch.py
```

### Ingest SEC filings

```bash
# Map ticker to CIK
python -m app.ingest.sec_map AAPL MSFT

# Fetch the latest filing
python -m app.ingest.sec_fetch AAPL 10-K --limit 1

# Parse the filing
python -m app.ingest.sec_parse --ticker AAPL --latest

# Index the filing
python -m app.ingest.sec_index --ticker AAPL --latest
```

---

## Example Queries

* What are the main risk factors in Apple’s latest 10-K?
* Summarize Item 1A as bullet points.
* What is the latest CPI YoY?
* Compare Apple’s and Microsoft’s risk factors.

---

## Roadmap

* Extend macroeconomic dataset coverage (GDP, Federal Funds Rate, etc.).
* Support additional SEC forms (10-Q, 8-K, proxy statements).
* Improve summarization and structured answer formatting.
* Support deployment to cloud platforms (AWS Bedrock, Azure OpenAI).
* Build a web-based interactive interface.

---

## License

MIT License (can be adjusted as required).

---

## Author

Aditya Vikram Singh
B.Tech Computer Science, SRM Institute of Science & Technology
[GitHub](https://github.com/brucewayneoptimusprime)

---
