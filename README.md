# Intech Engineering RAG System

## Project Overview
This repository contains a Retrieval-Augmented Generation (RAG) system designed to index and query technical documentation stored in Google Drive. The system is specifically configured to process technical specifications for aviation test stands and financial reports, enabling efficient data retrieval for the engineering team.

## Key Functionalities
- **Automated Ingestion:** Direct integration with Google Drive API for document extraction.
- **Selective Indexing:** Automated exclusion logic for operational files (e.g., Data_Monday).
- **Vectorized Search:** High-performance semantic search utilizing Gemini Pro embeddings.
- **Incremental Synchronization:** Differential updates to ensure data parity with the cloud repository.

## Technical Stack
- **Language:** Python 3.10+
- **Frameworks:** LlamaIndex, Streamlit.
- **LLM Provider:** Google Gemini API.
- **Data Source:** Google Drive API.

## Repository Structure
- `/src`: Core logic modules for data loading and vector indexing.
- `app.py`: Streamlit-based user interface for natural language querying.
- `.github/workflows`: CI/CD pipelines for automated data synchronization.
