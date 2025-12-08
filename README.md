#   Insights Agent (RAG) + Semantic Data Quality Guardrail

A GenAI Platform Prototype demonstrating **Agentic Data Quality** and **Self-Service Analytics** using the AWS Native stack.

##  Architecture
*   **Ingestion:** Python-based ETL with a **Hybrid Data Loader** (Real + Synthetic injection).
*   **Guardrails:** A **Semantic Judge Agent** (Amazon Titan/Claude) that scans data for PII, Sentiment Mismatches, and Ambiguity *before* ingestion.
*   **Storage:** **ChromaDB** (Vector Store) for RAG.
*   **Retrieval:** **Amazon Titan Embeddings** for semantic search.
*   **Reasoning:** **Amazon Titan Text / Claude 3** for summarization.
*   **Frontend:** Streamlit.

##  Key Features
1.  **Quarantine Pattern:** Toxic data (PII, Mismatched Sentiment) is intercepted and logged to a "Quarantine" view, preventing vector store pollution.
2.  **RAG Pipeline:** Users can query unstructured customer reviews using natural language.
3.  **AWS Native:** Built using `boto3` and Bedrock to ensure VPC-readiness.

##  Setup
1.  Configure AWS Credentials (`aws configure`).
2.  Install dependencies:
    ```bash
    pip install streamlit boto3 pandas chromadb
    ```
3.  Run the application:
    ```bash
    python -m streamlit run app.py
    ```