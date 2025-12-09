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

   


---

### **The "Online" Architecture: The 4-Stage Reduction Funnel**

*Optimizing for the fundamental constraint of GenAI: **The Context Window Limit.**

You physically **cannot** send 1 million records to Bedrock in real-time because:
1.  **Token Limit:** Even Claude 3 (200k tokens) can only hold about ~500 reviews max.
2.  **Latency:** It would take minutes to process.
3.  **Cost:** It would cost $20+ per query.*

 **RAG is essentially a massive filtering funnel.**


*"For the Real-Time path, my goal is to reduce 1 Million records down to the **'Golden 10'** that fit in the prompt. I use a 4-stage funnel:"*

#### **Stage 1: Metadata Filtering (The SQL Filter)**
*   **User Intent:** "How was the battery **last week**?"
*   **Action:** We don't search the whole Vector DB. We apply a **Hard Filter** first.
*   **Tech:** `WHERE timestamp > NOW() - 7 DAYS AND category = 'Battery'`.
*   **Reduction:** 1,000,000 $\rightarrow$ 5,000 candidates.

#### **Stage 2: Vector Retrieval (The Semantic Filter)**
*   **Action:** We compare the user's query embedding against those 5,000 candidates.
*   **Tech:** Cosine Similarity Search.
*   **Reduction:** 5,000 $\rightarrow$ **Top 20 Chunks** (most relevant snippets).

#### **Stage 3: Deterministic Tool Use (The Metric)**
*   **Action:** As you noted, we do **not** ask the LLM to calculate the average rating from those 20 chunks (it’s bad at math).
*   **Tech:** The Router calls a SQL Tool: `SELECT AVG(rating)...`.
*   **Result:** A single number: "3.2 Stars".

#### **Stage 4: Context Stuffing & Synthesis (The LLM)**
*   **Action:** We construct the final prompt.
*   **The Prompt:**
    > "User asked about battery last week.
    > Hard Data: Average Rating is 3.2.
    > Relevant Reviews: [Insert the Top 20 text chunks here].
    > Task: Summarize the reviews and use the rating to verify sentiment."
*   **Reduction:** The LLM processes only ~4,000 tokens (fast and cheap).

---



### **Summary Visualization**

| Step | Data Volume | Technology | Cost |
| :--- | :--- | :--- | :--- |
| **Total Data** | 1,000,000 rows | Data Lake / S3 | Storage Only |
| **1. Metadata Filter** | 5,000 rows | Vector DB (Filter) | Low (Index lookup) |
| **2. Semantic Search** | 20 rows | Vector DB (KNN) | Low (Compute) |
| **3. Final Prompt** | ~3,000 Tokens | LLM Context Window | Medium (Inference) |

