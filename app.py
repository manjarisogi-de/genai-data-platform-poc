import streamlit as st
import pandas as pd
from bedrock_client import BedrockClient
from data_loader import load_hybrid_data
from vector_store import ingest_reviews 

# --- 1. CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Internal Insights Agent", layout="wide")

# --- 2. CACHED RESOURCE ---
@st.cache_resource
def get_knowledge_base():
    """
    Initialize resources.
    Cached so it only runs once per session.
    Returns: (Chroma Collection, Bedrock Client, Quarantine DataFrame)
    """
    # 1. Init Bedrock Client 
    bedrock = BedrockClient()
    
    # 2. Load Data (Real + Synthetic)
    df = load_hybrid_data()
    
    # 3. Ingest Data (Returns Collection AND Quarantine Log)
    collection, quarantine_df = ingest_reviews(df, bedrock)
    
    return collection, bedrock, quarantine_df

# --- 3. MAIN APP UI ---
def main():
    st.title("🛡️ Insights Agent")
    st.caption("Powered by AWS Bedrock (Titan Embeddings + Claude 3 Haiku) & ChromaDB")

    # --- INITIALIZATION PHASE ---
    if "kb_resources" not in st.session_state:
        with st.spinner("Initializing System (Running Semantic DQ)..."):
            # Unpack and store all three resources
            st.session_state.kb_resources = get_knowledge_base()
        st.success("System Ready. Knowledge Base Loaded.")

    # Unpack resources from session state
    collection, bedrock, quarantine_df = st.session_state.kb_resources
    
    # --- UI TABS ---
    tab1, tab2 = st.tabs(["🤖 Search Agent (Clean Data)", "🛡️ DQ Quarantine Log"])
    
    # ==========================================
    # TAB 1: THE SEARCH AGENT (Product View)
    # ==========================================
    with tab1:
        st.info("ℹ️ This agent searches ONLY validated data. Bad data has been blocked.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            query = st.text_input("Ask about Battery/Camera issues:", placeholder="e.g., Is the battery life reliable in cold weather?")

        if query:
            st.subheader("Results")
            
            # A. RETRIEVAL (RAG)
            with st.spinner("Searching knowledge base..."):
                try:
                    # 1. Embed the User's Query using Titan
                    query_embedding = bedrock.get_embedding(query)
                    
                    # 2. Search ChromaDB
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3
                    )
                    
                    if results['documents']:
                        retrieved_docs = results['documents'][0]
                        retrieved_metas = results['metadatas'][0]
                    else:
                        st.warning("No relevant documents found.")
                        return
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    return
                
            # Display Retrieved Context
            st.markdown("##### 🔍 Retrieved Context")
            cols = st.columns(3)
            context_str = ""
            
            for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
                source = meta.get('source', 'Unknown')
                rating = meta.get('rating', 'N/A')
                asin = meta.get('asin', 'N/A')

                with cols[idx]:
                    with st.container(border=True):
                        st.caption(f"Source: {source}")
                        st.write(doc[:200] + "..." if len(doc) > 200 else doc)
                        st.caption(f"Rating: {rating} | ASIN: {asin}")
                
                context_str += f"REVIEW {idx+1} (Source: {source}, Rating: {rating}):\n{doc}\n\n"
                
            # B. GENERATION (LLM)
            st.markdown("#####  AI Summary")
            with st.spinner("Generating answer with Claude 3 Haiku..."):
                prompt = f"""
                You are a helpful customer support agent analyzing product feedback.
                USER QUESTION: "{query}"
                RETRIEVED REVIEWS:
                {context_str}
                Based ONLY on the reviews, answer the user.
                """
                try:
                    answer = bedrock.generate_text(prompt)
                    st.info(answer)
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    # ==========================================
    # TAB 2: THE QUARANTINE LOG (Engineering View)
    # ==========================================
    with tab2:
        st.header(" Semantic Data Quality Log")
        st.markdown("The following rows were **blocked** from the Vector Store by the AI Guardrail.")
        
        # Metrics Row
        m1, m2 = st.columns(2)
        m1.metric("Total Rows Blocked", len(quarantine_df))
        m2.metric("Blocking Agent", "Claude 3 Haiku")
        
        st.divider()

        if not quarantine_df.empty:
            # Display the DataFrame with specific column formatting
            st.dataframe(
                quarantine_df, 
                column_config={
                    "block_reason": st.column_config.TextColumn("⚠️ Reason", width="medium"),
                    "text": st.column_config.TextColumn("Raw Review Text", width="large"),
                    "source": "Source ID",
                    "original_index": "Row ID"
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No data quality issues detected. All rows passed.")

if __name__ == "__main__":
    main()