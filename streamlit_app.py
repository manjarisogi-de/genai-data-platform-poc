import streamlit as st
import pandas as pd
import chromadb
from bedrock_client import BedrockClient

# Page Config
st.set_page_config(page_title="Bedrock RAG App", layout="wide")

@st.cache_resource
def get_bedrock_client():
    return BedrockClient()

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

def main():
    st.title("AWS Bedrock & ChromaDB Explorer")
    
    bedrock = get_bedrock_client()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text Generation")
        prompt = st.text_area("Ask Claude 3 Haiku:")
        if st.button("Generate"):
            with st.spinner("Thinking..."):
                try:
                    response = bedrock.generate_text(prompt)
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
                    
    with col2:
        st.subheader("Embeddings")
        text_to_embed = st.text_input("Text to embed:")
        if st.button("Get Embedding"):
            with st.spinner("Embedding..."):
                try:
                    vec = bedrock.get_embedding(text_to_embed)
                    st.success(f"Generated vector of dimension {len(vec)}")
                    st.json(vec[:5]) # Show first 5 dims
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()