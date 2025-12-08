import chromadb
import pandas as pd

def ingest_reviews(df, bedrock_client):
    """
    Ingests data, runs Semantic DQ, and segregates Clean vs. Quarantined.
    Arguments:
        df: The Pandas DataFrame containing reviews.
        bedrock_client: The initialized BedrockClient object.
    Returns: 
        collection: The ChromaDB collection
        quarantine_df: Pandas DataFrame of blocked rows
    """
    # 1. Setup Chroma (In-Memory)
    chroma_client = chromadb.Client()
    # Reset collection if it exists to avoid duplicates during testing
    try:
        chroma_client.delete_collection("ring_reviews")
    except:
        pass
    collection = chroma_client.create_collection(name="ring_reviews")
    
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    # List to hold bad rows
    quarantine_list = []
    
    print("🚀 Starting Semantic Ingestion (With DQ Guardrails)...")
    
    # Iterate through the DataFrame
    # This is where your error was: df is now correctly the DataFrame
    for index, row in df.iterrows():
        text_chunk = f"Title: {row['title']}\nReview: {row['text']}"
        
        # --- THE GATEKEEPER ---
        # Run the DQ Check using the passed bedrock_client
        dq_result = bedrock_client.validate_review(row['text'], row['rating'])
        
        if not dq_result['is_valid']:
            # 🛑 BLOCK: Add to Quarantine
            print(f"🛑 BLOCKED Row {index}: {dq_result['reason']}")
            quarantine_list.append({
                "original_index": index,
                "text": row['text'],
                "source": row.get('source', 'Unknown'),
                "block_reason": dq_result['reason']
            })
            continue # Skip the embedding step for this row
        
        # ✅ PASS: Proceed to Embed
        try:
            vector = bedrock_client.get_embedding(text_chunk)
            
            ids.append(str(index))
            documents.append(text_chunk)
            embeddings.append(vector)
            
            # Metadata must be strings/ints/floats (no Nones)
            metadatas.append({
                "rating": int(row['rating']), 
                "category": str(row.get('category', 'General')),
                "source": str(row.get('source', 'Unknown')),
                "asin": str(row.get('asin', 'N/A'))
            })
        except Exception as e:
            print(f"⚠️ Error embedding row {index}: {e}")
            continue
        
    # Add only CLEAN data to Chroma
    if ids:
        collection.add(
            ids=ids, 
            documents=documents, 
            embeddings=embeddings, 
            metadatas=metadatas
        )
        print(f"✅ Ingested {len(ids)} clean documents.")
        
    return collection, pd.DataFrame(quarantine_list)