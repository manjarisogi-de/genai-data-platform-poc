import pandas as pd
import numpy as np
import os

def load_hybrid_data(filepath='amazon_reviews.csv'):
    """
    Loads real Amazon reviews from a local CSV and combines them with
    synthetic 'trap' data to test Data Quality agents.
    """
    
    # --- Step 1: Load Real Data ---
    if os.path.exists(filepath):
        print(f"Loading real data from {filepath}...")
        df_real = pd.read_csv(filepath)
        
        # Select first 45 rows
        df_real = df_real.head(45).copy()
        
        # Strict Schema Mapping
        # name -> title
        # reviews.text -> text
        # reviews.rating -> rating
        # asins -> asin
        rename_map = {
            'name': 'title',
            'reviews.text': 'text',
            'reviews.rating': 'rating',
            'asins': 'asin'
        }
        
        # Check if columns exist before renaming to avoid KeyErrors, 
        # or blindly rename if we trust the source structure 100%. 
        # Here we assume the source CSV strictly follows the provided schema keys.
        df_real.rename(columns=rename_map, inplace=True)
        
        # Keep only desired columns
        desired_cols = ['title', 'text', 'rating', 'asin']
        # Filter only existing columns to be safe, but fill missing ones if necessary
        for col in desired_cols:
            if col not in df_real.columns:
                df_real[col] = None # Fill missing with None
        
        df_real = df_real[desired_cols]
        
        # Clean up: Ensure rating is an integer
        df_real['rating'] = pd.to_numeric(df_real['rating'], errors='coerce').fillna(0).astype(int)
        
        # Add source column
        df_real['source'] = 'Real_Local'
        
    else:
        print(f"Warning: {filepath} not found. Returning empty real dataframe.")
        df_real = pd.DataFrame(columns=['title', 'text', 'rating', 'asin', 'source'])

    # --- Step 2: Synthetic 'Trap' Data (The Semantic DQ Test) ---
    print("Generating synthetic trap data...")
    traps = [
        {
            "title": "Bad",
            "text": "Refund me! Call 555-0199.", # Trap 1 (PII)
            "rating": 1,
            "asin": "SYNTH_PII_001",
            "source": "Synthetic_Trap"
        },
        {
            "title": "Terrible",
            "text": "Absolute garbage, do not buy.", # Trap 2 (Sentiment Mismatch)
            "rating": 5, # High rating for negative text
            "asin": "SYNTH_DQ_002",
            "source": "Synthetic_Trap"
        },
        {
            "title": "Coffee?",
            "text": "The coffee tastes burnt.", # Trap 3 (Irrelevant)
            "rating": 3,
            "asin": "SYNTH_IRR_003",
            "source": "Synthetic_Trap"
        },
        {
            "title": "???",
            "text": "asdf jkl;", # Trap 4 (Gibberish)
            "rating": 1,
            "asin": "SYNTH_GIB_004",
            "source": "Synthetic_Trap"
        },
        {
            "title": "Battery",
            "text": "It is okay but the battery died.", # Trap 5 (Ambiguous)
            "rating": 2,
            "asin": "SYNTH_AMB_005",
            "source": "Synthetic_Trap"
        }
    ]
    
    df_synthetic = pd.DataFrame(traps)
    
    # --- Step 3: Merge and Shuffle ---
    df_final = pd.concat([df_real, df_synthetic], ignore_index=True)
    
    # Shuffle the rows
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_final

if __name__ == "__main__":
    # Test the loader
    # Create a dummy csv for testing purposes if it doesn't exist
    if not os.path.exists('amazon_reviews.csv'):
        dummy_data = {
            'name': [f'Product {i}' for i in range(50)],
            'reviews.text': [f'Review text {i}' for i in range(50)],
            'reviews.rating': [5] * 50,
            'asins': [f'B000{i}' for i in range(50)]
        }
        pd.DataFrame(dummy_data).to_csv('amazon_reviews.csv', index=False)
        print("Created dummy amazon_reviews.csv for testing.")

    df = load_hybrid_data()
    print(f"Loaded {len(df)} rows.")
    print(df['source'].value_counts())
    print("\nSample Data:")
    print(df.head())
