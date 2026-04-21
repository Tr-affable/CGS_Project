import os
import pandas as pd
import numpy as np

# Typology Mapping (Head-Final vs Head-Initial)
TYPOLOGY_MAP = {
    "hindi": "SOV",
    "japanese": "SOV",
    "korean": "SOV",
    "turkish": "SOV",
    "latin": "SOV",
    "english": "SVO",
    "french": "SVO",
    "spanish": "SVO",
    "russian": "SVO",
    "Italian": "SVO"
}

def prepare_dataset(df_features: pd.DataFrame, df_waves: pd.DataFrame):
    """
    Cleans and merges topological features with temporal waveform features.
    Ensures that sentence ordering matches across datasets.
    """
    print("Preparing consolidated ML dataset...")
    
    # 1. Ensure common indices for merging
    # We add a 'sentence_idx' within each (Language, Type) group 
    # to guarantee we aren't merging Sentence A's static features 
    # with Sentence B's waveform.
    df_features = df_features.copy()
    df_features['sentence_idx'] = df_features.groupby(['Language', 'Type']).cumcount()
    
    df_waves = df_waves.copy()
    df_waves['sentence_idx'] = df_waves.groupby(['Language', 'Type']).cumcount()
    
    # 2. Merge on Language, Type, and the new index
    # We only keep rows that exist in both (Inner Join)
    df_merged = pd.merge(
        df_features, 
        df_waves, 
        on=['Language', 'Type', 'sentence_idx'],
        how='inner'
    )
    
    # 3. Add Machine Learning Targets
    # Target A: Real (1) vs Random (0)
    df_merged['is_real'] = (df_merged['Type'] == 'Real').astype(int)
    
    # Target B: Typology Mapping
    df_merged['Typology'] = df_merged['Language'].map(TYPOLOGY_MAP).fillna('Other')
    
    # 4. Clean up columns
    # We drop 'Waveform' and 'Interp_Waveform' for standard ML 
    # (unless we were doing Deep Learning on the raw signals)
    cols_to_drop = ['sentence_idx', 'Waveform', 'Interp_Waveform']
    df_ml = df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns])
    
    print(f"  ✓ Success: Merged into {df_ml.shape[0]} samples with {df_ml.shape[1]} columns.")
    print(f"  ✓ Targets created: 'is_real' and 'Typology'.")
    
    return df_ml
