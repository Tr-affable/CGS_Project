import os
import pickle
import numpy as np
import pandas as pd
from waveform import compute_memory_burden

RANDOM_DATA_DIR = "Random_Data"
ABSTRACT_ROOT = 1000
MAX_SENTENCES = 1500

def load_and_compute(n_bins=20):
    """
    Loads all real and random trees from the Random_Data directory,
    computes their Active Memory Burden waveforms, and interpolates
    them into `n_bins` for normalized comparison.
    
    Returns a unified pandas DataFrame with the waveform features.
    """
    rows = []
    
    for fname in sorted(os.listdir(RANDOM_DATA_DIR)):
        if not (fname.startswith("Random_Data_") and fname.endswith(".pkl")):
            continue
            
        lang = fname.replace("Random_Data_", "").replace(".pkl", "")
        filepath = os.path.join(RANDOM_DATA_DIR, fname)
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        if len(data) > MAX_SENTENCES:
            data = data[:MAX_SENTENCES]
            
        for rec in data:
            real_tree = rec["tree"]
            random_tree = rec.get("random_tree")
            
            # Helper to process waveform
            def process_waveform(wf, type_label):
                if not wf or len(wf) < 2: return
                
                # Interpolate to target bins (e.g. 20 bins)
                orig_x = np.linspace(0, 100, len(wf))
                new_x = np.linspace(0, 100, n_bins)
                interp_wf = np.interp(new_x, orig_x, wf)
                
                rows.append({
                    "Language": lang, 
                    "Type": type_label, 
                    "Variance": float(np.var(wf)), 
                    "Std": float(np.std(wf)), 
                    "Peak": max(wf), 
                    "Mean": float(np.mean(wf)), 
                    "Length": len(wf), 
                    "Waveform": wf, 
                    "Interp_Waveform": interp_wf
                })

            # Real tree waveform (root = 0)
            wf_real = compute_memory_burden(real_tree, root=0)
            if wf_real: 
                process_waveform(wf_real, "Real")
                
            # Random tree waveform (root = ABSTRACT_ROOT)
            if random_tree is not None:
                wf_rand = compute_memory_burden(random_tree, root=ABSTRACT_ROOT)
                if wf_rand: 
                    process_waveform(wf_rand, "Random")
                
    return pd.DataFrame(rows)

def load_master_features():
    """
    Loads the topological features from Master_Features.csv 
    (computed in Step 2) safely.
    """
    features_file = os.path.join(RANDOM_DATA_DIR, "Master_Features.csv")
    if os.path.exists(features_file):
        return pd.read_csv(features_file)
    return None
