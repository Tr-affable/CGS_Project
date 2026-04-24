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
                
                # --- Advanced Length-Normalized Features ---
                length = len(wf)
                mean_val = float(np.mean(wf))
                peak_val = max(wf)
                
                # 1. Relative Peak Position (0.0 to 1.0)
                # Where in the sentence does the maximum memory burden occur?
                rel_peak_pos = float(np.argmax(wf)) / length
                
                # 2. Max-to-Mean Ratio (Stress Gap)
                # How much higher is the peak compared to the average?
                peak_mean_ratio = float(peak_val / mean_val) if mean_val > 0 else 0.0
                
                # 3. Change Rate (Volatility Density)
                # Average change in burden per word (step-to-step absolute difference)
                abs_diffs = np.abs(np.diff(wf))
                change_rate = float(np.sum(abs_diffs)) / length
                
                # 4. Waveform Skewness & Kurtosis
                # use pandas Series for these stat moments
                wf_series = pd.Series(wf)
                skewness = float(wf_series.skew()) if length >= 3 else 0.0
                kurtosis = float(wf_series.kurt()) if length >= 4 else 0.0
                
                # 5. AUC (Area Under Curve) - total energy
                # Using trapezoidal integration normalized by sentence length
                auc = float(np.trapz(wf)) / length
                
                # 6. Peak Count (Cognitive Bottlenecks)
                # Simple local maxima count: word where burden is > neighbors
                # normalized by length (peaks per word)
                peak_count = 0
                for j in range(1, length - 1):
                    if wf[j] > wf[j-1] and wf[j] >= wf[j+1]:
                        peak_count += 1
                if length > 0:
                    peak_count = (peak_count + 1) / length # +1 for global max if monotonic
                
                # 7. Temporal Centroid (Mass Center)
                # Weighted average of positions by burden (normalized 0.0 to 1.0)
                burden_sum = np.sum(wf)
                if burden_sum > 0:
                    centroid = np.sum(np.arange(length) * wf) / (burden_sum * length)
                else:
                    centroid = 0.5
                
                rows.append({
                    "Language": lang, 
                    "Type": type_label, 
                    "Variance": float(np.var(wf)), 
                    "Std": float(np.std(wf)), 
                    "Peak": peak_val, 
                    "Mean": mean_val, 
                    "Length": length, 
                    "Rel_Peak_Pos": rel_peak_pos,
                    "Peak_Mean_Ratio": peak_mean_ratio,
                    "Change_Rate": change_rate,
                    "Skewness": skewness if pd.notna(skewness) else 0.0,
                    "Kurtosis": kurtosis if pd.notna(kurtosis) else 0.0,
                    "AUC": auc,
                    "Peak_Count": peak_count,
                    "Centroid": centroid,
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
