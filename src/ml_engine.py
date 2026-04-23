import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    df_features = df_features.copy()
    df_features['sentence_idx'] = df_features.groupby(['Language', 'Type']).cumcount()
    
    df_waves = df_waves.copy()
    df_waves['sentence_idx'] = df_waves.groupby(['Language', 'Type']).cumcount()
    
    # 2. Merge on Language, Type, and the new index
    df_merged = pd.merge(
        df_features, 
        df_waves, 
        on=['Language', 'Type', 'sentence_idx'],
        how='inner'
    )
    
    # 3. Add Machine Learning Targets
    df_merged['is_real'] = (df_merged['Type'] == 'Real').astype(int)
    df_merged['Typology'] = df_merged['Language'].map(TYPOLOGY_MAP).fillna('Other')
    
    # 4. Clean up columns
    cols_to_drop = ['sentence_idx', 'Waveform', 'Interp_Waveform']
    df_ml = df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns])
    
    print(f"  ✓ Success: Merged into {df_ml.shape[0]} samples with {df_ml.shape[1]} columns.")
    print(f"  ✓ Targets created: 'is_real' and 'Typology'.")
    
    return df_ml

class DependencyML:
    def __init__(self):
        pass

    def get_feature_columns(self, df):
        # Exclude metadata and target columns
        exclude_cols = ['Language', 'Type', 'sentence_idx', 'Waveform', 'Interp_Waveform', 'is_real', 'Typology', 'Unnamed: 0']
        features = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        # Original Topological features:
        topo_base = ['Max_Arity', 'Avg_Arity', 'Max_Depth', 'Graph_Density', 'Avg_ICM', 'Avg_DLM', 'Directionality', 'Crossings']
        # Original Waveform features:
        wave_base = ['Mean', 'Std']
        
        # Determine new features
        new_features = [c for c in features if c not in topo_base and c not in wave_base]
        
        print(f"Detected Features ({len(features)} total):")
        print(f" - Topo Base: {topo_base}")
        print(f" - Wave Base: {wave_base}")
        print(f" - NEW Features Detected: {new_features}")
        
        topological = [c for c in topo_base if c in features]
        waveform = [c for c in wave_base if c in features] + new_features 
        
        return topological, waveform, features

    def train_evaluate(self, X_train, X_test, y_train, y_test, feature_names):
        clf = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        clf.fit(X_train[feature_names], y_train)
        preds = clf.predict(X_test[feature_names])
        probs = clf.predict_proba(X_test[feature_names])[:, 1]
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
        
        # Enhanced results
        report = classification_report(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        return clf, acc, f1, auc, report, cm

    def task_a_ablation(self, df):
        print("\n" + "="*50)
        print("=== Task A: Real vs. Random (Ablation Study) ===")
        print("="*50)
        
        topo_feats, wave_features, all_features = self.get_feature_columns(df)
        
        # Drop any missing values in targets or features
        df = df.dropna(subset=all_features + ['is_real', 'Language'])
        X = df
        y = df['is_real']
        
        # Stratify by Language so all languages are fairly represented
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['Language'])
        
        results = []
        
        # 1. Topology-Only
        print("\n--- Model 1: Topology-Only ---")
        clf_topo, acc, f1, auc, rep, cm = self.train_evaluate(X_train, X_test, y_train, y_test, topo_feats)
        results.append({'Model': 'Topology-Only', 'Accuracy': acc, 'F1-Score': f1, 'AUC': auc})
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", rep)
        
        # 2. Waveform + New Features
        print("\n--- Model 2: Waveform & New Features ---")
        clf_wave, acc, f1, auc, rep, cm = self.train_evaluate(X_train, X_test, y_train, y_test, wave_features)
        results.append({'Model': 'Waveform/New', 'Accuracy': acc, 'F1-Score': f1, 'AUC': auc})
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", rep)
        
        # 3. All Features
        print("\n--- Model 3: All Features Combined ---")
        clf_comb, acc, f1, auc, rep, cm = self.train_evaluate(X_train, X_test, y_train, y_test, all_features)
        results.append({'Model': 'All Features', 'Accuracy': acc, 'F1-Score': f1, 'AUC': auc})
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", rep)
        
        # FINAL RESULTS TABLE
        df_results = pd.DataFrame(results)
        print("\n" + "="*50)
        print("=== FINAL ABLATION RESULTS SUMMARY ===")
        print("="*50)
        print(df_results[['Model', 'Accuracy', 'F1-Score', 'AUC']].to_string(index=False))
        print("="*50)

        # Run SHAP Explainability on Model 3
        print("\n=== Running SHAP Explainability on Model 3 ===")
        explainer = shap.TreeExplainer(clf_comb)
        shap_values = explainer.shap_values(X_test[all_features])
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[all_features], show=False)
        plt.title("SHAP Summary Plot (Model C - All Features)", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()
        
        return df_results, clf_comb

    def plot_feature_importance(self, clf, feature_names, title):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(
            x=importances[indices], 
            y=np.array(feature_names)[indices], 
            palette="viridis",
            edgecolor='black',
            linewidth=1.2
        )
        
        # Add labels to the end of each bar for better visibility
        for i, v in enumerate(importances[indices]):
            ax.text(v + 0.005, i, f'{v:.3f}', color='black', va='center', fontweight='bold', fontsize=11)
            
        plt.title(title, fontsize=18, weight='bold', pad=20)
        plt.xlabel("XGBoost Feature Importance Score", fontsize=14, weight='semibold')
        plt.ylabel("Dependency Feature", fontsize=14, weight='semibold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def task_b_typology(self, df):
        """
        Task B: SOV vs SVO Typology Classification.
        Uses Leave-One-Language-Out (LOLO) cross-validation, which is the
        gold standard for typology prediction — it tests whether the model
        can generalize to an entirely unseen language.
        """
        print("\n" + "="*50)
        print("=== Task B: SOV vs. SVO Typology Classification ===")
        print("="*50)

        topo_feats, wave_features, all_features = self.get_feature_columns(df)

        # Filter to Real trees only and drop 'Other' typology
        df_real = df[(df['Type'] == 'Real') & (df['Typology'].isin(['SOV', 'SVO']))].copy()
        df_real = df_real.dropna(subset=all_features + ['Typology', 'Language'])

        # Encode target: SOV=0, SVO=1
        df_real['typo_label'] = (df_real['Typology'] == 'SVO').astype(int)

        X = df_real
        y = df_real['typo_label']
        groups = df_real['Language']

        logo = LeaveOneGroupOut()

        model_configs = [
            ('Topology-Only', topo_feats),
            ('Waveform/New', wave_features),
            ('All Features', all_features),
        ]

        results = []

        for model_name, feat_cols in model_configs:
            print(f"\n--- {model_name} (Leave-One-Language-Out) ---")
            all_preds = []
            all_true = []
            all_probs = []
            per_lang_results = []

            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                test_lang = groups.iloc[test_idx].iloc[0]

                clf = xgb.XGBClassifier(
                    n_estimators=100, random_state=42,
                    use_label_encoder=False, eval_metric='logloss', n_jobs=-1
                )
                clf.fit(X_train[feat_cols], y_train)
                preds = clf.predict(X_test[feat_cols])
                probs = clf.predict_proba(X_test[feat_cols])[:, 1]

                acc = accuracy_score(y_test, preds)
                f1_val = f1_score(y_test, preds, average='weighted')

                per_lang_results.append({
                    'Language': test_lang,
                    'True_Typology': 'SVO' if y_test.iloc[0] == 1 else 'SOV',
                    'Accuracy': acc,
                    'F1': f1_val
                })

                all_preds.extend(preds)
                all_true.extend(y_test)
                all_probs.extend(probs)

            # Overall metrics
            overall_acc = accuracy_score(all_true, all_preds)
            overall_f1 = f1_score(all_true, all_preds, average='weighted')
            try:
                overall_auc = roc_auc_score(all_true, all_probs)
            except Exception:
                overall_auc = np.nan

            results.append({
                'Model': model_name,
                'Accuracy': overall_acc,
                'F1-Score': overall_f1,
                'AUC': overall_auc
            })

            print(f"Overall Accuracy: {overall_acc:.4f} | F1: {overall_f1:.4f} | AUC: {overall_auc:.4f}")
            print("Confusion Matrix:")
            print(confusion_matrix(all_true, all_preds))
            print("Classification Report:")
            print(classification_report(all_true, all_preds, target_names=['SOV', 'SVO']))

            # Per-Language Breakdown
            df_per_lang = pd.DataFrame(per_lang_results)
            print("\nPer-Language Breakdown:")
            print(df_per_lang.to_string(index=False))

        # FINAL RESULTS TABLE
        df_results = pd.DataFrame(results)
        print("\n" + "="*50)
        print("=== TASK B: TYPOLOGY RESULTS SUMMARY ===")
        print("="*50)
        print(df_results[['Model', 'Accuracy', 'F1-Score', 'AUC']].to_string(index=False))
        print("="*50)

        # Train a final "All Features" model on the full dataset for SHAP/Importance visualization
        print("\n=== Running Feature Importance & SHAP for Task B (All Features) ===")
        clf_full = xgb.XGBClassifier(
            n_estimators=100, random_state=42,
            use_label_encoder=False, eval_metric='logloss', n_jobs=-1
        )
        clf_full.fit(X[all_features], y)

        # XGBoost Feature Importance Bar Plot
        self.plot_feature_importance(clf_full, all_features, "Task B: Feature Importance (SOV vs SVO)")

        # SHAP Summary Plot
        explainer = shap.TreeExplainer(clf_full)
        shap_values = explainer.shap_values(X[all_features])
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X[all_features], show=False)
        plt.title("SHAP Summary Plot (Task B - SOV vs SVO)", fontsize=18, weight='bold', pad=15)
        plt.tight_layout()
        plt.show()

        return df_results
