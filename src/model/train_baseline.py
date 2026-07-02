"""
Step 01: Baseline XGBoost Model
Trains on the Kaggle GiveMeSomeCredit dataset to establish a credit scoring baseline.
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import KAGGLE_TRAIN_PATH, BASELINE_MODEL_PATH

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


def load_and_prepare_kaggle(path: str) -> tuple:
    """Load and clean the Kaggle GiveMeSomeCredit dataset."""
    df = pd.read_csv(path, index_col=0)
    
    # Rename for clarity
    df.rename(columns={"SeriousDlqin2yrs": "default_label"}, inplace=True)
    
    target = "default_label"
    features = [c for c in df.columns if c != target]
    
    # Impute missing values with median
    for col in features:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Cap extreme outliers at 99th percentile
    for col in features:
        cap = df[col].quantile(0.99)
        df[col] = np.minimum(df[col], cap)
    
    X = df[features].values
    y = df[target].values
    return X, y, features


def train_baseline():
    """Train the baseline XGBoost model on the Kaggle dataset."""
    print("=" * 60)
    print("STEP 01: BASELINE MODEL TRAINING (Kaggle GiveMeSomeCredit)")
    print("=" * 60)
    
    print("\nLoading dataset...")
    X, y, features = load_and_prepare_kaggle(KAGGLE_TRAIN_PATH)
    print(f"  Samples: {len(X):,} | Features: {len(features)} | Default rate: {y.mean():.2%}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining XGBoost Baseline Model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=int(np.sum(y == 0) / np.sum(y == 1)),
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print(f"\n  Validation AUC: {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Good (0)", "Default (1)"]))
    
    # Save model
    os.makedirs(os.path.dirname(BASELINE_MODEL_PATH), exist_ok=True)
    model.save_model(BASELINE_MODEL_PATH)
    print(f"\nBaseline model saved to: {BASELINE_MODEL_PATH}")
    
    # Save feature names
    feature_names_path = BASELINE_MODEL_PATH.replace(".json", "_features.pkl")
    with open(feature_names_path, "wb") as f:
        pickle.dump(features, f)
    
    print("\nBaseline training complete!")
    return model, auc


if __name__ == "__main__":
    train_baseline()
