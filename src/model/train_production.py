"""
Step 02: Production XGBoost Model
Trains on the synthetic transaction dataset with One-Hot Encoded merchant categories.
Outputs: Risk Ratio, Credit Score, Maximum Credit Purchase Amount.
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    SYNTHETIC_DATA_PATH, PRODUCTION_MODEL_PATH, PRODUCTION_ENCODER_PATH,
    MERCHANT_CATEGORIES, SCORE_MIN, SCORE_RANGE
)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder


def load_and_prepare_production(path: str) -> tuple:
    """
    Load the synthetic transaction dataset and apply One-Hot Encoding
    to the 'merchant_category' column to isolate vertical purchase variance.
    """
    df = pd.read_csv(path)
    
    target = "default_label"
    categorical_col = "merchant_category"
    id_col = "phone_number"
    
    # Drop identifier
    feature_df = df.drop(columns=[id_col, target])
    
    # One-Hot Encode merchant_category
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                            categories=[MERCHANT_CATEGORIES])
    encoded = encoder.fit_transform(feature_df[[categorical_col]])
    encoded_df = pd.DataFrame(encoded, columns=[f"cat_{c}" for c in MERCHANT_CATEGORIES])
    
    # Drop original categorical col and concat encoded
    feature_df = feature_df.drop(columns=[categorical_col])
    feature_df = pd.concat([feature_df.reset_index(drop=True), encoded_df], axis=1)
    
    X = feature_df.values
    y = df[target].values
    feature_names = feature_df.columns.tolist()
    
    return X, y, feature_names, encoder


def calculate_credit_limit(monthly_volume: float, score: float,
                            success_rate: float, insufficient_funds_count: int,
                            late_payment_count: int) -> float:
    """
    Dynamic credit limit formula.
    Penalizes insufficient funds and late payments.
    Returns 0 if score < 500.
    """
    if score < 500:
        return 0.0
    score_factor = (score - SCORE_MIN) / SCORE_RANGE  # 0 to 1
    base_limit = monthly_volume * 1.5 * score_factor * success_rate
    penalty = (500 * insufficient_funds_count) + (1500 * late_payment_count)
    limit = max(0.0, base_limit - penalty)
    return round(limit, 2)


def train_production(regenerate_data: bool = False):
    """
    Train the production XGBoost model on the synthetic transaction dataset.
    Returns the trained model, encoder, feature names, and performance metrics.
    """
    from src.model.data_generator import generate_synthetic_transactions
    
    print("=" * 60)
    print("STEP 02: PRODUCTION MODEL TRAINING (Synthetic Transactions)")
    print("=" * 60)
    
    # Generate or load data
    if regenerate_data or not os.path.exists(SYNTHETIC_DATA_PATH):
        from src.config import SYNTHETIC_DATASET_SIZE
        print(f"\nGenerating {SYNTHETIC_DATASET_SIZE:,} synthetic transaction records...")
        df = generate_synthetic_transactions()
        df.to_csv(SYNTHETIC_DATA_PATH, index=False)
        print(f"  Saved to: {SYNTHETIC_DATA_PATH}")
    else:
        print(f"\nLoading existing synthetic data from: {SYNTHETIC_DATA_PATH}")
    
    print("\nPreparing features...")
    X, y, feature_names, encoder = load_and_prepare_production(SYNTHETIC_DATA_PATH)
    
    df_raw = pd.read_csv(SYNTHETIC_DATA_PATH)
    print(f"  Samples: {len(X):,} | Features: {len(feature_names)} | Default rate: {y.mean():.2%}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining Production XGBoost Model...")
    pos_weight = max(1, int(np.sum(y == 0) / max(np.sum(y == 1), 1)))
    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
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
    
    # Feature importance
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\n  Top 10 Feature Importances:")
    print(feat_imp.head(10).to_string())
    
    # Save model and encoder
    os.makedirs(os.path.dirname(PRODUCTION_MODEL_PATH), exist_ok=True)
    model.save_model(PRODUCTION_MODEL_PATH)
    with open(PRODUCTION_ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    
    # Save feature names
    feat_names_path = PRODUCTION_MODEL_PATH.replace(".json", "_features.pkl")
    with open(feat_names_path, "wb") as f:
        pickle.dump(feature_names, f)
    
    print(f"\nProduction model saved to: {PRODUCTION_MODEL_PATH}")
    print("Production training complete!")
    
    return model, encoder, feature_names, {
        "auc": round(auc, 4),
        "default_rate": round(y.mean(), 4),
        "n_samples": len(X),
        "n_features": len(feature_names),
        "feature_importances": feat_imp.head(10).to_dict(),
    }


if __name__ == "__main__":
    train_production(regenerate_data=True)
