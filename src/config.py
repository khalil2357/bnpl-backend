"""
AI Credit Scoring System
Configuration constants shared across the project.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
KAGGLE_TRAIN_PATH = os.path.join(BASE_DIR, "GiveMeSomeCredit-training.csv")
SYNTHETIC_DATA_PATH = os.path.join(BASE_DIR, "synthetic_transactions.csv")

# Model save paths
BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_xgb.json")
PRODUCTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "production_xgb.json")
PRODUCTION_ENCODER_PATH = os.path.join(BASE_DIR, "models", "production_encoder.pkl")

# Synthetic dataset size
SYNTHETIC_DATASET_SIZE = 50000

# Merchant categories
MERCHANT_CATEGORIES = ["Electronics", "Grocery", "Fashion", "Travel", "Utilities", "Food"]

# Credit scoring parameters
SCORE_MIN = 300
SCORE_MAX = 850
SCORE_RANGE = SCORE_MAX - SCORE_MIN  # 550

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5,
    "use_label_encoder": False,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}
