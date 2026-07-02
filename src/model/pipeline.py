"""
Inference Pipeline
Loads trained models and provides a unified interface for credit scoring predictions.
Outputs: Risk Ratio, Credit Score, Maximum Credit Purchase Amount.
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    PRODUCTION_MODEL_PATH, PRODUCTION_ENCODER_PATH,
    BASELINE_MODEL_PATH, MERCHANT_CATEGORIES,
    SCORE_MIN, SCORE_RANGE
)

from xgboost import XGBClassifier


def _load_model(path: str) -> XGBClassifier:
    """Load an XGBoost model from disk."""
    model = XGBClassifier()
    model.load_model(path)
    return model


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_credit_score(risk_prob: float) -> int:
    """Convert risk probability to credit score (300–850)."""
    return int(SCORE_MIN + (1 - risk_prob) * SCORE_RANGE)


def compute_credit_limit(monthly_volume: float, score: int,
                          success_rate: float,
                          insufficient_funds_count: int,
                          late_payment_count: int) -> float:
    """
    Calculate the Maximum Credit Purchase Amount.
    Returns 0 if score < 500 (high-risk block).
    """
    if score < 500:
        return 0.0
    score_factor = (score - SCORE_MIN) / SCORE_RANGE
    base_limit = monthly_volume * 1.5 * score_factor * success_rate
    penalty = (500 * insufficient_funds_count) + (1500 * late_payment_count)
    return round(max(0.0, base_limit - penalty), 2)


def get_risk_tier(score: int) -> str:
    """Return a human-readable risk tier label based on credit score."""
    if score >= 750:
        return "LOW RISK"
    elif score >= 600:
        return "MODERATE RISK"
    elif score >= 500:
        return "HIGH RISK"
    else:
        return "VERY HIGH RISK"


class CreditScoringPipeline:
    """
    Unified inference interface for the production credit scoring system.
    """
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.feature_names = None
        self._metrics = {}
        self._loaded = False
    
    def load(self):
        """Load production model, encoder, and feature names from disk."""
        if not os.path.exists(PRODUCTION_MODEL_PATH):
            raise FileNotFoundError(
                f"Production model not found at {PRODUCTION_MODEL_PATH}. "
                "Please run training first."
            )
        self.model = _load_model(PRODUCTION_MODEL_PATH)
        self.encoder = _load_pickle(PRODUCTION_ENCODER_PATH)
        feat_path = PRODUCTION_MODEL_PATH.replace(".json", "_features.pkl")
        self.feature_names = _load_pickle(feat_path)
        self._loaded = True
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def predict(self, input_data: dict) -> dict:
        """
        Run inference on a single customer record.
        
        Args:
            input_data: dict with keys:
                - phone_number (str, optional)
                - total_transactions (int)
                - monthly_volume (float)
                - success_rate (float, 0–1)
                - avg_gap_days (float)
                - max_gap_days (float)
                - insufficient_funds_count (int)
                - merchant_category (str)
                - late_payment_count (int)
        
        Returns:
            dict with:
                - risk_ratio (float, 0–1)
                - credit_score (int, 300–850)
                - risk_tier (str)
                - max_credit_limit (float, BDT)
        """
        if not self._loaded:
            self.load()
        
        # Validate merchant category
        merchant = input_data.get("merchant_category", "Grocery")
        if merchant not in MERCHANT_CATEGORIES:
            merchant = "Grocery"
        
        # One-hot encode merchant category
        encoded = self.encoder.transform([[merchant]])
        cat_cols = {f"cat_{c}": encoded[0][i] for i, c in enumerate(MERCHANT_CATEGORIES)}
        
        # Build feature row
        row = {
            "total_transactions": float(input_data.get("total_transactions", 50)),
            "monthly_volume": float(input_data.get("monthly_volume", 10000)),
            "success_rate": float(input_data.get("success_rate", 0.9)),
            "avg_gap_days": float(input_data.get("avg_gap_days", 5)),
            "max_gap_days": float(input_data.get("max_gap_days", 15)),
            "insufficient_funds_count": float(input_data.get("insufficient_funds_count", 0)),
            "late_payment_count": float(input_data.get("late_payment_count", 0)),
        }
        row.update(cat_cols)
        
        # Align to training feature order
        X = np.array([[row.get(f, 0.0) for f in self.feature_names]])
        
        # Predict
        risk_prob = float(self.model.predict_proba(X)[0, 1])
        credit_score = compute_credit_score(risk_prob)
        credit_limit = compute_credit_limit(
            monthly_volume=row["monthly_volume"],
            score=credit_score,
            success_rate=row["success_rate"],
            insufficient_funds_count=int(row["insufficient_funds_count"]),
            late_payment_count=int(row["late_payment_count"]),
        )
        
        return {
            "phone_number": input_data.get("phone_number", "N/A"),
            "risk_ratio": round(risk_prob, 6),
            "credit_score": credit_score,
            "risk_tier": get_risk_tier(credit_score),
            "max_credit_limit": credit_limit,
        }
    
    def set_metrics(self, metrics: dict):
        self._metrics = metrics
    
    def get_metrics(self) -> dict:
        return self._metrics


# Singleton instance
pipeline = CreditScoringPipeline()
