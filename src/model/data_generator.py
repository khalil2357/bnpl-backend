"""
Synthetic Transaction Dataset Generator
Generates 50,000 realistic BNPL transaction records for model training.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import SYNTHETIC_DATA_PATH, SYNTHETIC_DATASET_SIZE, MERCHANT_CATEGORIES


def generate_synthetic_transactions(n: int = SYNTHETIC_DATASET_SIZE, seed: int = 42) -> pd.DataFrame:
    """
    Generates n rows of synthetic BNPL transaction data with realistic correlations.
    Features:
        - phone_number (unique identifier)
        - total_transactions
        - monthly_volume (BDT)
        - success_rate
        - avg_gap_days
        - max_gap_days
        - insufficient_funds_count
        - merchant_category
        - late_payment_count
        - default_label (0=Good, 1=Default)
    """
    rng = np.random.default_rng(seed)

    # --- Customer risk tiers (latent variable) ---
    # 70% good customers, 20% medium risk, 10% high risk
    risk_tier = rng.choice([0, 1, 2], size=n, p=[0.70, 0.20, 0.10])

    # --- Phone numbers (unique) ---
    base_numbers = rng.integers(1_700_000_000, 1_999_999_999, size=n)
    # Ensure uniqueness
    unique_numbers = list(dict.fromkeys(base_numbers.tolist()))
    while len(unique_numbers) < n:
        extra = rng.integers(1_700_000_000, 1_999_999_999, size=(n - len(unique_numbers) + 100))
        unique_numbers = list(dict.fromkeys(unique_numbers + extra.tolist()))[:n]
    phone_numbers = ["0" + str(x)[-9:] for x in unique_numbers[:n]]

    # --- Total transactions (risk tiers: good customers transact more) ---
    total_transactions = np.where(
        risk_tier == 0,
        rng.integers(20, 200, size=n),
        np.where(risk_tier == 1, rng.integers(5, 60, size=n), rng.integers(1, 30, size=n)),
    )

    # --- Monthly volume (BDT) ---
    monthly_volume = np.where(
        risk_tier == 0,
        rng.uniform(5000, 150000, size=n),
        np.where(risk_tier == 1, rng.uniform(1000, 30000, size=n), rng.uniform(500, 10000, size=n)),
    )

    # --- Success rate ---
    success_rate = np.clip(
        np.where(
            risk_tier == 0,
            rng.beta(9, 1, size=n),  # high success
            np.where(risk_tier == 1, rng.beta(5, 3, size=n), rng.beta(2, 5, size=n)),  # lower success
        ),
        0.01, 1.0,
    )

    # --- Avg gap days (days between transactions) ---
    avg_gap_days = np.clip(
        np.where(
            risk_tier == 0,
            rng.exponential(scale=3, size=n),
            np.where(risk_tier == 1, rng.exponential(scale=10, size=n), rng.exponential(scale=20, size=n)),
        ),
        0.5, 90.0,
    )

    # --- Max gap days ---
    max_gap_days = avg_gap_days * rng.uniform(1.5, 4.5, size=n)
    max_gap_days = np.clip(max_gap_days, avg_gap_days + 1, 180.0)

    # --- Insufficient funds count ---
    insufficient_funds_count = np.where(
        risk_tier == 0,
        rng.integers(0, 2, size=n),
        np.where(risk_tier == 1, rng.integers(1, 8, size=n), rng.integers(3, 25, size=n)),
    )

    # --- Merchant category ---
    category_probs = {
        0: [0.25, 0.30, 0.20, 0.10, 0.10, 0.05],   # Good: mostly grocery/electronics
        1: [0.20, 0.20, 0.25, 0.15, 0.10, 0.10],   # Medium: fashion/travel heavier
        2: [0.15, 0.15, 0.20, 0.20, 0.10, 0.20],   # High risk: travel/food heavier
    }
    merchant_category = np.array([
        rng.choice(MERCHANT_CATEGORIES, p=category_probs[t]) for t in risk_tier
    ])

    # --- Late payment count ---
    late_payment_count = np.where(
        risk_tier == 0,
        rng.integers(0, 2, size=n),
        np.where(risk_tier == 1, rng.integers(1, 6, size=n), rng.integers(3, 20, size=n)),
    )

    # --- Default label (computed probabilistically from features) ---
    # Build a linear score to simulate real relationship
    log_odds = (
        -4.0
        + 2.5 * (risk_tier == 2).astype(float)
        + 1.2 * (risk_tier == 1).astype(float)
        - 0.05 * success_rate * 100
        + 0.15 * avg_gap_days
        + 0.08 * max_gap_days
        + 0.6 * insufficient_funds_count
        + 0.8 * late_payment_count
        - 0.00002 * monthly_volume
        + 0.01 * (90 - np.minimum(total_transactions, 90))
        + rng.normal(0, 0.1, size=n)  # reduced noise
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default_label = (rng.uniform(0, 1, size=n) < prob_default).astype(int)

    df = pd.DataFrame({
        "phone_number": phone_numbers,
        "total_transactions": total_transactions,
        "monthly_volume": np.round(monthly_volume, 2),
        "success_rate": np.round(success_rate, 4),
        "avg_gap_days": np.round(avg_gap_days, 2),
        "max_gap_days": np.round(max_gap_days, 2),
        "insufficient_funds_count": insufficient_funds_count,
        "merchant_category": merchant_category,
        "late_payment_count": late_payment_count,
        "default_label": default_label,
    })

    return df


if __name__ == "__main__":
    print(f"Generating {SYNTHETIC_DATASET_SIZE:,} synthetic transaction records...")
    df = generate_synthetic_transactions()
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f"Saved to: {SYNTHETIC_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['default_label'].mean():.2%}")
    print(f"\nSample:\n{df.head(3)}")
