"""
Flask Backend API
Serves the credit scoring model via REST endpoints and hosts the frontend UI.
"""

import os
import sys
import json
import threading
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.model.pipeline import pipeline
from src.config import MERCHANT_CATEGORIES, SYNTHETIC_DATA_PATH

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

_training_lock = threading.Lock()
_is_training = False


def _do_training(regenerate: bool = False):
    """Background training function."""
    global _is_training
    try:
        from src.model.train_production import train_production
        model, encoder, feature_names, metrics = train_production(regenerate_data=regenerate)
        pipeline.load()
        pipeline.set_metrics(metrics)
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        _is_training = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", merchant_categories=MERCHANT_CATEGORIES)


@app.route("/api/status", methods=["GET"])
def status():
    """Return system status and model readiness."""
    global _is_training
    return jsonify({
        "model_loaded": pipeline.is_loaded(),
        "is_training": _is_training,
        "merchant_categories": MERCHANT_CATEGORIES,
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    """Return current model performance metrics."""
    m = pipeline.get_metrics()
    return jsonify(m if m else {"message": "Model not trained yet"})

import platform
import psutil

@app.route("/api/system_info", methods=["GET"])
def system_info():
    """Return hardware system info (CPU, RAM, Network) for both the process and global system."""
    process = psutil.Process(os.getpid())
    
    # Process CPU
    app_cpu = process.cpu_percent(interval=0.1)
    # Global CPU
    sys_cpu = psutil.cpu_percent(interval=0.1)
    
    # Process RAM (Resident Set Size)
    mem_info = process.memory_info()
    app_ram_mb = mem_info.rss / (1024 ** 2)
    
    # Global RAM
    sys_mem = psutil.virtual_memory()
    sys_ram_percent = sys_mem.percent
    sys_ram_gb = sys_mem.used / (1024 ** 3)
    sys_ram_total_gb = sys_mem.total / (1024 ** 3)
    
    # Network (Global)
    net = psutil.net_io_counters()

    return jsonify({
        "os": platform.system(),
        "app_cpu": f"{app_cpu:.1f}%",
        "sys_cpu": f"{sys_cpu:.1f}%",
        "app_ram": f"{app_ram_mb:.1f} MB",
        "sys_ram": f"{sys_ram_gb:.1f} GB / {sys_ram_total_gb:.1f} GB",
        "sys_ram_percent": f"{sys_ram_percent:.1f}%",
        "gpu_percent": "0.0%",
        "gpu_mem": "N/A (CPU-Only)",
        "net_bytes_sent": net.bytes_sent,
        "net_bytes_recv": net.bytes_recv
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Single-user credit score prediction.
    Accepts JSON body with transaction features.
    Returns risk_ratio, credit_score, risk_tier, max_credit_limit.
    """
    global _is_training
    
    if _is_training:
        return jsonify({"error": "Model is currently being retrained. Please wait."}), 503
    
    if not pipeline.is_loaded():
        return jsonify({"error": "Model not loaded. Please train first via /api/train"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    
    try:
        result = pipeline.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def train():
    """
    Trigger model training (or re-training with freshly generated data).
    Optional JSON body: {"regenerate": true} to regenerate the synthetic dataset.
    """
    global _is_training
    
    with _training_lock:
        if _is_training:
            return jsonify({"message": "Training already in progress"}), 409
        _is_training = True
    
    data = request.get_json(silent=True) or {}
    regenerate = data.get("regenerate", False)
    
    thread = threading.Thread(target=_do_training, args=(regenerate,), daemon=True)
    thread.start()
    
    return jsonify({
        "message": "Training started",
        "regenerate_data": regenerate,
    })


@app.route("/api/sample_data", methods=["GET"])
def sample_data():
    """Return a random sample of 20 records from the synthetic dataset with predictions."""
    global _is_training
    
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        return jsonify({"error": "Synthetic dataset not generated yet"}), 404
    
    if not pipeline.is_loaded():
        return jsonify({"error": "Model not loaded. Please train first."}), 503
    
    try:
        df = pd.read_csv(SYNTHETIC_DATA_PATH).sample(20, random_state=42)
        results = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            pred = pipeline.predict(row_dict)
            results.append({
                "phone_number": row_dict["phone_number"],
                "total_transactions": int(row_dict["total_transactions"]),
                "monthly_volume": float(row_dict["monthly_volume"]),
                "success_rate": float(row_dict["success_rate"]),
                "merchant_category": row_dict["merchant_category"],
                "default_label": int(row_dict["default_label"]),
                "risk_ratio": pred["risk_ratio"],
                "credit_score": pred["credit_score"],
                "risk_tier": pred["risk_tier"],
                "max_credit_limit": pred["max_credit_limit"],
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_features", methods=["POST"])
def update_features():
    """
    Simulate the 24-hour feature matrix update:
    Regenerates fresh synthetic data and retrains the production model.
    """
    global _is_training
    
    with _training_lock:
        if _is_training:
            return jsonify({"message": "Update already in progress"}), 409
        _is_training = True
    
    thread = threading.Thread(target=_do_training, args=(True,), daemon=True)
    thread.start()
    
    return jsonify({
        "message": "24-hour feature update initiated. Fresh data generated and model retraining started.",
        "regenerate_data": True,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Credit Scoring System - Backend API")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Open: http://127.0.0.1:5050\n")
    
    # Auto-train on startup if models don't exist
    from src.config import PRODUCTION_MODEL_PATH
    if not os.path.exists(PRODUCTION_MODEL_PATH):
        print("No trained model found. Starting initial training...")
        _is_training = True
        thread = threading.Thread(target=_do_training, args=(True,), daemon=True)
        thread.start()
    else:
        try:
            pipeline.load()
            print("Production model loaded successfully.")
        except Exception as e:
            print(f"Could not load model: {e}")
    
    app.run(debug=False, port=5050, host="0.0.0.0")
