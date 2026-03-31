"""
Model evaluation utilities for the customer affinity scoring engine.

Computes RMSE, confidence distribution, and segment-level coverage metrics
for a fitted CustomerAffinityModel against a reference dataset.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import CustomerAffinityModel
from src.preprocessing import load_raw_data, clean_data, engineer_features, encode_features
from config.model_config import model_config


def compute_rmse(model: CustomerAffinityModel, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Calculate Root Mean Squared Error of affinity score predictions.

    Args:
        model: Fitted CustomerAffinityModel instance.
        X: Encoded feature DataFrame.
        y: Ground-truth affinity scores.

    Returns:
        RMSE value as a float.
    """
    predictions = model.predict(X)
    return float(np.sqrt(np.mean((predictions - y.values) ** 2)))


def compute_coverage_by_confidence(model: CustomerAffinityModel, X: pd.DataFrame) -> dict:
    """
    Calculate what fraction of predictions fall in each confidence band.

    Args:
        model: Fitted CustomerAffinityModel instance.
        X: Encoded feature DataFrame.

    Returns:
        Dictionary mapping confidence labels to their percentage share.
    """
    predictions = model.predict(X)
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for score in predictions:
        for label, (lo, hi) in model_config.confidence_levels.items():
            if lo <= score <= hi:
                counts[label] += 1
                break
    total = len(predictions)
    return {k: round(v / total, 4) if total > 0 else 0 for k, v in counts.items()}


def run_evaluation(model_dir: str, data_path: str, output_dir: str) -> dict:
    """
    Full evaluation run: load model, compute metrics, write evaluation.json.

    Args:
        model_dir: Directory containing the saved model artifact.
        data_path: Path to the validation CSV file.
        output_dir: Directory where evaluation.json is written.

    Returns:
        Evaluation metrics dictionary.
    """
    model_path = os.path.join(model_dir, "affinity_model.pkl")
    model = CustomerAffinityModel.load(model_path)

    df = load_raw_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    df, _, _ = encode_features(df, fit=False, encoders=model.encoders_, scaler=model.scaler_)

    feature_cols = [c for c in model.feature_cols_ if c in df.columns]
    X = df[feature_cols]
    y = df[model_config.target_column]

    rmse = compute_rmse(model, X, y)
    confidence_dist = compute_coverage_by_confidence(model, X)

    metrics = {
        "accuracy": round(max(0.0, 1.0 - rmse), 4),
        "rmse": round(rmse, 4),
        "confidence_distribution": confidence_dist,
        "evaluation_samples": len(df),
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation complete. Metrics written to: {output_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the customer affinity model")
    parser.add_argument("--model-dir", required=True, help="Directory containing affinity_model.pkl")
    parser.add_argument("--data", required=True, help="Path to validation CSV file")
    parser.add_argument("--output-dir", default="./evaluation", help="Output directory for evaluation.json")
    args = parser.parse_args()
    run_evaluation(args.model_dir, args.data, args.output_dir)
