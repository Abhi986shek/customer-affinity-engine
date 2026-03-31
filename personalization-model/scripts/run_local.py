"""
Local end-to-end run script for the customer affinity scoring engine.

Trains the XGBoost model from a local CSV file, evaluates it, and
optionally stores affinity records. Use this for local development
and testing without SageMaker.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import load_raw_data, clean_data, engineer_features, encode_features
from src.model import CustomerAffinityModel
from scripts.evaluate_model import compute_rmse, compute_coverage_by_confidence
from config.model_config import model_config


def run_local(data_path: str, model_output: str = None) -> None:
    """
    Execute the full preprocessing → training → evaluation pipeline locally.

    Args:
        data_path: Path to the customer interaction CSV file.
        model_output: Optional path to save the fitted model artifact.
    """
    print("=" * 60)
    print("CustomerAffinity — Local Training Pipeline")
    print("=" * 60)

    print(f"\n[1/4] Loading data from: {data_path}")
    df = load_raw_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    print(f"      {len(df)} records loaded")

    print("\n[2/4] Encoding features...")
    df_enc, encoders, scaler = encode_features(df, fit=True)

    feature_cols = (
        model_config.categorical_features
        + model_config.numerical_features
        + ["recency_bucket", "value_tier", "engagement_score"]
    )
    feature_cols = [c for c in feature_cols if c in df_enc.columns]
    X = df_enc[feature_cols]
    y = df_enc[model_config.target_column]

    print("\n[3/4] Training XGBoost model...")
    model = CustomerAffinityModel()
    model.fit(X, y, encoders, scaler)

    if model_output:
        model.save(model_output)
        print(f"      Model saved to: {model_output}")

    print("\n[4/4] Evaluating...")
    rmse = compute_rmse(model, X, y)
    confidence_dist = compute_coverage_by_confidence(model, X)
    print(f"      RMSE:            {rmse:.4f}")
    print(f"      Confidence dist: {confidence_dist}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the affinity scoring pipeline locally")
    parser.add_argument("--data", required=True, help="Path to customer interaction CSV")
    parser.add_argument("--model-output", default=None, help="Optional path to save the model")
    args = parser.parse_args()
    run_local(args.data, args.model_output)
