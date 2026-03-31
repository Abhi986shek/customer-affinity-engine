"""
SageMaker training entry point for the customer affinity scoring model.

Executed inside the SageMaker training container. Reads hyperparameters
from the environment, loads training data from the SageMaker input channel,
trains the XGBoost affinity model, evaluates it, and saves the artifact.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import load_raw_data, clean_data, engineer_features, encode_features, prepare_training_data
from src.model import CustomerAffinityModel
from scripts.evaluate_model import compute_rmse, compute_coverage_by_confidence


def parse_args() -> argparse.Namespace:
    """
    Parse SageMaker-injected hyperparameters and channel paths.

    Returns:
        Parsed argument namespace with all training configuration values.
    """
    parser = argparse.ArgumentParser(description="Train the customer affinity scoring model")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--n-estimators", type=int, default=int(os.environ.get("n_estimators", "300")))
    parser.add_argument("--max-depth", type=int, default=int(os.environ.get("max_depth", "6")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("learning_rate", "0.05")))
    parser.add_argument("--subsample", type=float, default=float(os.environ.get("subsample", "0.8")))
    return parser.parse_args()


def find_training_file(train_dir: str) -> str:
    """
    Locate the first CSV file in the SageMaker training input channel directory.

    Args:
        train_dir: Path to the training channel directory.

    Returns:
        Full path to the first CSV file found.

    Raises:
        FileNotFoundError: If no CSV exists in the directory.
    """
    for filename in os.listdir(train_dir):
        if filename.endswith(".csv"):
            return os.path.join(train_dir, filename)
    raise FileNotFoundError(f"No CSV training file found in: {train_dir}")


def main() -> None:
    """
    Full training pipeline executed by SageMaker.

    Loads data, preprocesses, trains the XGBoost model, evaluates RMSE
    and confidence coverage, and saves the model artifact with metrics.
    """
    args = parse_args()

    print(f"Training config: n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"learning_rate={args.learning_rate}, subsample={args.subsample}")

    data_path = find_training_file(args.train)
    print(f"Loading training data from: {data_path}")

    df = load_raw_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    print(f"Training data loaded: {len(df)} records")

    _, encoders, scaler = encode_features(df, fit=True)

    from config.model_config import model_config
    feature_cols = (
        model_config.categorical_features
        + model_config.numerical_features
        + ["recency_bucket", "value_tier", "engagement_score"]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols]
    y = df[model_config.target_column]

    print("Training XGBoost affinity model...")
    model = CustomerAffinityModel()
    model.fit(X, y, encoders, scaler)

    rmse = compute_rmse(model, X, y)
    coverage = compute_coverage_by_confidence(model, X)
    print(f"Training RMSE: {rmse:.4f}")
    print(f"Confidence coverage: {coverage}")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "affinity_model.pkl")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    metrics = {
        "rmse": round(rmse, 4),
        "confidence_coverage": coverage,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "training_samples": len(df),
    }
    metrics_path = os.path.join(args.model_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
