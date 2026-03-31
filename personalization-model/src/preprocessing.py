"""
Data preprocessing pipeline for the customer affinity scoring engine.

Handles loading, cleaning, feature engineering, and encoding of customer
behavioral interaction data before model training or inference.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.model_config import model_config


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw customer interaction data from a CSV file.

    Args:
        filepath: Path to the CSV file containing customer-product interactions.

    Returns:
        DataFrame containing raw interaction records.
    """
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning to the raw customer interaction DataFrame.

    Removes duplicates, drops rows missing the target score, clips behavioral
    metrics to valid ranges, and normalises string columns to uppercase.

    Args:
        df: Raw customer interaction DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=[model_config.target_column])

    for col in model_config.categorical_features:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip().str.upper()

    if "Purchase_Frequency" in df.columns:
        df["Purchase_Frequency"] = pd.to_numeric(df["Purchase_Frequency"], errors="coerce").fillna(0).clip(0, 365)

    if "Avg_Order_Value" in df.columns:
        df["Avg_Order_Value"] = pd.to_numeric(df["Avg_Order_Value"], errors="coerce").fillna(0).clip(0)

    if "Days_Since_Last_Purchase" in df.columns:
        df["Days_Since_Last_Purchase"] = pd.to_numeric(df["Days_Since_Last_Purchase"], errors="coerce").fillna(999).clip(0, 999)

    if "Browse_Count" in df.columns:
        df["Browse_Count"] = pd.to_numeric(df["Browse_Count"], errors="coerce").fillna(0).clip(0)

    if "Cart_Abandonment_Rate" in df.columns:
        df["Cart_Abandonment_Rate"] = pd.to_numeric(df["Cart_Abandonment_Rate"], errors="coerce").fillna(0).clip(0, 1)

    if "Session_Duration_Mins" in df.columns:
        df["Session_Duration_Mins"] = pd.to_numeric(df["Session_Duration_Mins"], errors="coerce").fillna(0).clip(0)

    if model_config.target_column in df.columns:
        df[model_config.target_column] = df[model_config.target_column].clip(0.0, 1.0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived behavioral features to improve model signal quality.

    Adds:
    - recency_bucket: Categorical bucket for Days_Since_Last_Purchase.
    - engagement_score: Composite numeric score from browse + session metrics.
    - value_tier: Categorical tier based on Avg_Order_Value.

    Args:
        df: Cleaned customer interaction DataFrame.

    Returns:
        DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    df["recency_bucket"] = pd.cut(
        df["Days_Since_Last_Purchase"],
        bins=[0, 7, 30, 90, 180, 999],
        labels=["<7d", "7-30d", "30-90d", "90-180d", ">180d"],
        include_lowest=True,
    )

    df["engagement_score"] = (
        df.get("Browse_Count", 0) * 0.4
        + df.get("Session_Duration_Mins", 0) * 0.3
        + (1 - df.get("Cart_Abandonment_Rate", 0)) * 0.3
    ).round(4)

    if "Avg_Order_Value" in df.columns:
        df["value_tier"] = pd.cut(
            df["Avg_Order_Value"],
            bins=[0, 500, 2000, 5000, float("inf")],
            labels=["Budget", "Mid-Range", "Premium", "Luxury"],
            include_lowest=True,
        )

    return df


def encode_features(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: Optional[dict] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, dict, StandardScaler]:
    """
    Label-encode categorical columns and standard-scale numerical columns.

    Args:
        df: Engineered DataFrame to encode.
        fit: If True, fit new encoders/scaler. If False, use provided ones.
        encoders: Pre-fitted label encoder mapping (required when fit=False).
        scaler: Pre-fitted StandardScaler (required when fit=False).

    Returns:
        Tuple of (encoded DataFrame, encoder mapping, fitted scaler).
    """
    encoded = df.copy()
    if encoders is None:
        encoders = {}

    encode_cols = model_config.categorical_features + ["recency_bucket", "value_tier"]

    for col in encode_cols:
        if col not in encoded.columns:
            continue
        encoded[col] = encoded[col].astype(str)
        if fit:
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                known = set(le.classes_)
                encoded[col] = encoded[col].apply(lambda x: x if x in known else le.classes_[0])
                encoded[col] = le.transform(encoded[col])

    num_cols = [c for c in model_config.numerical_features + ["engagement_score"] if c in encoded.columns]
    if fit:
        scaler = StandardScaler()
        encoded[num_cols] = scaler.fit_transform(encoded[num_cols].fillna(0))
    else:
        encoded[num_cols] = scaler.transform(encoded[num_cols].fillna(0))

    return encoded, encoders, scaler


def prepare_training_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series, dict, StandardScaler]:
    """
    Full preprocessing pipeline for model training.

    Args:
        filepath: Path to the raw customer interaction CSV file.

    Returns:
        Tuple of (feature DataFrame, target Series, encoder mapping, fitted scaler).
    """
    df = load_raw_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders, scaler = encode_features(df, fit=True)

    feature_cols = (
        model_config.categorical_features
        + model_config.numerical_features
        + ["recency_bucket", "value_tier", "engagement_score"]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[model_config.target_column]
    return X, y, encoders, scaler


def prepare_inference_input(record: dict, encoders: dict, scaler: StandardScaler) -> pd.DataFrame:
    """
    Preprocess a single inference request dictionary into a model-ready DataFrame.

    Args:
        record: Dictionary with feature keys matching training columns.
        encoders: Fitted label encoder mapping from training.
        scaler: Fitted StandardScaler from training.

    Returns:
        Single-row DataFrame ready for model prediction.
    """
    df = pd.DataFrame([record])
    df = engineer_features(df)
    df, _, _ = encode_features(df, fit=False, encoders=encoders, scaler=scaler)
    return df
