"""
XGBoost-based customer affinity scoring model.

Wraps an XGBoost regressor to predict a continuous affinity score (0–1)
representing the probability that a given customer will engage with a
specific product. Provides fit, predict, save, and load helpers.
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config.model_config import model_config


class CustomerAffinityModel:
    """
    Gradient Boosting affinity scoring model for customer-product personalisation.

    Uses XGBoost regression to predict an Affinity_Score in the range [0, 1]
    for a given combination of customer attributes and behavioral signals with
    product context features.

    Attributes:
        model_: Fitted XGBRegressor instance.
        encoders_: Label encoder mapping fitted during training.
        scaler_: StandardScaler fitted during training.
        feature_cols_: List of feature column names in training order.
        is_fitted_: Whether the model has been trained.
    """

    def __init__(self):
        self.model_: Optional[xgb.XGBRegressor] = None
        self.encoders_: dict = {}
        self.scaler_: Optional[StandardScaler] = None
        self.feature_cols_: list = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        encoders: dict,
        scaler: StandardScaler,
    ) -> "CustomerAffinityModel":
        """
        Train the XGBoost regressor on the prepared feature matrix.

        Args:
            X: Encoded and scaled feature DataFrame from preprocessing pipeline.
            y: Affinity score target Series (values in [0, 1]).
            encoders: Fitted label encoder mapping from preprocessing.
            scaler: Fitted StandardScaler from preprocessing.

        Returns:
            Self reference for method chaining.
        """
        self.model_ = xgb.XGBRegressor(
            n_estimators=model_config.n_estimators,
            max_depth=model_config.max_depth,
            learning_rate=model_config.learning_rate,
            subsample=model_config.subsample,
            colsample_bytree=model_config.colsample_bytree,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
        )
        self.model_.fit(X, y)
        self.encoders_ = encoders
        self.scaler_ = scaler
        self.feature_cols_ = list(X.columns)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict affinity scores for a batch of encoded feature rows.

        Clamps predictions to the valid [0, 1] range.

        Args:
            X: Encoded and scaled feature DataFrame.

        Returns:
            NumPy array of predicted affinity scores clipped to [0, 1].

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        scores = self.model_.predict(X)
        return np.clip(scores, 0.0, 1.0)

    def predict_single(self, record: dict) -> dict:
        """
        Run inference for a single customer-product context dictionary.

        Preprocesses the input, runs prediction, and returns a structured
        response including the affinity score and confidence label.

        Args:
            record: Dictionary of customer and product feature values.

        Returns:
            Dictionary with affinity_score, confidence, and input echo.
        """
        from src.preprocessing import prepare_inference_input

        df = prepare_inference_input(record, self.encoders_, self.scaler_)
        df = df[[c for c in self.feature_cols_ if c in df.columns]]
        score = float(self.predict(df)[0])

        confidence = "LOW"
        for label, (lo, hi) in model_config.confidence_levels.items():
            if lo <= score <= hi:
                confidence = label
                break

        return {
            "Customer_Segment": record.get("Customer_Segment", "UNKNOWN"),
            "Product_Category": record.get("Product_Category", "UNKNOWN"),
            "Brand": record.get("Brand", "UNKNOWN"),
            "Location_Region": record.get("Location_Region", "UNKNOWN"),
            "Affinity_Score": round(score, 4),
            "Confidence": confidence,
        }

    def save(self, filepath: str) -> None:
        """
        Serialise the fitted model, encoders, and scaler to a pickle file.

        Args:
            filepath: Destination path for the model artifact.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model_,
                "encoders": self.encoders_,
                "scaler": self.scaler_,
                "feature_cols": self.feature_cols_,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "CustomerAffinityModel":
        """
        Deserialise a fitted model from a pickle file.

        Args:
            filepath: Path to the saved model artifact.

        Returns:
            Loaded and ready-to-use CustomerAffinityModel instance.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance.model_ = data["model"]
        instance.encoders_ = data["encoders"]
        instance.scaler_ = data["scaler"]
        instance.feature_cols_ = data["feature_cols"]
        instance.is_fitted_ = True
        return instance
