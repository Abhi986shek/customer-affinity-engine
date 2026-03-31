"""
Unit tests for the customer affinity preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import clean_data, engineer_features, encode_features, prepare_inference_input


def _make_sample_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Customer_Segment": rng.choice(["PREMIUM", "STANDARD", "BUDGET"], n),
        "Age_Group": rng.choice(["18-24", "25-34", "35-44", "45+"], n),
        "Location_Region": rng.choice(["NORTH", "SOUTH", "EAST", "WEST"], n),
        "Product_Category": rng.choice(["Electronics", "Apparel", "HomeGarden"], n),
        "Product_Sub_Category": rng.choice(["Laptops", "Shirts", "Furniture"], n),
        "Brand": rng.choice(["BrandA", "BrandB", "BrandC"], n),
        "Device_Type": rng.choice(["MOBILE", "DESKTOP", "TABLET"], n),
        "Purchase_Frequency": rng.integers(0, 52, n).astype(float),
        "Avg_Order_Value": rng.uniform(100, 10000, n),
        "Days_Since_Last_Purchase": rng.integers(0, 365, n).astype(float),
        "Browse_Count": rng.integers(0, 50, n).astype(float),
        "Cart_Abandonment_Rate": rng.uniform(0, 1, n),
        "Session_Duration_Mins": rng.uniform(1, 60, n),
        "Affinity_Score": rng.uniform(0, 1, n),
    })


class TestCleanData:
    def test_drops_missing_target(self):
        df = _make_sample_df(20)
        df.loc[:4, "Affinity_Score"] = np.nan
        cleaned = clean_data(df)
        assert cleaned["Affinity_Score"].isna().sum() == 0

    def test_clips_affinity_score_to_1(self):
        df = _make_sample_df(10)
        df["Affinity_Score"] = 1.5
        cleaned = clean_data(df)
        assert cleaned["Affinity_Score"].max() <= 1.0

    def test_clips_cart_abandonment_rate(self):
        df = _make_sample_df(10)
        df["Cart_Abandonment_Rate"] = 1.5
        cleaned = clean_data(df)
        assert cleaned["Cart_Abandonment_Rate"].max() <= 1.0


class TestEngineerFeatures:
    def test_adds_recency_bucket(self):
        df = _make_sample_df(20)
        df = clean_data(df)
        df = engineer_features(df)
        assert "recency_bucket" in df.columns

    def test_adds_engagement_score(self):
        df = _make_sample_df(20)
        df = clean_data(df)
        df = engineer_features(df)
        assert "engagement_score" in df.columns
        assert df["engagement_score"].between(0, 100).all()

    def test_adds_value_tier(self):
        df = _make_sample_df(20)
        df = clean_data(df)
        df = engineer_features(df)
        assert "value_tier" in df.columns


class TestEncodeFeatures:
    def test_encodes_categoricals_to_int(self):
        df = _make_sample_df(30)
        df = clean_data(df)
        df = engineer_features(df)
        encoded, encoders, scaler = encode_features(df, fit=True)
        assert encoded["Customer_Segment"].dtype in [np.int32, np.int64, int]
        assert "Customer_Segment" in encoders
        assert scaler is not None

    def test_inference_encoding_consistent(self):
        df = _make_sample_df(30)
        df = clean_data(df)
        df = engineer_features(df)
        _, encoders, scaler = encode_features(df, fit=True)

        new_df = _make_sample_df(5)
        new_df = clean_data(new_df)
        new_df = engineer_features(new_df)
        encoded_new, _, _ = encode_features(new_df, fit=False, encoders=encoders, scaler=scaler)
        assert encoded_new["Customer_Segment"].dtype in [np.int32, np.int64, int]
