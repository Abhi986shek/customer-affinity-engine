"""
Unit tests for the CustomerAffinityModel.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import CustomerAffinityModel
from src.preprocessing import _make_sample_df 

# Inline helper to avoid import dependency
def _make_xy(n: int = 100):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "Customer_Segment": rng.integers(0, 3, n),
        "Age_Group": rng.integers(0, 4, n),
        "Location_Region": rng.integers(0, 4, n),
        "Product_Category": rng.integers(0, 3, n),
        "Product_Sub_Category": rng.integers(0, 3, n),
        "Brand": rng.integers(0, 3, n),
        "Device_Type": rng.integers(0, 3, n),
        "Purchase_Frequency": rng.uniform(0, 52, n),
        "Avg_Order_Value": rng.uniform(100, 10000, n),
        "Days_Since_Last_Purchase": rng.uniform(0, 365, n),
        "Browse_Count": rng.uniform(0, 50, n),
        "Cart_Abandonment_Rate": rng.uniform(0, 1, n),
        "Session_Duration_Mins": rng.uniform(1, 60, n),
        "recency_bucket": rng.integers(0, 5, n),
        "value_tier": rng.integers(0, 4, n),
        "engagement_score": rng.uniform(0, 50, n),
    })
    y = pd.Series(rng.uniform(0, 1, n))
    return X, y


class TestCustomerAffinityModel:
    def test_fit_sets_is_fitted_flag(self):
        from sklearn.preprocessing import StandardScaler
        model = CustomerAffinityModel()
        X, y = _make_xy()
        scaler = StandardScaler()
        model.fit(X, y, {}, scaler)
        assert model.is_fitted_ is True

    def test_predict_returns_clipped_scores(self):
        from sklearn.preprocessing import StandardScaler
        model = CustomerAffinityModel()
        X, y = _make_xy()
        model.fit(X, y, {}, StandardScaler())
        preds = model.predict(X)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_predict_raises_when_not_fitted(self):
        model = CustomerAffinityModel()
        X, _ = _make_xy(5)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_predict_length_matches_input(self):
        from sklearn.preprocessing import StandardScaler
        model = CustomerAffinityModel()
        X, y = _make_xy(80)
        model.fit(X, y, {}, StandardScaler())
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_save_and_load_round_trip(self):
        from sklearn.preprocessing import StandardScaler
        model = CustomerAffinityModel()
        X, y = _make_xy()
        scaler = StandardScaler()
        model.fit(X, y, {"test": "enc"}, scaler)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "affinity_model.pkl")
            model.save(path)
            loaded = CustomerAffinityModel.load(path)
            assert loaded.is_fitted_ is True
            assert loaded.feature_cols_ == model.feature_cols_
            preds = loaded.predict(X)
            assert len(preds) == len(X)
