"""
Model and training configuration for the customer affinity scoring engine.

Defines hyperparameters, feature columns, and SageMaker infrastructure
configuration. All credential and infrastructure values are loaded from
environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class ModelConfig:
    """
    Configuration for the Gradient Boosting customer affinity scoring model.

    The model predicts a continuous affinity score (0–1) representing the
    likelihood that a given customer will engage with a specific product,
    given their behavioral history, segment, and product attributes.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum depth of each decision tree.
        learning_rate: Shrinkage applied to each tree's contribution.
        subsample: Fraction of samples used per boosting round.
        colsample_bytree: Fraction of features used per tree.
        categorical_features: Columns requiring label encoding.
        numerical_features: Columns used as-is after scaling.
        target_column: Name of the affinity score target column.
        confidence_levels: Score bands mapped to label strings.
        endpoint_name: SageMaker endpoint for live inference.
        instance_type: SageMaker instance type for training and inference.
        sagemaker_role: IAM role ARN for SageMaker execution.
    """

    n_estimators: int = int(os.getenv("MODEL_N_ESTIMATORS", "300"))
    max_depth: int = int(os.getenv("MODEL_MAX_DEPTH", "6"))
    learning_rate: float = float(os.getenv("MODEL_LEARNING_RATE", "0.05"))
    subsample: float = float(os.getenv("MODEL_SUBSAMPLE", "0.8"))
    colsample_bytree: float = float(os.getenv("MODEL_COLSAMPLE_BYTREE", "0.8"))

    categorical_features = [
        "Customer_Segment",
        "Age_Group",
        "Location_Region",
        "Product_Category",
        "Product_Sub_Category",
        "Brand",
        "Device_Type",
    ]

    numerical_features = [
        "Purchase_Frequency",
        "Avg_Order_Value",
        "Days_Since_Last_Purchase",
        "Browse_Count",
        "Cart_Abandonment_Rate",
        "Session_Duration_Mins",
    ]

    target_column: str = "Affinity_Score"

    confidence_levels = {
        "HIGH": (0.70, 1.0),
        "MEDIUM": (0.40, 0.70),
        "LOW": (0.0, 0.40),
    }

    endpoint_name: str = os.getenv("SAGEMAKER_ENDPOINT_NAME", "customer-affinity-endpoint")
    instance_type: str = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.xlarge")
    sagemaker_role: str = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::your-account-id:role/SageMakerRole")


model_config = ModelConfig()
