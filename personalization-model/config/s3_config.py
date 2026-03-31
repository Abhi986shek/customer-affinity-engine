"""
S3 configuration for the customer affinity scoring engine.

All bucket names, prefixes, and paths are read from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class S3Config:
    """
    Centralized S3 configuration for data input/output paths.

    Attributes:
        bucket_name: Name of the S3 bucket holding customer behavior data.
        data_prefix: Key prefix for raw input data files.
        model_prefix: Key prefix for trained model artifacts.
        region: AWS region where the bucket is hosted.
    """

    bucket_name: str = os.getenv("S3_BUCKET_NAME", "your-s3-bucket-name")
    data_prefix: str = os.getenv("S3_DATA_PREFIX", "data/customer-behavior/")
    model_prefix: str = os.getenv("S3_MODEL_PREFIX", "models/affinity/")
    region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    training_data_key: str = f"{data_prefix}customer_interactions.csv"
    model_artifact_key: str = f"{model_prefix}affinity_model.tar.gz"


s3_config = S3Config()
