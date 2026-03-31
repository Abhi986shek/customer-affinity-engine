"""
SageMaker endpoint deployment script for the customer affinity model.

Deploys a trained model artifact from S3 to a real-time SageMaker endpoint.
All configuration is read from environment variables.
"""

import os
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from dotenv import load_dotenv

load_dotenv()

SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::your-account-id:role/SageMakerRole")
MODEL_ARTIFACT_URI = os.getenv("MODEL_ARTIFACT_S3_URI", "s3://your-bucket/models/affinity/model.tar.gz")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "customer-affinity-endpoint")
INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def deploy_endpoint() -> str:
    """
    Deploy the customer affinity model to a SageMaker real-time endpoint.

    Returns:
        The deployed endpoint name.
    """
    session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))

    model = SKLearnModel(
        model_data=MODEL_ARTIFACT_URI,
        role=SAGEMAKER_ROLE,
        entry_point="endpoints/inference/inference.py",
        framework_version="1.2-1",
        sagemaker_session=session,
    )

    print(f"Deploying to endpoint: {ENDPOINT_NAME} ({INSTANCE_TYPE})")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"Endpoint ready: {predictor.endpoint_name}")
    return predictor.endpoint_name


if __name__ == "__main__":
    deploy_endpoint()
