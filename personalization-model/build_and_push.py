"""
Docker image build and ECR push script for the customer affinity engine.

Reads all configuration from environment variables. Authenticates with ECR,
builds the Docker image, and pushes it to the registry.
"""

import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

ECR_REPO_URI = os.getenv("ECR_REPO_URI", "your-account.dkr.ecr.us-east-1.amazonaws.com/customer-affinity-engine")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
IMAGE_TAG = os.getenv("IMAGE_TAG", "latest")


def run(command: str) -> None:
    """
    Execute a shell command and stream output.

    Args:
        command: Shell command string to execute.
    """
    print(f"$ {command}")
    subprocess.run(command, shell=True, check=True)


def main() -> None:
    """
    Authenticate with ECR, build the Docker image, and push it.
    """
    image_uri = f"{ECR_REPO_URI}:{IMAGE_TAG}"
    account_id = ECR_REPO_URI.split(".")[0]

    print(f"Building and pushing: {image_uri}")
    run(f"aws ecr get-login-password --region {AWS_REGION} | "
        f"docker login --username AWS --password-stdin "
        f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com")
    run(f"docker build -t {image_uri} .")
    run(f"docker push {image_uri}")
    print(f"\nSuccessfully pushed: {image_uri}")


if __name__ == "__main__":
    main()
