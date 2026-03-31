"""
AWS Lambda handler for real-time customer affinity score computation.

This function is triggered by customer activity events (e.g., product page
views, cart additions, or login events) forwarded via API Gateway or SQS.
It invokes the customer affinity SageMaker endpoint and persists the resulting
affinity score to the configured database for downstream personalisation use.
"""

import os
import json
import logging
import boto3
import psycopg2
from datetime import datetime
from typing import Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AFFINITY_ENDPOINT = os.environ.get("AFFINITY_ENDPOINT_NAME", "customer-affinity-endpoint")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_sagemaker_client():
    """
    Create and return a boto3 SageMaker Runtime client.

    Returns:
        Configured SageMaker Runtime boto3 client.
    """
    return boto3.client("sagemaker-runtime", region_name=AWS_REGION)


def invoke_affinity_endpoint(client, payload: dict) -> Optional[dict]:
    """
    Invoke the customer affinity SageMaker endpoint.

    Args:
        client: SageMaker Runtime boto3 client.
        payload: Customer and product context dictionary.

    Returns:
        Parsed affinity score response, or None on failure.
    """
    try:
        response = client.invoke_endpoint(
            EndpointName=AFFINITY_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)
    except Exception as error:
        logger.error(f"Affinity endpoint invocation failed: {error}")
        return None


def store_affinity_score(customer_id: str, product_context: dict, result: dict) -> bool:
    """
    Persist a computed affinity score to the PostgreSQL database.

    Args:
        customer_id: Unique identifier for the customer.
        product_context: Dictionary of product context fields from the event.
        result: Affinity endpoint response containing Affinity_Score and Confidence.

    Returns:
        True if the record was inserted successfully, False otherwise.
    """
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is not set")
        return False

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO customer_affinity_scores
                (customer_id, product_category, brand, affinity_score, confidence, computed_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                customer_id,
                result.get("Product_Category", "UNKNOWN"),
                result.get("Brand", "UNKNOWN"),
                result.get("Affinity_Score", 0.0),
                result.get("Confidence", "LOW"),
                datetime.utcnow(),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as error:
        logger.error(f"Database insert failed: {error}")
        return False


def lambda_handler(event, context):
    """
    Main Lambda entry point for real-time customer affinity scoring.

    Processes one or more customer activity events, invokes the affinity
    endpoint for each, and stores the resulting scores to the database.

    Expected event format:
    {
        "customer_events": [
            {
                "customer_id": "CUST001",
                "Customer_Segment": "PREMIUM",
                "Age_Group": "25-34",
                "Location_Region": "NORTH",
                "Device_Type": "MOBILE",
                "Product_Category": "Electronics",
                "Product_Sub_Category": "Laptops",
                "Brand": "BrandX",
                "Purchase_Frequency": 12,
                "Avg_Order_Value": 3500.0,
                "Days_Since_Last_Purchase": 14,
                "Browse_Count": 8,
                "Cart_Abandonment_Rate": 0.2,
                "Session_Duration_Mins": 12.5
            }
        ]
    }

    Args:
        event: Lambda event payload with a list of customer activity contexts.
        context: Lambda runtime context object (unused).

    Returns:
        Dictionary with statusCode and a processing summary body.
    """
    logger.info("CustomerAffinity Lambda handler invoked")

    customer_events = event.get("customer_events", [])

    if not customer_events:
        logger.warning("No customer events in payload")
        return {"statusCode": 400, "body": json.dumps({"message": "No customer_events in event"})}

    sm_client = get_sagemaker_client()
    stored_count = 0
    failed_count = 0

    for activity in customer_events:
        customer_id = activity.get("customer_id", "UNKNOWN")
        logger.info(f"Scoring affinity for customer: {customer_id}")

        result = invoke_affinity_endpoint(sm_client, activity)
        if not result:
            logger.warning(f"No affinity result for customer: {customer_id}")
            failed_count += 1
            continue

        success = store_affinity_score(customer_id, activity, result)
        if success:
            logger.info(f"Stored affinity score {result.get('Affinity_Score')} for {customer_id}")
            stored_count += 1
        else:
            logger.error(f"Failed to store affinity for customer: {customer_id}")
            failed_count += 1

    summary = {
        "total_processed": len(customer_events),
        "stored": stored_count,
        "failed": failed_count,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(f"Lambda complete: {summary}")

    return {
        "statusCode": 200 if failed_count == 0 else 207,
        "body": json.dumps(summary),
    }
