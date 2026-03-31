"""
SageMaker inference handler for the customer affinity scoring model.

Implements the four functions required by the SageMaker SKLearn inference
contract: model_fn, input_fn, predict_fn, output_fn. Accepts JSON with
customer and product context, returns an affinity score and confidence label.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model import CustomerAffinityModel


def model_fn(model_dir: str) -> CustomerAffinityModel:
    """
    Load the trained affinity model from the model directory.

    Called once by SageMaker when the endpoint initialises.

    Args:
        model_dir: Directory where SageMaker extracted the model artifact.

    Returns:
        Loaded CustomerAffinityModel instance ready for prediction.
    """
    model_path = os.path.join(model_dir, "affinity_model.pkl")
    print(f"Loading model from: {model_path}")
    model = CustomerAffinityModel.load(model_path)
    print("Customer affinity model loaded successfully")
    return model


def input_fn(request_body: str, content_type: str = "application/json") -> dict:
    """
    Deserialise the incoming inference request body.

    Expected JSON keys:
    - Customer_Segment, Age_Group, Location_Region, Device_Type
    - Product_Category, Product_Sub_Category, Brand
    - Purchase_Frequency, Avg_Order_Value, Days_Since_Last_Purchase
    - Browse_Count, Cart_Abandonment_Rate, Session_Duration_Mins

    Args:
        request_body: Raw request body string.
        content_type: MIME type of the request (must be application/json).

    Returns:
        Parsed customer-product context dictionary.

    Raises:
        ValueError: If content type is not application/json.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Expected application/json.")
    return json.loads(request_body)


def predict_fn(input_data: dict, model: CustomerAffinityModel) -> dict:
    """
    Run the affinity score prediction for the provided customer-product context.

    Args:
        input_data: Parsed context dictionary from input_fn.
        model: Loaded CustomerAffinityModel from model_fn.

    Returns:
        Dictionary containing Affinity_Score (0–1), Confidence label,
        and echoed key context fields.
    """
    return model.predict_single(input_data)


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """
    Serialise the prediction result to JSON for the endpoint response.

    Args:
        prediction: Prediction dictionary from predict_fn.
        accept: Expected response MIME type.

    Returns:
        JSON string of the prediction.

    Raises:
        ValueError: If accept type is not application/json.
    """
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}. Expected application/json.")
    return json.dumps(prediction)
