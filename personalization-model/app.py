"""
Flask inference server for local and containerised deployment.

Exposes /ping and /invocations endpoints matching the SageMaker real-time
inference API contract, allowing the same container image to be tested
locally before deploying to SageMaker.
"""

import os
import sys
import json
from flask import Flask, request, Response

sys.path.insert(0, os.path.dirname(__file__))

from endpoints.inference.inference import model_fn, input_fn, predict_fn, output_fn

app = Flask(__name__)

MODEL_DIR = os.getenv("SM_MODEL_DIR", "./model")
_model = None


def _get_model():
    """
    Lazily load the model on first request and cache it.

    Returns:
        Loaded CustomerAffinityModel instance.
    """
    global _model
    if _model is None:
        _model = model_fn(MODEL_DIR)
    return _model


@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check endpoint required by the SageMaker container contract.

    Returns HTTP 200 when the model is loaded and ready.
    """
    try:
        model = _get_model()
        if model.is_fitted_:
            return Response(response=json.dumps({"status": "healthy"}), status=200, mimetype="application/json")
    except Exception as error:
        return Response(response=json.dumps({"status": "unhealthy", "error": str(error)}), status=500, mimetype="application/json")
    return Response(response=json.dumps({"status": "not ready"}), status=500, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Inference endpoint matching the SageMaker real-time inference contract.

    Accepts JSON body with customer and product context fields, returns
    affinity score and confidence label.
    """
    content_type = request.content_type or "application/json"
    try:
        input_data = input_fn(request.data.decode("utf-8"), content_type)
        model = _get_model()
        prediction = predict_fn(input_data, model)
        result = output_fn(prediction)
        return Response(response=result, status=200, mimetype="application/json")
    except ValueError as error:
        return Response(response=json.dumps({"error": str(error)}), status=400, mimetype="application/json")
    except Exception as error:
        return Response(response=json.dumps({"error": str(error)}), status=500, mimetype="application/json")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Starting CustomerAffinity inference server on port {port}")
    app.run(host="0.0.0.0", port=port)
