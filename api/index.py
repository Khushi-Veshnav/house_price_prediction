"""Vercel serverless Flask API (pure Python, no numpy/scikit-learn needed).

Endpoints (all under /api):  GET /api/health   GET /api/schema   POST /api/predict
Model parameters come from model/params.json, written by train_model.py.
"""
import json
import math
from pathlib import Path

from flask import Flask, jsonify, request

PARAMS = json.loads((Path(__file__).resolve().parent.parent / "model" / "params.json").read_text())
FEATURES = PARAMS["features"]

app = Flask(__name__)


def _predict(values):
    z = sum(c * (v - m) / s for v, m, s, c in zip(values, PARAMS["mean"], PARAMS["scale"], PARAMS["coef"]))
    return max(PARAMS["intercept"] + z, 0.0)


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/schema")
def schema():
    return jsonify(features=FEATURES, ranges=PARAMS["ranges"], defaults=PARAMS["defaults"],
                   model=PARAMS["model"], metrics=PARAMS["metrics"], n_rows=PARAMS["n_rows"])


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        values = [float(data[f]) for f in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify(error=f"Provide numeric values for: {', '.join(FEATURES)}"), 400
    if any(not math.isfinite(v) for v in values):
        return jsonify(error="Values must be finite numbers."), 400
    for f, v in zip(FEATURES, values):
        lo, hi = PARAMS["ranges"][f]
        span = hi - lo
        if v < lo - span or v > hi + span:
            return jsonify(error=f"'{f}' is far outside the training range [{lo:g}, {hi:g}]."), 400
    return jsonify(**dict(zip(FEATURES, values)), predicted_price=round(_predict(values), 2))
