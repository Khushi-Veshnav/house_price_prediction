"""Flask REST API serving house price predictions.

Run:  python backend/app.py      (default http://127.0.0.1:5000, override with PORT env var)
Features are read from model/metrics.json, so the API follows whatever dataset was trained.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"


def load_artifacts():
    """Load model; if missing or incompatible with the installed scikit-learn, retrain."""
    try:
        return (joblib.load(MODEL_DIR / "model.pkl"),
                joblib.load(MODEL_DIR / "scaler.pkl"),
                json.loads((MODEL_DIR / "metrics.json").read_text()))
    except Exception as exc:  # noqa: BLE001
        print(f"[backend] Could not load model ({exc!r}). Retraining...", file=sys.stderr)
        subprocess.check_call([sys.executable, str(ROOT / "train_model.py")])
        return (joblib.load(MODEL_DIR / "model.pkl"),
                joblib.load(MODEL_DIR / "scaler.pkl"),
                json.loads((MODEL_DIR / "metrics.json").read_text()))


model, scaler, meta = load_artifacts()
FEATURES = meta["features"]

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/schema")
def schema():
    return jsonify(features=FEATURES, ranges=meta.get("ranges", {}), defaults=meta.get("defaults", {}))


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        values = [float(data[f]) for f in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify(error=f"Provide numeric values for: {', '.join(FEATURES)}"), 400
    if any(not np.isfinite(v) for v in values):
        return jsonify(error="Values must be finite numbers."), 400
    for f, v in zip(FEATURES, values):
        lo, hi = meta.get("ranges", {}).get(f, [-np.inf, np.inf])
        span = hi - lo
        if v < lo - span or v > hi + span:  # allow some extrapolation, reject nonsense
            return jsonify(error=f"'{f}' is far outside the training range [{lo:g}, {hi:g}]."), 400

    X = scaler.transform(np.array([values]))  # same feature order as training
    price = float(model.predict(X)[0])
    return jsonify(**dict(zip(FEATURES, values)), predicted_price=round(max(price, 0.0), 2))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
