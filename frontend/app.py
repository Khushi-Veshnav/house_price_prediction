"""Streamlit UI. Calls the Flask API; falls back to the local model if the API is down.

Run:  streamlit run frontend/app.py      (set API_URL if the API is not on 127.0.0.1:5000)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_local():
    """Load model artifacts, retraining if they are missing/incompatible."""
    try:
        return (joblib.load(ROOT / "model" / "model.pkl"), joblib.load(ROOT / "model" / "scaler.pkl"),
                json.loads((ROOT / "model" / "metrics.json").read_text()))
    except Exception:  # noqa: BLE001
        subprocess.check_call([sys.executable, str(ROOT / "train_model.py")])
        return (joblib.load(ROOT / "model" / "model.pkl"), joblib.load(ROOT / "model" / "scaler.pkl"),
                json.loads((ROOT / "model" / "metrics.json").read_text()))


@st.cache_data
def load_data(name: str):
    return pd.read_csv(ROOT / "data" / name)


def predict(values: dict):
    """Return (price, source)."""
    try:
        r = requests.post(f"{API_URL}/predict", json=values, timeout=3)
        r.raise_for_status()
        return r.json()["predicted_price"], "Flask API"
    except requests.RequestException:
        model, scaler, meta = load_local()
        x = np.array([[values[f] for f in meta["features"]]])
        return max(float(model.predict(scaler.transform(x))[0]), 0.0), "local model (API unreachable)"


_, _, meta = load_local()
FEATURES, TARGET = meta["features"], meta.get("target", "price")
df = load_data(meta.get("dataset", "house_price.csv"))

st.title("🏠 House Price Prediction")
st.caption("Linear Regression model served through a Flask API.")

left, right = st.columns([1, 2])
with left:
    st.subheader("Property details")
    values = {}
    for f in FEATURES:
        lo, hi = meta["ranges"][f]
        default = meta["defaults"][f]
        if float(df[f].round().eq(df[f]).all()):  # integer-valued column
            values[f] = st.number_input(f.replace("_", " "), int(lo), int(hi), int(round(default)), step=1)
        else:
            values[f] = st.number_input(f.replace("_", " "), float(lo), float(hi), float(default))
    if st.button("Predict price", type="primary", width="stretch"):
        price, src = predict({k: float(v) for k, v in values.items()})
        st.metric("Estimated price", f"${price:,.0f}")
        st.caption(f"Source: {src}")
        st.session_state["point"] = (float(values[FEATURES[0]]), price)

with right:
    st.subheader("Where this fits in the data")
    color = FEATURES[1] if len(FEATURES) > 1 else None
    fig = px.scatter(df, x=FEATURES[0], y=TARGET, color=color, opacity=0.6)
    if "point" in st.session_state:
        a, p = st.session_state["point"]
        fig.add_scatter(x=[a], y=[p], mode="markers", name="Your prediction",
                        marker=dict(size=16, color="red", symbol="star"))
    st.plotly_chart(fig, width="stretch")

with st.expander("Dataset preview"):
    st.dataframe(df.head(20), width="stretch")
