"""Train, evaluate and save the house price model.

Usage:  python train_model.py
Outputs: model/model.pkl, model/scaler.pkl, model/metrics.json, report_images/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
# Uses the real dataset if present, otherwise the synthetic one.
_REAL = ROOT / "data" / "house_price_regression_dataset.csv"
DATA = _REAL if _REAL.exists() else ROOT / "data" / "house_price.csv"
MODEL_DIR = ROOT / "model"
IMG_DIR = ROOT / "report_images"
TARGET = "House_Price" if DATA == _REAL else "price"
FEATURES = None  # auto: every column except TARGET


def evaluate(model, X_tr, y_tr, X_te, y_te):
    pred = model.predict(X_te)
    return {
        "MAE": float(mean_absolute_error(y_te, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "R2": float(r2_score(y_te, pred)),
        "CV_R2": float(cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2").mean()),
    }


def main():
    if not DATA.exists():
        raise SystemExit("Dataset missing. Put house_price_regression_dataset.csv in data/ or run: python data/generate_dataset.py")
    MODEL_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA).dropna()
    features = [c for c in df.columns if c != TARGET]
    X, y = df[features], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_tr.values)  # fit on train only -> no leakage
    Xs_tr, Xs_te = scaler.transform(X_tr.values), scaler.transform(X_te.values)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso (alpha=1.0)": Lasso(alpha=1.0),
    }
    results = {}
    for name, m in candidates.items():
        m.fit(Xs_tr, y_tr)
        results[name] = evaluate(m, Xs_tr, y_tr, Xs_te, y_te)
        print(f"{name:20s}", {k: round(v, 3) for k, v in results[name].items()})

    # Linear Regression is the deployed model (regularisation adds nothing with 2 features)
    final = candidates["Linear Regression"]
    joblib.dump(final, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    meta = {
        "features": features,
        "target": TARGET,
        "dataset": DATA.name,
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in features},
        "defaults": {c: float(df[c].median()) for c in features},
        "model": "Linear Regression",
        "n_rows": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "coefficients": dict(zip(features, map(float, final.coef_))),
        "intercept": float(final.intercept_),
        "results": results,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(meta, indent=2))

    # Dependency-free parameters used by the Vercel serverless API (api/index.py).
    # A linear model is just: intercept + sum(coef_i * (x_i - mean_i) / scale_i)
    params = {
        "features": features,
        "target": TARGET,
        "mean": list(map(float, scaler.mean_)),
        "scale": list(map(float, scaler.scale_)),
        "coef": list(map(float, final.coef_)),
        "intercept": float(final.intercept_),
        "ranges": meta["ranges"],
        "defaults": meta["defaults"],
        "model": meta["model"],
        "metrics": results["Linear Regression"],
        "n_rows": meta["n_rows"],
    }
    (MODEL_DIR / "params.json").write_text(json.dumps(params, indent=2))

    # ---- charts for the report ----
    pred = final.predict(Xs_te)
    plt.figure(figsize=(6, 4.5))
    plt.scatter(y_te, pred, alpha=0.6)
    lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    plt.plot(lim, lim, "r--", label="Perfect prediction")
    plt.xlabel("Actual price"); plt.ylabel("Predicted price")
    plt.title("Actual vs Predicted (test set)"); plt.legend(); plt.tight_layout()
    plt.savefig(IMG_DIR / "actual_vs_predicted.png", dpi=150); plt.close()

    size_col = features[0]
    plt.figure(figsize=(6, 4.5))
    plt.scatter(df[size_col], df[TARGET], c=df[features[1]], cmap="viridis", alpha=0.6)
    plt.colorbar(label=features[1]); plt.xlabel(size_col); plt.ylabel(TARGET)
    plt.title(f"{TARGET} vs {size_col}"); plt.tight_layout()
    plt.savefig(IMG_DIR / "price_vs_area.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4.5))
    resid = y_te - pred
    plt.hist(resid, bins=25, edgecolor="white")
    plt.xlabel("Residual (actual - predicted)"); plt.ylabel("Count")
    plt.title("Residual distribution"); plt.tight_layout()
    plt.savefig(IMG_DIR / "residuals.png", dpi=150); plt.close()

    names = list(results)
    plt.figure(figsize=(6, 4.5))
    plt.bar(names, [results[n]["R2"] for n in names], color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylim(min(0.0, min(results[n]["R2"] for n in names)) - 0.02, 1.0)
    plt.ylabel("Test R²"); plt.title("Model comparison"); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(IMG_DIR / "model_comparison.png", dpi=150); plt.close()
    print("Saved model, scaler, metrics and charts.")


if __name__ == "__main__":
    main()
