# House Price Prediction

End-to-end ML app: **scikit-learn** model → **Flask** REST API → **Streamlit** UI, plus auto-generated Word/HTML reports.

## Structure
```
backend/app.py          Flask API (/predict, /health)
frontend/app.py         Streamlit UI (calls the API; falls back to local model)
data/generate_dataset.py  Builds data/house_price.csv (synthetic; swap in your own data)
train_model.py          Trains Linear/Ridge/Lasso, saves model/, metrics, charts
house_price_main.py     One-shot: create data (if missing) + train
generate_report.py      Builds the .docx and .html reports
tests_smoke.py          API smoke tests
linear_regression_regularization_lasso_ridge.ipynb   Regularization analysis
```

## Deploy to Vercel
Vercel serves `public/index.html` (static UI) and `api/index.py` (Flask, pure Python, only needs `flask`).
Model weights are read from `model/params.json`, so scikit-learn/Streamlit are **not** installed on Vercel.
```bash
npm i -g vercel
vercel          # preview deploy
vercel --prod   # production
```
Or push to GitHub and import the repo at vercel.com/new (no build settings needed).
Endpoints: `GET /api/health`, `GET /api/schema`, `POST /api/predict`.
After retraining (`python train_model.py`), `model/params.json` is regenerated: commit it and redeploy.
Local dev/report dependencies live in `requirements-dev.txt`. Test locally with `vercel dev`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python house_price_main.py        # creates data + trains model
```

## Run
Terminal 1 – API:
```bash
python backend/app.py
```
Terminal 2 – UI:
```bash
streamlit run frontend/app.py
```
Open http://localhost:8501.

API example:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
     -d '{"Square_Footage":2000,"Num_Bedrooms":3,"Num_Bathrooms":2,"Year_Built":2000,"Lot_Size":2.5,"Garage_Size":1,"Neighborhood_Quality":7}'
curl http://127.0.0.1:5000/schema     # lists the expected fields
```

## Reports
```bash
python generate_report.py
```
Save Streamlit screenshots as `report_images/ui_*.png` before running to embed them.

## Data
`data/house_price_regression_dataset.csv` (1000 rows, 7 features, target `House_Price`) is used if present;
otherwise the synthetic `house_price.csv` is used. Features are auto-detected (all columns except the target)
and stored in `model/metrics.json`, so the API and UI adapt automatically. Re-run `python train_model.py` after changing data.

## Troubleshooting
- **Port 5000 in use (macOS AirPlay):** `PORT=5001 python backend/app.py` and `API_URL=http://127.0.0.1:5001 streamlit run frontend/app.py`
- **Pickle / scikit-learn version errors:** the API and UI now retrain automatically; or run `python train_model.py`.
- **UI shows "local model (API unreachable)":** start the API first, or check `API_URL`.
- Run commands from the project root.

## Notes
- The scaler is fit on the training split only, and the API uses `transform` with the same feature order.
- Regularization gives near-identical results with two features, so plain Linear Regression is deployed.
- Pickled models depend on the scikit-learn version; retrain if you upgrade.
