"""Quick smoke test for the API: python tests_smoke.py"""
import sys
sys.path.insert(0, "backend")
from app import app, FEATURES  # noqa: E402

c = app.test_client()
sample = {f: 1 for f in FEATURES}
schema = c.get("/schema").get_json()
sample = {f: schema["defaults"][f] for f in FEATURES}
ok = c.post("/predict", json=sample)
assert ok.status_code == 200 and ok.get_json()["predicted_price"] > 0, ok.get_json()
bad = dict(sample); bad[FEATURES[0]] = -1e9
assert c.post("/predict", json=bad).status_code == 400
missing = dict(sample); missing.pop(FEATURES[0])
assert c.post("/predict", json=missing).status_code == 400
assert c.post("/predict", data="junk").status_code == 400
assert c.get("/health").get_json() == {"status": "ok"}
print("All API tests passed:", ok.get_json())

# Vercel API (api/index.py) must give the same answer as the sklearn-backed API
sys.path.insert(0, "api")
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("vercel_api", "api/index.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
v = mod.app.test_client().post("/api/predict", json=sample).get_json()["predicted_price"]
assert abs(v - ok.get_json()["predicted_price"]) < 0.02, (v, ok.get_json())
print("Vercel API matches:", v)
