"""Generate a synthetic house price dataset (data/house_price.csv).

Replace this with your own real dataset if you have one. Required columns: area, rooms, price.
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

area = rng.normal(1800, 550, N).clip(500, 4500).round()
rooms = np.clip(np.round(area / 600 + rng.normal(0, 0.7, N)), 1, 8).astype(int)
price = 25_000 + 110 * area + 8_000 * rooms + rng.normal(0, 22_000, N)

df = pd.DataFrame({"area": area.astype(int), "rooms": rooms, "price": price.round(-2).astype(int)})
out = Path(__file__).parent / "house_price.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
