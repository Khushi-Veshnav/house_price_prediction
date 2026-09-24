"""Convenience entry point: generate data (if missing) and train the model."""
import subprocess, sys
from pathlib import Path

root = Path(__file__).parent
if not (root / "data" / "house_price.csv").exists():
    subprocess.check_call([sys.executable, str(root / "data" / "generate_dataset.py")])
subprocess.check_call([sys.executable, str(root / "train_model.py")])
