"""
Module 2 - Analytics Pipeline And reload the saved pipeline and prove it works

"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
pipeline = joblib.load(BASE / "best_pipeline.joblib")
meta = json.loads((BASE / "results" / "best_model_meta.json").read_text())

print(f"Loaded: {meta['best_model']}")
print("Pipeline steps:", [name for name, _ in pipeline.steps])
print("Preprocessing steps:", [name for name, _, _ in pipeline.named_steps["prep"].transformers_])
print("Final estimator:", type(pipeline.named_steps["clf"]).__name__)
assert hasattr(pipeline.named_steps["prep"], "transformers_"), "preprocessing is not fitted"

# ---- 1. RAW hold-out rows: must reproduce the recorded predictions -----------------
holdout = pd.read_csv(BASE / "results" / "holdout_test_raw.csv")
X_raw, y_true = holdout[meta["features"]], holdout["survived"]  # raw: NaNs, strings, unscaled
pred = pipeline.predict(X_raw)
acc = float((pred == y_true).mean())
print(f"\nHold-out accuracy after reload: {acc:.4f} (recorded at training time: {meta['test_accuracy']:.4f})")
assert np.array_equal(pred, np.array(meta["test_predictions"])), "predictions differ after reload!"
assert np.isclose(acc, meta["test_accuracy"])
print("OK - identical predictions to the in-memory pipeline on raw input.")

# ---- 2. Brand-new raw passengers ---------------------------------------------------
new_passengers = pd.DataFrame([
    {"pclass": 1, "age": 29.0, "sibsp": 0, "parch": 0, "fare": 211.3, "sex": "female", "embarked": "S"},
    {"pclass": 3, "age": 35.0, "sibsp": 0, "parch": 0, "fare": 7.9, "sex": "male", "embarked": "S"},
    {"pclass": 2, "age": np.nan, "sibsp": 1, "parch": 2, "fare": 26.0, "sex": "female", "embarked": "C"},  # missing age
    {"pclass": 3, "age": 4.0, "sibsp": 1, "parch": 1, "fare": 16.7, "sex": "male", "embarked": "Q"},
])
proba = pipeline.predict_proba(new_passengers)[:, 1]
out = new_passengers.assign(pred_survived=pipeline.predict(new_passengers), p_survive=proba.round(3))
print("\nPredictions on new raw passengers:")
print(out.to_string(index=False))
assert len(out) == len(new_passengers) and ((proba >= 0) & (proba <= 1)).all()
print("\nReload check PASSED.")
