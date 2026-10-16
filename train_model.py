import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance_demo.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)
target = "final_exam_score"
features = [
    "study_hours_per_day", "attendance_percentage", "previous_exam_score",
    "assignment_score", "sleep_hours_per_day", "class_participation"
]
required = features + [target]
missing_columns = [c for c in required if c not in data.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")
data = data[required].copy()
for col in required:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.dropna(subset=[target])
if len(data) < 30:
    raise ValueError("At least 30 usable rows are recommended for this demonstration.")

X, y = data[features], data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

candidates = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])
}

comparison = []
trained = {}
for name, estimator in candidates.items():
    estimator.fit(X_train, y_train)
    pred = np.clip(estimator.predict(X_test), 0, 100)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    comparison.append({"Model": name, "MAE": round(mae, 3),
                       "RMSE": round(rmse, 3), "R2": round(r2, 4)})
    trained[name] = (estimator, {"mae": float(mae), "rmse": rmse, "r2": float(r2)})

# Select model with lowest MAE on the same held-out test set for demonstration.
# For rigorous model selection, use cross-validation on training data and evaluate
# the chosen model once on a final untouched test set.
best_name = min(comparison, key=lambda item: item["MAE"])["Model"]
best_model, metrics = trained[best_name]
bundle = {
    "model": best_model,
    "features": features,
    "target": target,
    "selected_model": best_name,
    "metrics": metrics,
    "comparison": comparison,
    "dataset_note": "Synthetic demonstration data; not real student records."
}
joblib.dump(bundle, os.path.join(MODEL_DIR, "student_performance_model.joblib"))
pd.DataFrame(comparison).to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)

with open(os.path.join(REPORT_DIR, "evaluation_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "selected_model": best_name,
        "metrics": metrics,
        "comparison": comparison,
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": features,
        "target": target,
        "note": "Synthetic demonstration dataset; metrics are not evidence of real-world performance."
    }, f, indent=2)

print("Training completed.")
print(f"Selected model: {best_name}")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
print(pd.DataFrame(comparison).to_string(index=False))
print(f"Saved model: {os.path.join(MODEL_DIR, 'student_performance_model.joblib')}")
