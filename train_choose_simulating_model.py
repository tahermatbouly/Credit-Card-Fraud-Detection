"""
train_choose_simulating_model.py

Purpose:
    1) Load processed PaySim data
    2) Train multiple fraud detection models
    3) Choose the best model using ROC-AUC
    4) Save the best model + metadata
    5) Save the exact feature list used during training

Run from the project root:
    python train_choose_simulating_model.py

Expected data path:
    data/processed/processed_paysim.csv

Outputs:
    output/models/best_model.pkl
    output/models/best_model_meta.json
    output/models/feature_names.json
    output/models/model_results.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


DATA_PATH = Path("data/processed/processed_paysim.csv")
OUTPUT_DIR = Path("output/models")
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COL = "isFraud"
DROP_COLS = ["isFraud", "isFlaggedFraud"]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed PaySim data and split into X/y."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at: {data_path}\n"
            "Make sure you are running the script from the project root."
        )

    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset must contain target column: {TARGET_COL}")

    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET_COL].astype(int)

    # Keep only numeric columns because sklearn models require numeric input.
    # The processed PaySim file should already be numeric / one-hot encoded.
    X = X.select_dtypes(include=[np.number]).copy()

    if X.empty:
        raise ValueError("No numeric feature columns found after preprocessing.")

    return X, y


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def build_models() -> Dict[str, Any]:
    """Create candidate models.

    SMOTE is applied inside each pipeline on the training fold only.
    Logistic Regression gets StandardScaler.
    Tree models do not need scaling.
    """
    models: Dict[str, Any] = {
        "logistic_regression": ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        print("[WARN] xgboost is not installed. Skipping XGBoost model.")

    return models


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def predict_proba_safe(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return fraud probability from a fitted classifier / pipeline."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must support predict_proba().")
    return model.predict_proba(X)[:, 1]


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.30) -> Dict[str, Any]:
    """Evaluate model using fraud detection metrics."""
    probs = predict_proba_safe(model, X_test)
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])

    return {
        "roc_auc": float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            preds,
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            zero_division=0,
        ),
    }


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def save_artifacts(best_name: str, best_model: Any, feature_names: list[str], results: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "best_model.pkl"
    meta_path = OUTPUT_DIR / "best_model_meta.json"
    feature_path = OUTPUT_DIR / "feature_names.json"
    results_path = OUTPUT_DIR / "model_results.csv"

    joblib.dump(best_model, model_path)

    meta = {
        "best_model_name": best_name,
        "selection_metric": "roc_auc",
        "roc_auc": results[best_name]["roc_auc"],
        "threshold_used_for_report": 0.30,
        "target_column": TARGET_COL,
        "dropped_columns": DROP_COLS,
        "data_path": str(DATA_PATH),
        "random_state": RANDOM_STATE,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "model": name,
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
    pd.DataFrame(rows).sort_values("roc_auc", ascending=False).to_csv(results_path, index=False)

    print("\nSaved artifacts:")
    print(f"- {model_path}")
    print(f"- {meta_path}")
    print(f"- {feature_path}")
    print(f"- {results_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    print(f"Features shape: {X.shape}")
    print("Class distribution:")
    print(y.value_counts(normalize=True).rename("ratio"))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    results: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}

    for name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training model: {name}")
        print("=" * 70)

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, threshold=0.30)

        results[name] = metrics
        fitted_models[name] = model

        print(f"ROC-AUC   : {metrics['roc_auc']:.6f}")
        print(f"Accuracy  : {metrics['accuracy']:.6f}")
        print(f"F1        : {metrics['f1']:.6f}")
        print(f"Precision : {metrics['precision']:.6f}")
        print(f"Recall    : {metrics['recall']:.6f}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(np.array(metrics["confusion_matrix"]))
        print("Classification report:")
        print(metrics["classification_report"])

    best_name = max(results, key=lambda model_name: results[model_name]["roc_auc"])
    best_model = fitted_models[best_name]

    print("\n" + "#" * 70)
    print(f"Best model selected by ROC-AUC: {best_name}")
    print(f"Best ROC-AUC: {results[best_name]['roc_auc']:.6f}")
    print("#" * 70)

    save_artifacts(
        best_name=best_name,
        best_model=best_model,
        feature_names=list(X.columns),
        results=results,
    )


if __name__ == "__main__":
    main()
