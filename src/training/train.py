from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.data.load_data import load_processed_data
from src.features.split_features import split_features
from src.evaluation.evaluate_model import show_final_visualization
from src.models.logistic_regression import run_model as lr_model
from src.models.random_forest import run_model as rf_model
from src.models.xg_boost import run_model as xgb_model


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_model_name(name: str) -> str:
    mapping = {
        "Random Forest": "random_forest",
        "Logistic Regression": "logistic_regression",
        "XGBoost": "xgboost",
        "xg_boost": "xgboost",
    }
    return mapping.get(name, name)


def main() -> None:
    df = load_processed_data()
    X, y = split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_outputs = [
        lr_model(X_train, X_test, y_train, y_test),
        rf_model(X_train, X_test, y_train, y_test),
        xgb_model(X_train, X_test, y_train, y_test),
    ]

    results: list[dict] = []
    cms = []
    roc_data = []

    for output in model_outputs:
        results.append(
            {
                "model_name": output["model_name"],
                "model": output["model"],
                "scaler": output.get("scaler"),
                "accuracy": output["accuracy"],
                "f1": output["f1"],
                "roc_auc": output["roc_auc"],
            }
        )
        cms.append((output["model_name"], output["confusion_matrix"]))
        fpr, tpr, auc = output["roc"]
        roc_data.append((output["model_name"], (fpr, tpr, auc)))

    print("\n================ MODEL RESULTS ================\n")
    for r in results:
        print(f"Model: {r['model_name']}")
        print(f"Accuracy: {r['accuracy']:.4f}")
        print(f"F1: {r['f1']:.4f}")
        print(f"ROC-AUC: {r['roc_auc']:.4f}")
        print("--------------------------------")

    best = max(results, key=lambda x: x["roc_auc"])

    print("\n================ BEST MODEL ================")
    print(f"Model: {best['model_name']}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")

    show_final_visualization(results, cms, roc_data)

    # Save best model (path relative to project root so cwd does not matter)
    out_dir = _project_root() / "output" / "models"
    os.makedirs(out_dir, exist_ok=True)

    model_path = out_dir / "best_model.pkl"
    joblib.dump(best["model"], model_path)

    meta_path = out_dir / "best_model_meta.json"
    meta = {
        "model_name": best["model_name"],
        "model_name_slug": normalize_model_name(best["model_name"]),
        "roc_auc": float(best["roc_auc"]),
        "has_scaler": best.get("scaler") is not None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if best.get("scaler") is not None:
        scaler_path = out_dir / "best_model_scaler.pkl"
        joblib.dump(best["scaler"], scaler_path)
        print(f"\n💾 Scaler saved at: {scaler_path}")

    print(f"\n💾 Best model saved at: {model_path}")
    print(f"💾 Metadata saved at: {meta_path}")


if __name__ == "__main__":
    main()
