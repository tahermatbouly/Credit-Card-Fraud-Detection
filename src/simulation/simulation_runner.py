from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.descision.decision_engine import FraudDecisionEngine
from src.explain.explainer import FraudExplainer
from src.simulation.attack_simulator import FraudAttackSimulator


def _predict_proba(
    model: Any,
    X: pd.DataFrame,
    scaler: Any | None,
) -> np.ndarray:
    if scaler is not None:
        Xv = scaler.transform(X.values)
        return model.predict_proba(Xv)[:, 1]
    return model.predict_proba(X)[:, 1]


def _counts_from_cm(cm: np.ndarray) -> dict[str, int]:
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def _evaluate_attack_batch(
    attack_name: str,
    X_attack: pd.DataFrame,
    y_true: pd.Series,
    probs: np.ndarray,
    engine: FraudDecisionEngine,
    explainer: FraudExplainer,
    model: Any,
    feature_names: list[str],
    explain_budget: list[int],
) -> dict[str, Any]:
    y_true_arr = y_true.astype(int).values
    y_pred = np.array([engine.fraud_prediction_label(float(p)) for p in probs])

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    counts = _counts_from_cm(cm)

    row_details: list[dict[str, Any]] = []
    for i, p in enumerate(probs):
        decision = engine.decide(float(p))
        explanation = None
        if decision == "BLOCK" and explain_budget[0] > 0:
            explanation = explainer.explain_tree_model(
                model,
                feature_names,
                X_attack.iloc[i].values,
            )
            explain_budget[0] -= 1

        row_details.append(
            {
                "attack_type": attack_name,
                "probability": float(p),
                "decision": decision,
                "true_fraud": int(y_true_arr[i]),
                "pred_fraud": int(y_pred[i]),
                "explanation": explanation,
            }
        )

    return {
        "attack_name": attack_name,
        "confusion_matrix": cm,
        "counts": counts,
        "y_true": y_true_arr,
        "y_pred": y_pred,
        "probabilities": probs,
        "rows": row_details,
    }


def run_simulation(
    model: Any,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    feature_names: list[str],
    *,
    threshold: float = 0.3,
    flood_repeats: int = 40,
    scaler: Any | None = None,
    max_explanations: int = 15,
) -> dict[str, Any]:
    """
    Flood the system with attack scenarios, score each transaction, then compute
    TP/TN/FP/FN and confusion matrices per scenario and overall.
    """
    engine = FraudDecisionEngine(threshold=threshold)
    explainer = FraudExplainer()

    simulator = FraudAttackSimulator(X_sample, y_sample)
    suite = simulator.build_attack_suite(flood_repeats=flood_repeats)

    per_attack: list[dict[str, Any]] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    flat_results: list[dict[str, Any]] = []
    explain_budget = [max(0, int(max_explanations))]

    for attack_name, X_attack, y_attack in suite:
        probs = _predict_proba(model, X_attack, scaler)
        batch_eval = _evaluate_attack_batch(
            attack_name,
            X_attack,
            y_attack,
            probs,
            engine,
            explainer,
            model,
            feature_names,
            explain_budget,
        )
        per_attack.append(batch_eval)
        all_y_true.extend(batch_eval["y_true"].tolist())
        all_y_pred.extend(batch_eval["y_pred"].tolist())
        flat_results.extend(batch_eval["rows"])

    overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    overall_counts = _counts_from_cm(overall_cm)

    return {
        "threshold": threshold,
        "per_attack": per_attack,
        "overall": {
            "confusion_matrix": overall_cm,
            "counts": overall_counts,
            "total_transactions": len(all_y_true),
        },
        "flat_results": flat_results,
    }


def print_evaluation_report(summary: dict[str, Any], preview_rows: int = 10) -> None:
    print("\n================ ATTACK SIMULATION REPORT ================\n")
    print(f"Decision threshold (fraud if prob >=): {summary['threshold']}")

    for block in summary["per_attack"]:
        name = block["attack_name"]
        cm = block["confusion_matrix"]
        c = block["counts"]
        print(f"\n--- Scenario: {name} ({len(block['y_true'])} tx) ---")
        print(f"TP={c['TP']}  TN={c['TN']}  FP={c['FP']}  FN={c['FN']}")
        print("Confusion matrix [rows=actual 0,1 | cols=predicted 0,1]:")
        print(cm)

    oc = summary["overall"]["counts"]
    ocm = summary["overall"]["confusion_matrix"]
    print("\n--- OVERALL (all scenarios combined) ---")
    print(f"Total transactions scored: {summary['overall']['total_transactions']}")
    print(f"TP={oc['TP']}  TN={oc['TN']}  FP={oc['FP']}  FN={oc['FN']}")
    print("Confusion matrix:")
    print(ocm)

    print(f"\n--- Sample blocked transactions (first {preview_rows}) ---")
    shown = 0
    for row in summary["flat_results"]:
        if row["decision"] != "BLOCK":
            continue
        print("\n------------------")
        print("Scenario:", row["attack_type"])
        print("Probability:", round(row["probability"], 4))
        print("Decision:", row["decision"])
        print("True fraud:", row["true_fraud"], "| Pred fraud:", row["pred_fraud"])
        print("Explanation:", row["explanation"])
        shown += 1
        if shown >= preview_rows:
            break
