from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from src.descision.decision_engine import FraudDecisionEngine
from src.explain.explainer import FraudExplainer
from src.simulation.attack_simulator import FraudAttackSimulator


# =========================================================
# Prediction Helper
# =========================================================

def _predict_proba(
    model: Any,
    X: pd.DataFrame,
    scaler: Any | None,
) -> np.ndarray:

    if scaler is not None:

        X_scaled = scaler.transform(X.values)

        return model.predict_proba(X_scaled)[:, 1]

    return model.predict_proba(X)[:, 1]


# =========================================================
# Confusion Matrix Metrics
# =========================================================

def _counts_from_cm(cm: np.ndarray) -> dict[str, int]:

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# =========================================================
# Single Attack Batch Evaluation
# =========================================================

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

    y_pred = np.array([
        engine.fraud_prediction_label(float(p))
        for p in probs
    ])

    cm = confusion_matrix(
        y_true_arr,
        y_pred,
        labels=[0, 1]
    )

    counts = _counts_from_cm(cm)

    detailed_rows: list[dict[str, Any]] = []

    # =====================================================
    # Evaluate Every Transaction
    # =====================================================

    for i, p in enumerate(probs):

        probability = float(p)

        decision = engine.decide(probability)

        include_features = explain_budget[0] > 0

        if include_features:
            explain_budget[0] -= 1

        # =================================================
        # Generate Detailed Explanation
        # =================================================

        explanation = explainer.explain_fraud_prediction(
            model=model,
            feature_names=feature_names,
            sample=X_attack.iloc[i].values,

            decision=decision,

            fraud_probability=probability,

            block_threshold=engine.threshold,

            review_threshold=None,

            attack_type=attack_name,

            true_fraud_label=int(y_true_arr[i]),

            pred_fraud_label=int(y_pred[i]),

            include_feature_evidence=include_features,

            top_k=5,
        )

        detailed_rows.append({

            "attack_type": attack_name,

            "probability": probability,

            "decision": decision,

            "true_fraud": int(y_true_arr[i]),

            "pred_fraud": int(y_pred[i]),

            "explanation": explanation,
        })

    # =====================================================
    # Attack Summary
    # =====================================================

    attack_summary = {
        "attack_name": attack_name,

        "attack_description":
            explainer.describe_attack_scenario(attack_name),

        "confusion_matrix": cm,

        "counts": counts,

        "y_true": y_true_arr,

        "y_pred": y_pred,

        "probabilities": probs,

        "rows": detailed_rows,
    }

    return attack_summary


# =========================================================
# Main Simulation Runner
# =========================================================

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
    Runs multiple fraud attack scenarios against the model.

    The system:
    - floods the model with attacks
    - scores every transaction
    - generates explanations
    - computes TP/TN/FP/FN
    - evaluates attack behavior in detail
    """

    engine = FraudDecisionEngine(
        threshold=threshold
    )

    explainer = FraudExplainer()

    simulator = FraudAttackSimulator(
        X_sample,
        y_sample
    )

    # =====================================================
    # Build Attack Suite
    # =====================================================

    suite = simulator.build_attack_suite(
        flood_repeats=flood_repeats
    )

    per_attack_results: list[dict[str, Any]] = []

    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    flat_results: list[dict[str, Any]] = []

    explain_budget = [
        max(0, int(max_explanations))
    ]

    # =====================================================
    # Run Every Attack Scenario
    # =====================================================

    for attack_name, X_attack, y_attack in suite:

        probs = _predict_proba(
            model,
            X_attack,
            scaler
        )

        batch_result = _evaluate_attack_batch(
            attack_name=attack_name,

            X_attack=X_attack,

            y_true=y_attack,

            probs=probs,

            engine=engine,

            explainer=explainer,

            model=model,

            feature_names=feature_names,

            explain_budget=explain_budget,
        )

        per_attack_results.append(batch_result)

        all_y_true.extend(
            batch_result["y_true"].tolist()
        )

        all_y_pred.extend(
            batch_result["y_pred"].tolist()
        )

        flat_results.extend(
            batch_result["rows"]
        )

    # =====================================================
    # Overall Evaluation
    # =====================================================

    overall_cm = confusion_matrix(
        all_y_true,
        all_y_pred,
        labels=[0, 1]
    )

    overall_counts = _counts_from_cm(
        overall_cm
    )

    return {

        "threshold": threshold,

        "per_attack": per_attack_results,

        "overall": {

            "confusion_matrix": overall_cm,

            "counts": overall_counts,

            "total_transactions": len(all_y_true),
        },

        "flat_results": flat_results,
    }


# =========================================================
# Console Evaluation Report
# =========================================================

def print_evaluation_report(
    summary: dict[str, Any],
    preview_rows: int = 10,
) -> None:

    print("\n===================================================")
    print("🚨 FRAUD ATTACK SIMULATION REPORT")
    print("===================================================\n")

    print(
        f"Fraud Threshold: "
        f"{summary['threshold']}"
    )

    # =====================================================
    # Per Attack Analysis
    # =====================================================

    for attack in summary["per_attack"]:

        name = attack["attack_name"]

        description = attack["attack_description"]

        counts = attack["counts"]

        cm = attack["confusion_matrix"]

        print("\n---------------------------------------------------")
        print(f"⚔️ Attack Scenario: {name}")
        print("---------------------------------------------------")

        print("\n📖 Attack Description:")
        print(description)

        print("\n📊 Metrics:")

        print(f"TP : {counts['TP']}")
        print(f"TN : {counts['TN']}")
        print(f"FP : {counts['FP']}")
        print(f"FN : {counts['FN']}")

        print("\n🧾 Confusion Matrix:")
        print(cm)

    # =====================================================
    # Overall System Performance
    # =====================================================

    overall = summary["overall"]

    counts = overall["counts"]

    print("\n===================================================")
    print("📈 OVERALL SYSTEM PERFORMANCE")
    print("===================================================\n")

    print(
        f"Total Transactions: "
        f"{overall['total_transactions']}"
    )

    print(f"TP : {counts['TP']}")
    print(f"TN : {counts['TN']}")
    print(f"FP : {counts['FP']}")
    print(f"FN : {counts['FN']}")

    print("\n🧾 Overall Confusion Matrix:")
    print(overall["confusion_matrix"])

    # =====================================================
    # Sample Detailed Explanations
    # =====================================================

    print("\n===================================================")
    print("🧠 SAMPLE FRAUD ANALYSIS")
    print("===================================================\n")

    shown = 0

    for row in summary["flat_results"]:

        if shown >= preview_rows:
            break

        print("\n---------------------------------------------------")

        print(f"Scenario       : {row['attack_type']}")
        print(f"Probability    : {round(row['probability'], 4)}")
        print(f"Decision       : {row['decision']}")

        print(
            f"Ground Truth   : {row['true_fraud']} | "
            f"Predicted      : {row['pred_fraud']}"
        )

        explanation = row["explanation"]

        if isinstance(explanation, dict):

            print("\n📖 Attack Analysis:")
            print(explanation.get("attack_analysis"))

            print("\n🧠 Prediction Summary:")
            print(explanation.get("prediction_summary"))

            truth_analysis = explanation.get(
                "truth_analysis"
            )

            if truth_analysis:

                print("\n✅ Truth Evaluation:")
                print(truth_analysis)

            features = explanation.get(
                "feature_evidence"
            )

            if features:

                print("\n🔍 Feature Evidence:")

                for feat in features:

                    interpretation = feat.get(
                        "interpretation"
                    )

                    print(f"- {interpretation}")

        else:

            print("\nExplanation:")
            print(explanation)

        shown += 1