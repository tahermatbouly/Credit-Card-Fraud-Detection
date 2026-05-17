"""
Fraud Probability Audit / Explanation Logger

This module explains:
1) where the fraud probability came from
2) how the final decision was made: ALLOW / REVIEW / BLOCK
3) which feature values most influenced the model output
4) saves these reasons to CSV/JSON so the dashboard can display them
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


LABEL_COLUMNS = {"isFraud", "isFlaggedFraud"}


FEATURE_HINTS = {
    "step": "Transaction time step in the simulation.",
    "amount": "Transaction amount. High values can increase fraud risk.",
    "oldbalanceOrg": "Sender balance before the transaction.",
    "newbalanceOrig": "Sender balance after the transaction.",
    "oldbalanceDest": "Receiver balance before the transaction.",
    "newbalanceDest": "Receiver balance after the transaction.",
    "balance_diff_orig": "Change in sender balance. Large changes can indicate draining.",
    "balance_diff_dest": "Change in receiver balance. Sudden destination balance movement can be suspicious.",
    "type_CASH_OUT": "Cash-out transaction type. This is usually higher risk.",
    "type_DEBIT": "Debit transaction type.",
    "type_PAYMENT": "Payment transaction type. Usually lower risk than transfer/cash-out.",
    "type_TRANSFER": "Transfer transaction type. This is usually higher risk.",
    "is_high_amount": "Flag for unusually high transaction amount.",
    "amount_log": "Log-transformed amount used to help the model handle large values.",
    "error_orig": "Accounting inconsistency for sender balance.",
    "error_dest": "Accounting inconsistency for destination balance.",
    "high_risk_transaction": "Composite flag for risky transaction behavior.",
}


SCENARIO_HINTS = {
    "01_normal_baseline": "Original transactions without attack mutation. Used as a reference baseline.",
    "02_random_noise_flood": "Small random changes are added to numeric values to test model stability.",
    "03_micro_transaction_probe": "Very small transaction amounts simulate an attacker testing if the account works.",
    "04_high_value_flood": "Transaction amounts are increased heavily to simulate high-value fraud attempts.",
    "05_rapid_fire_flood": "Transactions are repeated many times to simulate a fast automated transaction flood.",
    "06_transfer_cashout_attack": "Transaction type is forced to high-risk types such as TRANSFER or CASH_OUT.",
    "07_balance_draining_attack": "Amount is made close to the sender balance to simulate account draining.",
    "08_mixed_obfuscation_attack": "Multiple suspicious changes are combined: high amount, risky type, balance draining, and noise.",
    "ALL": "All attack scenarios are being evaluated.",
}


def clean_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return columns that should be used as model inputs."""
    return [str(c) for c in columns if str(c) not in LABEL_COLUMNS]


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Align X to the exact feature order used during training."""
    return X.reindex(columns=feature_names, fill_value=0)


def predict_fraud_probability(model, X: pd.DataFrame, scaler=None) -> np.ndarray:
    """
    Calculate P(fraud) using model.predict_proba.

    predict_proba(X) returns two probabilities:
        column 0 = P(legitimate)
        column 1 = P(fraud)
    """
    if scaler is not None:
        X_for_model = scaler.transform(X.values)
    else:
        X_for_model = X

    probs = model.predict_proba(X_for_model)

    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("model.predict_proba must return probabilities for at least two classes.")

    return probs[:, 1]


def decision_from_probability(
    probability: float,
    *,
    alert_threshold: float = 0.12,
    block_threshold: float = 0.30,
) -> tuple[str, str, int]:
    """Convert fraud probability into ALLOW / REVIEW / BLOCK."""
    p = float(probability)

    if p >= block_threshold:
        return "BLOCK", "BLOCK", 1

    if p >= alert_threshold:
        return "REVIEW", "ALERT", 0

    return "ALLOW", "LOG", 0


def get_model_feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Get global feature importance from the trained model."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel()).astype(float)
    else:
        values = np.ones(len(feature_names), dtype=float)

    if len(values) != len(feature_names):
        values = np.resize(values, len(feature_names))

    total = float(np.sum(values))
    if total > 0:
        values = values / total

    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def _feature_reason(feature: str, value: float) -> str:
    """Human-friendly reason line for one feature value."""
    hint = FEATURE_HINTS.get(feature, "Model input feature.")

    try:
        numeric_value = float(value)
    except Exception:
        numeric_value = value

    return f"{feature} = {numeric_value} | {hint}"


def top_feature_reasons(
    row: pd.Series,
    importances: pd.Series,
    *,
    top_k: int = 5,
) -> tuple[str, str]:
    """Return top feature names and detailed feature explanation."""
    reasons = []
    used_features = []

    for feature in importances.head(top_k).index:
        value = row.get(feature, 0)
        used_features.append(str(feature))
        reasons.append(_feature_reason(str(feature), value))

    return ", ".join(used_features), "\n".join(reasons)


def probability_formula_text(
    probability: float,
    *,
    alert_threshold: float,
    block_threshold: float,
) -> str:
    """Explain the threshold calculation in plain language."""
    p = float(probability)

    if p >= block_threshold:
        return (
            f"P(fraud) = {p:.4f}. "
            f"Since {p:.4f} >= block_threshold {block_threshold:.2f}, "
            f"the system returns BLOCK and binary fraud prediction = 1."
        )

    if p >= alert_threshold:
        return (
            f"P(fraud) = {p:.4f}. "
            f"Since alert_threshold {alert_threshold:.2f} <= {p:.4f} < "
            f"block_threshold {block_threshold:.2f}, "
            f"the system returns REVIEW and binary fraud prediction = 0."
        )

    return (
        f"P(fraud) = {p:.4f}. "
        f"Since {p:.4f} < alert_threshold {alert_threshold:.2f}, "
        f"the system returns ALLOW and binary fraud prediction = 0."
    )


def build_detection_reason(
    *,
    scenario_name: str,
    probability: float,
    decision: str,
    action: str,
    feature_reasons: str,
    alert_threshold: float,
    block_threshold: float,
) -> str:
    """Full explanation used by dashboard/export."""
    scenario_text = SCENARIO_HINTS.get(
        scenario_name,
        "Custom attack scenario. The model only sees the modified feature values, not the scenario name.",
    )

    threshold_text = probability_formula_text(
        probability,
        alert_threshold=alert_threshold,
        block_threshold=block_threshold,
    )

    if decision == "BLOCK":
        decision_text = (
            "Fraud was detected because the model probability passed the blocking threshold. "
            "The transaction should be stopped immediately."
        )
    elif decision == "REVIEW":
        decision_text = (
            "The transaction is suspicious but not risky enough for automatic blocking. "
            "It should be sent to manual review or an alert queue."
        )
    else:
        decision_text = (
            "The transaction risk is below the alert threshold, so it is allowed and only logged."
        )

    return (
        f"Scenario pattern: {scenario_text}\n\n"
        f"Probability source: The trained ML model calculated this using model.predict_proba(). "
        f"The fraud probability is the second class probability: predict_proba(X)[:, 1].\n\n"
        f"Threshold calculation: {threshold_text}\n\n"
        f"Decision reason: {decision_text}\n\n"
        f"Top feature evidence:\n{feature_reasons}"
    )


def audit_predictions(
    *,
    model,
    X: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    scenario_name: str = "custom",
    feature_names: Optional[list[str]] = None,
    scaler=None,
    alert_threshold: float = 0.12,
    block_threshold: float = 0.30,
    max_rows: Optional[int] = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """Score transactions and return probabilities, decisions, and reasons."""
    X = X.copy()

    if feature_names is None:
        if hasattr(model, "feature_names_in_"):
            feature_names = [str(c) for c in model.feature_names_in_]
        else:
            feature_names = clean_feature_columns(X.columns)

    X_aligned = align_features(X, feature_names)

    if max_rows is not None:
        X_aligned = X_aligned.head(int(max_rows)).copy()
        if y_true is not None:
            y_true = y_true.reset_index(drop=True).head(int(max_rows))

    probabilities = predict_fraud_probability(model, X_aligned, scaler=scaler)
    importances = get_model_feature_importance(model, feature_names)

    audit_rows = []

    for i, (_, row) in enumerate(X_aligned.iterrows()):
        probability = float(probabilities[i])

        decision, action, pred_fraud = decision_from_probability(
            probability,
            alert_threshold=alert_threshold,
            block_threshold=block_threshold,
        )

        top_features, feature_reasons = top_feature_reasons(
            row,
            importances,
            top_k=top_k,
        )

        reason = build_detection_reason(
            scenario_name=scenario_name,
            probability=probability,
            decision=decision,
            action=action,
            feature_reasons=feature_reasons,
            alert_threshold=alert_threshold,
            block_threshold=block_threshold,
        )

        true_label = None
        result_type = "UNKNOWN"

        if y_true is not None:
            true_label = int(y_true.iloc[i])

            if true_label == 1 and pred_fraud == 1:
                result_type = "TP"
            elif true_label == 0 and pred_fraud == 0:
                result_type = "TN"
            elif true_label == 0 and pred_fraud == 1:
                result_type = "FP"
            elif true_label == 1 and pred_fraud == 0:
                result_type = "FN"

        audit_rows.append(
            {
                "transaction_id": i,
                "scenario": scenario_name,
                "fraud_probability": round(probability, 6),
                "alert_threshold": float(alert_threshold),
                "block_threshold": float(block_threshold),
                "decision": decision,
                "action": action,
                "pred_fraud": int(pred_fraud),
                "true_fraud": true_label,
                "result_type": result_type,
                "top_features": top_features,
                "probability_calculation": probability_formula_text(
                    probability,
                    alert_threshold=alert_threshold,
                    block_threshold=block_threshold,
                ),
                "reason": reason,
            }
        )

    return pd.DataFrame(audit_rows)


def save_audit_logs(
    audit_df: pd.DataFrame,
    *,
    output_dir: str | Path = "output/simulation",
    base_name: str = "fraud_probability_audit",
) -> tuple[Path, Path]:
    """Save audit dataframe to CSV and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"

    audit_df.to_csv(csv_path, index=False)

    records = audit_df.to_dict(orient="records")
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return csv_path, json_path
