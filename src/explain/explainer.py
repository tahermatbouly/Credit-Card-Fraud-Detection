from __future__ import annotations

from typing import Any

import numpy as np


# =========================================================
# Attack Scenario Documentation
# =========================================================

_ATTACK_SCENARIO_DETAIL: dict[str, str] = {
    "normal_baseline": (
        "Baseline traffic simulation using real processed transactions "
        "without any artificial manipulation. "
        "This scenario measures how the fraud detection system behaves "
        "under realistic production-like traffic."
    ),

    "high_value_flood": (
        "High-value attack simulation where transaction amounts are "
        "artificially amplified. This attack attempts to trigger abnormal "
        "financial behavior by creating unusually large transfers. "
        "The goal is to evaluate whether the model can identify suspicious "
        "large-value movements while avoiding unnecessary false positives."
    ),

    "rapid_fire_flood": (
        "Rapid-fire transaction flood simulation. "
        "The same or similar transaction patterns are repeatedly submitted "
        "at high frequency to mimic transaction spamming or bot-driven abuse. "
        "This evaluates system stability, throughput behavior, and fraud "
        "detection consistency under heavy transaction volume."
    ),

    "random_noise_flood": (
        "Noise injection attack where numerical transaction values are "
        "randomly perturbed using Gaussian-style multiplicative noise. "
        "This tests model robustness against unstable measurements, "
        "sensor noise, corrupted data, or adversarial perturbations."
    ),
}


# =========================================================
# Feature Explanations
# =========================================================

_FEATURE_HINTS: dict[str, str] = {
    "step": "time-based transaction step or temporal indicator",
    "amount": "transaction monetary amount",
    "oldbalanceOrg": "sender balance before transaction",
    "newbalanceOrig": "sender balance after transaction",
    "oldbalanceDest": "receiver balance before transfer",
    "newbalanceDest": "receiver balance after transfer",
    "balance_diff_orig": "difference in sender account balance",
    "balance_diff_dest": "difference in receiver account balance",
    "type_CASH_OUT": "cash-out transaction indicator",
    "type_DEBIT": "debit transaction indicator",
    "type_PAYMENT": "payment transaction indicator",
    "type_TRANSFER": "transfer transaction indicator",
    "is_high_amount": "engineered feature detecting unusually high amounts",
    "amount_log": "log-scaled amount feature",
    "error_orig": "sender-side accounting inconsistency",
    "error_dest": "receiver-side accounting inconsistency",
    "high_risk_transaction": "engineered high-risk transaction marker",
}


# =========================================================
# Fraud Explainer
# =========================================================

class FraudExplainer:
    """
    Generates detailed explanations for:
    - fraud predictions
    - attack scenarios
    - model decisions
    - feature importance behavior
    """

    # =====================================================
    # Attack Scenario Explanation
    # =====================================================

    @staticmethod
    def describe_attack_scenario(attack_name: str) -> str:

        if attack_name not in _ATTACK_SCENARIO_DETAIL:

            return (
                f"Attack scenario '{attack_name}' is not officially documented. "
                "This may represent a custom synthetic fraud pattern or an "
                "experimental attack configuration."
            )

        return _ATTACK_SCENARIO_DETAIL[attack_name]

    # =====================================================
    # Feature Importance Extraction
    # =====================================================

    def _feature_importances(
        self,
        model: Any,
        feature_names: list[str]
    ) -> tuple[np.ndarray, str]:

        # Tree-based models
        if hasattr(model, "feature_importances_"):

            v = np.asarray(model.feature_importances_, dtype=float)

            return v, "tree_feature_importance"

        # Linear models
        if hasattr(model, "coef_"):

            v = np.asarray(np.abs(model.coef_)).ravel()

            if v.size != len(feature_names):

                v = np.pad(
                    v,
                    (0, max(0, len(feature_names) - v.size)),
                    constant_values=0.0,
                )[: len(feature_names)]

            return v, "linear_coefficients"

        return np.zeros(len(feature_names)), "unavailable"

    # =====================================================
    # Feature Annotation
    # =====================================================

    def _annotate_feature_rows(
        self,
        rows: list[dict[str, Any]],
        attribution_source: str,
    ) -> list[dict[str, Any]]:

        for row in rows:

            fname = str(row.get("feature"))

            hint = _FEATURE_HINTS.get(
                fname,
                "custom or derived feature"
            )

            row["role"] = hint
            row["attribution_source"] = attribution_source

            val = row.get("value")
            imp = row.get("importance", 0.0)

            if val is None:

                row["interpretation"] = (
                    f"Feature '{fname}' contributes importance score "
                    f"{imp:.6f} according to the model."
                )

            else:

                row["interpretation"] = (
                    f"Feature '{fname}' ({hint}) has value {val} "
                    f"with importance score {imp:.6f}. "
                    f"Higher importance indicates stronger influence "
                    f"on fraud prediction behavior."
                )

        return rows

    # =====================================================
    # Per-Transaction Feature Analysis
    # =====================================================

    def explain_tree_model(
        self,
        model: Any,
        feature_names: Any,
        sample: np.ndarray,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        names = list(feature_names)

        importances, source = self._feature_importances(
            model,
            names
        )

        if source == "unavailable" or not np.any(importances):

            return [
                {
                    "feature": "unknown",
                    "importance": 0.0,
                    "interpretation": (
                        "The current model does not expose "
                        "feature importance information."
                    )
                }
            ]

        top_indices = np.argsort(importances)[::-1][:top_k]

        sample_flat = np.asarray(sample).ravel()

        explanation = []

        for idx in top_indices:

            idx = int(idx)

            value = None

            if idx < sample_flat.size:

                try:
                    value = sample_flat[idx].item()

                except Exception:
                    value = str(sample_flat[idx])

            explanation.append({
                "feature": names[idx],
                "importance": float(importances[idx]),
                "value": value,
            })

        return self._annotate_feature_rows(
            explanation,
            source
        )

    # =====================================================
    # Decision Narrative
    # =====================================================

    def _decision_narrative(
        self,
        decision: str,
        probability: float,
        block_threshold: float,
        review_threshold: float | None,
    ) -> str:

        p = round(probability, 4)

        if decision == "BLOCK":

            return (
                f"The fraud probability reached {p}, exceeding the "
                f"BLOCK threshold ({block_threshold}). "
                "The system classified this transaction as high-risk "
                "and recommended immediate blocking."
            )

        if decision == "REVIEW" and review_threshold is not None:

            return (
                f"The fraud probability reached {p}, which falls "
                f"between REVIEW ({review_threshold}) and BLOCK "
                f"({block_threshold}) thresholds. "
                "The system recommends manual inspection or "
                "additional verification."
            )

        return (
            f"The fraud probability was {p}, remaining below the "
            f"BLOCK threshold ({block_threshold}). "
            "The transaction is considered low-risk and allowed."
        )

    # =====================================================
    # Ground Truth Evaluation
    # =====================================================

    def _truth_narrative(
        self,
        predicted_label: int,
        true_label: int | None,
    ) -> str | None:

        if true_label is None:
            return None

        if predicted_label == true_label:

            if true_label == 1:

                return (
                    "The transaction was truly fraudulent and the "
                    "model correctly detected it."
                )

            return (
                "The transaction was legitimate and correctly "
                "classified as safe."
            )

        if predicted_label == 1 and true_label == 0:

            return (
                "False Positive: the model incorrectly classified "
                "a legitimate transaction as fraud."
            )

        return (
            "False Negative: the transaction was fraudulent but "
            "the model failed to detect it."
        )

    # =====================================================
    # Full Fraud Prediction Explanation
    # =====================================================

    def explain_fraud_prediction(
        self,
        model: Any,
        feature_names: Any,
        sample: np.ndarray,
        *,
        decision: str,
        fraud_probability: float,
        block_threshold: float,
        review_threshold: float | None = None,
        attack_type: str | None = None,
        true_fraud_label: int | None = None,
        pred_fraud_label: int | None = None,
        include_feature_evidence: bool = True,
        top_k: int = 5,
    ) -> dict[str, Any]:

        pred = pred_fraud_label

        if pred is None:

            pred = (
                1
                if fraud_probability >= block_threshold
                else 0
            )

        attack_description = (
            self.describe_attack_scenario(attack_type)
            if attack_type
            else "No attack scenario specified."
        )

        decision_summary = self._decision_narrative(
            decision,
            fraud_probability,
            block_threshold,
            review_threshold,
        )

        truth_summary = self._truth_narrative(
            pred,
            true_fraud_label,
        )

        report = {
            "attack_type": attack_type,
            "attack_analysis": attack_description,
            "decision": decision,
            "fraud_probability": float(fraud_probability),
            "prediction_summary": decision_summary,
            "truth_analysis": truth_summary,
            "feature_evidence": None,
        }

        if include_feature_evidence:

            report["feature_evidence"] = self.explain_tree_model(
                model,
                feature_names,
                sample,
                top_k=top_k,
            )

        return report